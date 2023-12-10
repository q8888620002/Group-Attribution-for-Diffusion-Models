"""Pruning and fine-tuning diffusion models"""
import argparse
import math
import os
import sys
import time

import torch
import torch.nn as nn
import torch_pruning as tp
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    LDMPipeline,
    UNet2DModel,
    VQModel,
)
from diffusers.models.attention import Attention
from diffusers.models.resnet import Downsample2D, Upsample2D
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from lightning.pytorch import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

import constants
import wandb
from ddpm_config import DDPMConfig
from utils import create_dataloaders


def parse_args():
    """Parsing arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba"],
        default="mnist",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )

    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )

    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=20,
    )

    # Pruning params

    parser.add_argument("--pruning_ratio", type=float, default=0.3)

    parser.add_argument(
        "--pruner",
        type=str,
        default="magnitude",
        choices=["taylor", "random", "magnitude", "reinit", "diff-pruning"],
    )
    parser.add_argument(
        "--thr", type=float, default=0.05, help="threshold for diff-pruning"
    )

    # fine-tuning params

    parser.add_argument(
        "grad_accum",
        default=False,
        action="store_true",
        help="whether to use gradient accumulation.",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="The dropout rate for fine-tuning."
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            "The scheduler type to use."
            'Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )

    parser.add_argument("--num_inference_steps", type=int, default=100)

    parser.add_argument("--num_train_steps", type=int, default=1000)

    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.",
    )

    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="The beta1 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="The beta2 parameter for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.0,
        help="Weight decay magnitude for the Adam optimizer.",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="Epsilon value for the Adam optimizer.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="The inverse gamma value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="The power value for the EMA decay.",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="The maximum decay magnitude for EMA.",
    )

    return parser.parse_args()


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def main(args):
    """Main function for pruning and fine-tuning."""
    # loading images for gradient-based pruning
    batch_size = args.batch_size
    dataset = args.dataset
    outdir = args.outdir
    device = args.device

    seed_everything(args.opt_seed, workers=True)

    if dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }

    elif dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 256, 256).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif dataset == "celeba":
        config = {**DDPMConfig.celeb_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 256, 256).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }

    (train_dataloader, _) = create_dataloaders(
        dataset_name=config["dataset"],
        batch_size=batch_size,
        excluded_class=None,
        unlearning=False,
        return_excluded=False,
    )

    clean_images = next(iter(train_dataloader))
    if isinstance(clean_images, (list, tuple)):
        clean_images = clean_images[0]
    clean_images = clean_images.to(args.device)
    noise = torch.randn(clean_images.shape).to(clean_images.device)

    pre_trained_path = os.path.join(outdir, f"pretrained_models/{args.dataset}")
    # Loading pretrained model
    print("Loading pretrained model from {}".format(pre_trained_path))

    # load model and scheduler
    if args.dataset == "cifar":

        pipeline = DDPMPipeline.from_pretrained(pre_trained_path)
        pipeline_scheduler = pipeline.scheduler
        model = pipeline.unet.eval()
        model.to(device)

    elif args.dataset == "celeba":

        model_id = "CompVis/ldm-celebahq-256"
        model = UNet2DModel.from_pretrained(model_id, subfolder="unet")
        vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae")
        pipeline_scheduler = DDIMScheduler.from_config(model_id, subfolder="scheduler")

        model.eval()
        model.to(device)
        vqvae.to(device)

    pruning_params = (
        f"pruner={args.pruner}_pruning_ratio={args.pruning_ratio}_threshold={args.thr}"
    )

    if args.pruning_ratio > 0:
        if args.pruner == "taylor":
            imp = tp.importance.TaylorImportance(
                multivariable=True
            )  # standard first-order taylor expansion
        elif args.pruner == "random" or args.pruner == "reinit":
            imp = tp.importance.RandomImportance()
        elif args.pruner == "magnitude":
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == "diff-pruning":
            imp = tp.importance.TaylorImportance(
                multivariable=False
            )  # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}

        if args.dataset == "celeba":

            # Prunig attention for celeba

            for m in model.modules():
                if isinstance(m, Attention):
                    channel_groups[m.to_q] = m.heads
                    channel_groups[m.to_k] = m.heads
                    channel_groups[m.to_v] = m.heads

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()

        if args.pruner in ["taylor", "diff-pruning"]:
            loss_max = 0
            print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(pipeline_scheduler.num_train_timesteps)):
                timesteps = (
                    step_k * torch.ones((batch_size,), device=clean_images.device)
                ).long()
                noisy_images = pipeline_scheduler.add_noise(
                    clean_images, noise, timesteps
                )
                model_output = model(noisy_images, timesteps).sample
                loss = nn.functional.mse_loss(model_output, noise)
                loss.backward()

                if args.pruner == "diff-pruning":
                    if loss > loss_max:
                        loss_max = loss
                    if loss < loss_max * args.thr:
                        # taylor expansion over pruned timesteps ( L_t / L_max > thr )
                        break

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params / 1e6, params / 1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs / 1e9, macs / 1e9))
        model.zero_grad()
        del pruner

        if args.pruner == "reinit":

            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()

            reset_parameters(model)

    if args.dataset == "cifar":
        pipeline.unet = model
        pipeline.to(device)

    elif args.dataset == "celeba":
        for m in vqvae.modules():
            m.requires_grad = False

        pipeline = LDMPipeline(
            unet=model,
            vqvae=vqvae,
            scheduler=pipeline_scheduler,
        ).to(device)

    pipeline_dir = os.path.join(outdir, dataset, f"pruned/pipelines/{pruning_params}")
    os.makedirs(pipeline_dir, exist_ok=True)
    pipeline.save_pretrained(pipeline_dir)

    start_epoch = 0
    global_steps = start_epoch * len(train_dataloader)

    if args.pruning_ratio > 0:
        model_outdir = os.path.join(outdir, dataset, "pruned/models", pruning_params)
        os.makedirs(model_outdir, exist_ok=True)
        torch.save(
            model, os.path.join(model_outdir, f"pruned_unet_{global_steps:0>8}.pth")
        )

    print("==================== fine-tuning on pruned model ====================")

    # Set unet dropout rate

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = args.dropout

    epochs = config["epochs"]["retrain"]

    ema_model = EMAModel(
        model.parameters(),
        decay=args.ema_max_decay,
        use_ema_warmup=False,
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        model_cls=UNet2DModel,
        model_config=model.config,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * epochs),
    )
    loss_fn = nn.MSELoss(reduction="mean")


    wandb.init(
        project="Data Shapley for Diffusion",
        notes=f"Experiment for pruning and fine-tuning.",
        tags=[f"pruning and fine-tuning"],
        config={
            "epochs": epochs,
            "batch_size": config["batch_size"],
            "model": model.config._class_name,
        },
    )

    ema_model.to(device)
    num_accumulation_steps = 64 // args.batch_size

    for epoch in range(start_epoch, epochs):

        steps_start_time = time.time()

        for j, (image, _) in enumerate(train_dataloader):

            model.train()

            image = image.to(device)

            if isinstance(pipeline, LDMPipeline):
                image = vqvae.encode(image, False)[0]

            noise = torch.randn_like(image).to(device)
            timesteps = torch.randint(
                low=0,
                high=args.num_train_steps,
                size=(len(image) // 2 + 1,),  # (len(image),),
                device=image.device,
            ).long()
            timesteps = torch.cat(
                [
                    timesteps,
                    pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
                ],
                dim=0,
            )[: len(image)]

            noisy_images = pipeline_scheduler.add_noise(image, noise, timesteps)
            eps = model(noisy_images, timesteps).sample

            loss = loss_fn(eps, noise)

            if args.grad_accum:
                # Gradient accumulation
                loss = loss / num_accumulation_steps
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                if ((j + 1) % num_accumulation_steps == 0) or (
                    j + 1 == len(train_dataloader)
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    ema_model.step(model.parameters())

                    grads = [
                        param.grad.detach().flatten()
                        for param in model.parameters()
                        if param.grad is not None
                    ]
                    optimizer.zero_grad()

                    grad_norm = torch.cat(grads).norm()

                    params = [
                        param.data.detach().flatten()
                        for param in model.parameters()
                        if param.data is not None
                    ]
                    params_norm = torch.cat(params).norm()

            else:

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                ema_model.step(model.parameters())

                grads = [
                    param.grad.detach().flatten()
                    for param in model.parameters()
                    if param.grad is not None
                ]
                optimizer.zero_grad()

                grad_norm = torch.cat(grads).norm()

                params = [
                    param.data.detach().flatten()
                    for param in model.parameters()
                    if param.data is not None
                ]
                params_norm = torch.cat(params).norm()

            if (j + 1) % args.log_freq == 0:
                steps_time = time.time() - steps_start_time
                info = f"Epoch[{epoch + 1}/{epochs}]"
                info += f", Step[{j + 1}/{len(train_dataloader)}]"
                info += f", steps_time: {steps_time:.3f}"
                info += f", loss: {loss.detach().cpu().item():.5f}"
                info += f", gradient norms: {grad_norm:.5f}"
                info += f", parameters norms: {params_norm:.5f}"
                info += f", lr: {lr_scheduler.get_last_lr()[0]:.6f}"
                print(info, flush=True)
                steps_start_time = time.time()

                wandb.log(
                    {
                        "Epoch": (epoch + 1),
                        "loss": loss.detach().cpu().item(),
                        "steps_time": steps_time,
                        "gradient norms": grad_norm,
                        "parameters norms": params_norm,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                )
            global_steps += 1

        # Generate samples for evaluation.
        if (
            (epoch + 1) == 1
            or (epoch + 1) % config["sample_freq"]["retrain"] == 0
            or (epoch + 1) == epochs
        ):

            model.eval()

            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

            sampling_start_time = time.time()

            with torch.no_grad():
                if args.dataset == "celeba":
                    pipeline = LDMPipeline(
                        unet=model,
                        vqvae=vqvae,
                        scheduler=pipeline_scheduler,
                    ).to(device)
                else:
                    pipeline = DDIMPipeline(
                        unet=model,
                        scheduler=DDIMScheduler(
                            num_train_timesteps=args.num_inference_steps
                        ),
                    )

                samples = pipeline(
                    batch_size=config["n_samples"],
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                ).images

            sampling_time = time.time() - sampling_start_time

            print(f", sampling_time: {sampling_time:.3f}")

            if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
                samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]

            sample_outdir = os.path.join(
                outdir, dataset, "pruned", "samples", pruning_params
            )
            os.makedirs(sample_outdir, exist_ok=True)

            save_image(
                torch.from_numpy(samples).permute([0, 3, 1, 2]),
                os.path.join(sample_outdir, f"steps_{global_steps:0>8}.png"),
                nrow=int(math.sqrt(config["n_samples"])),
            )

            ema_model.restore(model.parameters())

        # Checkpoints for training.
        if (epoch + 1) % config["ckpt_freq"]["retrain"] == 0 or (epoch + 1) == epochs:

            model.eval()
            model.zero_grad()
            torch.save(
                model, os.path.join(model_outdir, f"unet_steps_{global_steps:0>8}.pt")
            )

            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

            torch.save(
                model,
                os.path.join(model_outdir, f"unet_ema_steps_{global_steps:0>8}.pt"),
            )

            # torch.save(ckpt, ckpt_file)
            print(f"Checkpoint saved at step {global_steps}")

            ema_model.restore(model.parameters())

    # Save updated pipeline
    pipeline_dir = os.path.join(outdir, dataset, f"pruned/pipelines/{pruning_params}")
    os.makedirs(pipeline_dir, exist_ok=True)
    pipeline.save_pretrained(pipeline_dir)


if __name__ == "__main__":
    """ """
    args = parse_args()
    print_args(args)
    main(args)
