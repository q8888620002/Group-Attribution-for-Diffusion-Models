"""Train or perform unlearning on a diffusion model."""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    UNet2DModel,
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from lightning.pytorch import seed_everything
from torchvision.utils import save_image

import constants
import wandb
from ddpm_config import DDPMConfig
from diffusion.models import CNN
from utils import create_dataloaders, get_max_steps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--load", type=str, help="path for loading pre-trained model", default=None
    )

    parser.add_argument(
        "--init",
        action="store_true",
        help="whether to initialize with a new model.",
        default=False,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar"],
        default="mnist",
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=20,
    )
    parser.add_argument(
        "--excluded_class",
        type=int,
        help="dataset class to exclude for class-wise data removal",
        default=None,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "gd", "ga", "esd"],
        required=True,
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--device", type=str, help="device of training", default="cuda:0"
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--db", type=str, help="database file for storing results", default=None
    )
    parser.add_argument(
        "--exp_name", type=str, help="experiment name in the database", default=None
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="whether to resume from the latest available model checkpoint",
    )

    # Training and fine-tuning parameters.
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        choices=[
            "constant",
            "constant_with_warmup",
            "cosine",
            "cosine_with_restarts",
            "linear",
            "polynomial",
        ],
        help="learning rate scheduler to use",
    )
    parser.add_argument(
        "--lr_warmup_steps",
        type=int,
        default=0,
        help="number of warmup steps in the learning rate scheduler",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        help="beta1 parameter in Adam optimizer",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.999,
        help="beta2 parameter in Adam optimizer",
    )
    parser.add_argument(
        "--adam_weight_decay",
        type=float,
        default=0.0,
        help="weight decay magnitude in Adam optimizer",
    )
    parser.add_argument(
        "--adam_epsilon",
        type=float,
        default=1e-08,
        help="epsilon value in Adam optimizer",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="inverse gamma value for EMA decay",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="power value for EMA decay",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="maximum decay magnitude EMA",
    )

    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of diffusion steps for generating images",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1000,
        help="number of diffusion steps during training",
    )

    return parser.parse_args()


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def main(args):
    """Main function for training or unlearning."""
    device = args.device

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
        classifier = CNN().to(device)
        classifier.load_state_dict(
            torch.load("eval/models/epochs=10_cnn_weights.pt", map_location=device)
        )
        classifier.eval()
    else:
        raise ValueError(f"dataset={args.dataset} is not one of ['cifar', 'mnist']")

    excluded_class = "full" if args.excluded_class is None else args.excluded_class
    model_outdir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        "models",
        f"{excluded_class}",
    )
    os.makedirs(model_outdir, exist_ok=True)

    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    (train_dataloader, forget_dataloader) = create_dataloaders(
        dataset_name=config["dataset"],
        batch_size=args.batch_size,
        excluded_class=args.excluded_class,
        unlearning=(args.method != "retrain"),
        return_excluded=(args.method == "ga"),
    )  # TODO: Make sure that this is right.

    if args.method == "retrain":
        forget_dataloader = train_dataloader

    start_epoch = 0
    epochs = config["epochs"][args.method]
    global_steps = start_epoch * len(train_dataloader)

    pretrained_steps = get_max_steps(args.load) if args.load else None

    if pretrained_steps is not None:
        # Loading model
        print("Loading pruned/pretrained model from {}".format(args.load))

        unet_out_dir = os.path.join(args.load, f"unet_steps_{pretrained_steps:0>8}.pt")
        unet_ema_out_dir = os.path.join(
            args.load, f"unet_ema_steps_{pretrained_steps:0>8}.pt"
        )

        model = torch.load(unet_out_dir, map_location=device)
        ema_model = torch.load(unet_ema_out_dir, map_location=device)

        ema_model = EMAModel(
            ema_model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

        global_steps = pretrained_steps

        if args.init:

            print(
                "initialized a new model from pruned/pretrained at {}.".format(
                    args.load
                )
            )

            for m in model.modules():
                if hasattr(m, "reset_parameters"):
                    m.reset_parameters()

            ema_model = EMAModel(
                model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=False,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=UNet2DModel,
                model_config=model.config,
            )

    else:
        # initializing standard model from scratch.

        print(f"Initializing model from scratch for {args.dataset}")

        model = UNet2DModel(**config["unet_config"]).to(device)

        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    pipeline = DDPMPipeline(
        unet=model, scheduler=DDPMScheduler(**config["scheduler_config"])
    )

    pipeline_scheduler = pipeline.scheduler

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

    # Load frozen model

    if args.method == "esd":
        pipeline_frozen = DDPMPipeline.from_pretrained(
            os.path.join(args.outdir, "pretrained_models/cifar")
        )
        frozen_unet = pipeline_frozen.unet.to(device)

    for epoch in range(start_epoch, epochs):

        steps_start_time = time.time()

        for j, ((image_r, _), (image_f, _)) in enumerate(
            zip(train_dataloader, forget_dataloader)
        ):

            model.train()
            optimizer.zero_grad()

            image_r = image_r.to(device)
            noise = torch.randn_like(image_r).to(device)
            timesteps = torch.randint(
                0,
                pipeline_scheduler.config.num_train_timesteps,
                (len(image_r) // 2 + 1,),
                device=image_r.device,
            ).long()
            timesteps = torch.cat(
                [
                    timesteps,
                    pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
                ],
                dim=0,
            )[: len(image_r)]

            noisy_images_r = pipeline_scheduler.add_noise(image_r, noise, timesteps)

            eps_r = model(noisy_images_r, timesteps).sample

            loss = loss_fn(eps_r, noise)

            if args.method == "ga":
                loss *= -1.0

            elif args.method == "esd":

                image_f = image_f.to(device)

                with torch.no_grad():

                    noisy_images_f = pipeline_scheduler.add_noise(
                        image_f, noise, timesteps
                    )

                    eps_r_frozen = frozen_unet(noisy_images_r, timesteps).sample
                    eps_f_frozen = frozen_unet(noisy_images_f, timesteps).sample

                loss += loss_fn(eps_r, (eps_r_frozen - 1e-4 * eps_f_frozen))

            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            ema_model.step(model.parameters())

            ## check gradient norm & params norm

            grads = [
                param.grad.detach().flatten()
                for param in model.parameters()
                if param.grad is not None
            ]
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
                        "Epoch": (epoch + 1) / epochs,
                        "loss": loss.detach().cpu().item(),
                        "steps_time": steps_time,
                        "gradient norms": grad_norm,
                        "parameters norms": params_norm,
                        "lr": lr_scheduler.get_last_lr()[0],
                    }
                )
            global_steps += 1

        # Generate samples for evaluation.
        if (epoch + 1) % config["sample_freq"][args.method] == 0 or (
            epoch + 1
        ) == epochs:

            model.eval()

            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

            sampling_start_time = time.time()

            with torch.no_grad():
                pipeline = DDIMPipeline(
                    unet=model,
                    scheduler=DDIMScheduler(num_train_timesteps=args.num_train_steps),
                )

                samples = pipeline(
                    batch_size=config["n_samples"],
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                ).images

                samples = torch.from_numpy(samples).permute([0, 3, 1, 2])

            sampling_time = time.time() - sampling_start_time

            if args.dataset == "mnist" and args.excluded_class is not None:
                with torch.no_grad():
                    outs = classifier(samples)[0]
                    preds = outs.argmax(dim=1)
                    mean_excluded_prob = outs.softmax(dim=1)[
                        :, args.excluded_class
                    ].mean()
                    excluded_prop = ((preds == args.excluded_class) * 1.0).mean()

                info += f", mean_excluded_prob: {mean_excluded_prob:.5f}"
                info += f", excluded_prop: {excluded_prop:.5f}"
                info += f", sampling_time: {sampling_time:.3f}"

                print(info, flush=True)

                info_dict = vars(args)
                info_dict["start_epoch"] = start_epoch
                info_dict["epoch"] = f"{epoch + 1}"
                info_dict["global_steps"] = f"{global_steps}"
                info_dict["loss"] = f"{loss.detach().cpu().item():.5f}"
                info_dict["lr"] = f"{lr_scheduler.get_last_lr()[0]:.6f}"
                info_dict["mean_excluded_prob"] = f"{mean_excluded_prob:.5f}"
                info_dict["exlcuded_prop"] = f"{excluded_prop:.5f}"
                info_dict["steps_time"] = f"{steps_time:.3f}"
                info_dict["sampling_time"] = f"{sampling_time:.3f}"
                if args.db is not None:
                    with open(args.db, "a+") as f:
                        f.write(json.dumps(info_dict) + "\n")
                print(f"Results saved to the database at {args.db}")

            sample_outdir = os.path.join(
                args.outdir,
                args.dataset,
                args.method,
                "samples",
                f"{excluded_class}",
            )
            os.makedirs(sample_outdir, exist_ok=True)

            if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
                samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]

            save_image(
                samples,
                os.path.join(sample_outdir, f"steps_{global_steps:0>8}.png"),
                nrow=int(math.sqrt(config["n_samples"])),
            )

        # Checkpoints for training.
        if (epoch + 1) % config["ckpt_freq"][args.method] == 0 or (epoch + 1) == epochs:

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


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
