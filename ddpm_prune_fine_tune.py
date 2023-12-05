import argparse
import math
import os
import sys
import time
import random
import torch
import torchvision
import torch_pruning as tp
import torch.nn as nn
from lightning.pytorch import seed_everything

from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from torchvision.utils import save_image

from tqdm import tqdm
from diffusers import DDPMPipeline, DDIMPipeline, DDIMScheduler, DDPMScheduler, UNet2DModel
from diffusers.models.resnet import Upsample2D, Downsample2D
from diffusers.utils import make_image_grid
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

from glob import glob
from PIL import Image

import constants

from ddpm_config import DDPMConfig
from utils import (
    ExponentialMovingAverage,
    create_dataloaders,
)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar"],
        default="mnist",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )
    parser.add_argument(
        "--device",
        type=str,
        default='cpu'
    )
    parser.add_argument(
        "--outdir",
        type=str,
        help="output parent directory",
        default=constants.OUTDIR
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

    ## Pruning params

    parser.add_argument(
        "--pruning_ratio",
          type=float,
          default=0.3
    )

    parser.add_argument(
        "--pruner",
        type=str,
        default='magnitude',
        choices=['taylor', 'random', 'magnitude', 'reinit', 'diff-pruning']
    )
    parser.add_argument("--thr", type=float, default=0.05, help="threshold for diff-pruning")

    return parser.parse_args()

def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def main(args):

    # loading images for gradient-based pruning
    batch_size = args.batch_size
    dataset = args.dataset
    outdir = args.outdir
    device = args.device

    seed_everything(args.opt_seed, workers=True)


    if dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
        example_inputs = {'sample': torch.randn(1, 3, 32, 32).to(device), 'timestep': torch.ones((1,)).long().to(device)}

    elif dataset  == "mnist":
        config = {**DDPMConfig.mnist_config}
        example_inputs = {'sample': torch.randn(1, 3, 256, 256).to(device), 'timestep': torch.ones((1,)).long().to(device)}

    (train_dataloader, _ , _) = create_dataloaders(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        excluded_class=None,
        unlearning=False,
        return_excluded=False
    )

    clean_images = next(iter(train_dataloader))
    if isinstance(clean_images, (list, tuple)):
        clean_images = clean_images[0]
    clean_images = clean_images.to(args.device)
    noise = torch.randn(clean_images.shape).to(clean_images.device)

    pre_trained_path = os.path.join(constants.DATASET_DIR, "pretrained_models/cifar")
    # Loading pretrained model
    print("Loading pretrained model from {}".format(pre_trained_path))

    # load model and scheduler

    pipeline = DDPMPipeline.from_pretrained(pre_trained_path)
    pipeline_scheduler = pipeline.scheduler
    model = pipeline.unet.eval()
    model.to(device)

    pruning_params = f"pruner={args.pruner}_pruning_ratio={args.pruning_ratio}_threshold={args.thr}"

    if args.pruning_ratio>0:
        if args.pruner == 'taylor':
            imp = tp.importance.TaylorImportance(multivariable=True) # standard first-order taylor expansion
        elif args.pruner == 'random' or args.pruner=='reinit':
            imp = tp.importance.RandomImportance()
        elif args.pruner == 'magnitude':
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == 'diff-pruning':
            imp = tp.importance.TaylorImportance(multivariable=False) # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}
        #from diffusers.models.attention import
        #for m in model.modules():
        #    if isinstance(m, AttentionBlock):
        #        channel_groups[m.query] = m.num_heads
        #        channel_groups[m.key] = m.num_heads
        #        channel_groups[m.value] = m.num_heads

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

        if args.pruner in ['taylor', 'diff-pruning']:
            loss_max = 0
            print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(1000)):
                timesteps = (step_k*torch.ones((batch_size,), device=clean_images.device)).long()
                noisy_images = pipeline_scheduler.add_noise(clean_images, noise, timesteps)
                model_output = model(noisy_images, timesteps).sample
                loss = nn.functional.mse_loss(model_output, noise)
                loss.backward()

                if args.pruner=='diff-pruning':
                    if loss>loss_max: loss_max = loss
                    if loss<loss_max * args.thr: break # taylor expansion over pruned timesteps ( L_t / L_max > thr )

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        print(model)
        print("#Params: {:.4f} M => {:.4f} M".format(base_params/1e6, params/1e6))
        print("#MACS: {:.4f} G => {:.4f} G".format(base_macs/1e9, macs/1e9))
        model.zero_grad()
        del pruner

        if args.pruner=='reinit':
            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, 'reset_parameters'):
                        m.reset_parameters()
            reset_parameters(model)

    pipeline_dir = os.path.join(outdir, dataset,f"pruned/pipelines/{pruning_params}")
    os.makedirs(pipeline_dir, exist_ok=True)
    pipeline.save_pretrained(pipeline_dir)


    start_epoch = 0
    global_steps = start_epoch * len(train_dataloader)

    if args.pruning_ratio > 0:
        model_outdir = os.path.join(outdir, dataset, "pruned/models", pruning_params)
        os.makedirs(model_outdir, exist_ok=True)
        torch.save(model,  os.path.join(model_outdir, f"pruned_unet_{global_steps:0>8}.pth"))

    # Sampling images from the pruned model
    pipeline = DDIMPipeline(
        unet = model,
        scheduler = pipeline_scheduler
    )
    with torch.no_grad():
        generator = torch.Generator(device=pipeline.device).manual_seed(args.opt_seed)
        pipeline.to(args.device)
        samples = pipeline(
            batch_size=config["n_samples"],
            num_inference_steps=100,
            generator=generator,
            output_type="numpy"
        ).images

        sample_outdir = os.path.join(
            outdir,
            dataset,
            "pruned",
            "samples",
            pruning_params
        )
        os.makedirs(sample_outdir, exist_ok=True)

        torchvision.utils.save_image(torch.from_numpy(samples).permute([0, 3, 1, 2]), os.path.join(sample_outdir, f"steps_{1:0>8}.png"))


    print("==================== fine-tuning on pruned model ====================")

    ## Set unet dropout rate

    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.1

    epochs = config["epochs"]["retrain"]

    ema_model = EMAModel(
            model.parameters(),
            decay=0.9999,
            use_ema_warmup=False,
            inv_gamma=1.0,
            power=3/4,
            model_cls=UNet2DModel,
            model_config=model.config,
        )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        betas=(0.9, 0.999),
        weight_decay=0 ,
        eps=1e-8
    )

    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps= 0,
        num_training_steps=(len(train_dataloader) * config["epochs"]["retrain"]),
    )

    ema_model.to(device)

    loss_fn = nn.MSELoss(reduction="mean")

    for epoch in range(start_epoch, epochs):

        model.train()
        steps_start_time = time.time()

        for j, (image, _) in enumerate(train_dataloader):
            optimizer.zero_grad()

            image=image.to(device)
            noise=torch.randn_like(image).to(device)
            timesteps = torch.randint(
                low=0,
                high=config["timesteps"],
                size=(len(image)//2 +1, ),  # (len(image),),
                device=image.device
            ).long()
            timesteps = torch.cat([timesteps, pipeline_scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:len(image)]

            noisy_images = pipeline_scheduler.add_noise(image, noise, timesteps)
            eps = model(noisy_images, timesteps).sample

            loss=loss_fn(eps,noise)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()

            ema_model.step(model.parameters())

            if (j + 1) % args.log_freq == 0:
                steps_time = time.time() - steps_start_time
                info = f"Epoch[{epoch + 1}/{epochs}]"
                info += f", Step[{j + 1}/{len(train_dataloader)}]"
                info += f", steps_time: {steps_time:.3f}"
                info += f", loss: {loss.detach().cpu().item():.5f}"
                info += f", lr: {lr_scheduler.get_last_lr()[0]:.6f}"
                print(info, flush=True)
                steps_start_time = time.time()

            global_steps += 1

        # Generate samples for evaluation.
        if (epoch + 1) == 1 or (epoch + 1) % config["sample_freq"]["retrain"] == 0 or (
            epoch + 1
        ) == epochs:
            model.eval()

            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

            sampling_start_time = time.time()

            with torch.no_grad():
                pipeline = DDIMPipeline(
                    unet=model,
                    scheduler=DDIMScheduler(num_train_timesteps=config["timesteps"])
                )

                samples = pipeline(
                    batch_size=config["n_samples"],
                    num_inference_steps=config["timesteps"],
                ).images

            sampling_time = time.time() - sampling_start_time

            print(f", sampling_time: {sampling_time:.3f}" )

            if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
                samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]

            image_grid = make_image_grid(
                samples,
                rows=int(math.sqrt(config["n_samples"])),
                cols=int(math.sqrt(config["n_samples"]))
            )
            image_grid.save(os.path.join(sample_outdir, f"steps_{global_steps:0>8}.png"))

        # Checkpoints for training.
        if  (epoch + 1) % config["ckpt_freq"]["retrain"] == 0 or (epoch + 1) == epochs:

            model.eval()

            torch.save(model, os.path.join(model_outdir, f"unet_steps_{global_steps:0>8}.pt"))

            ema_model.store(model.parameters())
            ema_model.copy_to(model.parameters())

            torch.save(model, os.path.join(model_outdir, f"unet_ema_steps_{global_steps:0>8}.pt"))

            # torch.save(ckpt, ckpt_file)
            print(f"Checkpoint saved at step {global_steps} at {ckpt_file}")


if __name__=="__main__":
    args = parse_args()
    print_args(args)
    main(args)
