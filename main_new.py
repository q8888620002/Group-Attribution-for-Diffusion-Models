"""Train or perform unlearning on a diffusion model."""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from lightning.pytorch import seed_everything
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

import constants

from ddpm_config import DDPMConfig
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel, DDIMPipeline, DDIMScheduler
from diffusers.utils import make_image_grid
from diffusion.models import CNN
from utils import (
    ExponentialMovingAverage,
    create_dataloaders,
    get_max_step_file,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--load", type=str, help="path for loading pre-trained model", default=None
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
        "--no_clip",
        action="store_true",
        help=(
            "set to normal sampling method without clip x_0 which could yield "
            "unstable samples"
        ),
    )
    parser.add_argument(
        "--excluded_class",
        type=int,
        help="dataset class to unlearn or retrain on",
        default=None,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="unlearning method",
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
        help="whether to resume from the latest model checkpoint",
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
        raise ValueError(
            f"Unknown dataset {config['dataset']}, choose 'cifar' or 'mnist'."
        )
    epochs = config["epochs"][args.method]
    full_epochs = config["epochs"]["retrain"]

    excluded_class = "full" if args.excluded_class is None else args.excluded_class

    model_outdir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        "models",
        f"{excluded_class}",
    )
    os.makedirs(model_outdir, exist_ok=True)

    seed_everything(args.opt_seed, workers=True)


    (train_dataloader, _ , forget_dataloader) = create_dataloaders(
        dataset_name=config["dataset"],
        batch_size=config["batch_size"],
        excluded_class=args.excluded_class,
        unlearning=False,
        return_excluded= args.method == "ga"
    )

    # torchvision ema setting
    # https://github.com/pytorch/vision/blob/main/references/classification/train.py
    # Use the same EMA setting as the fully trained model.
    adjust = 1 * config["batch_size"] * config["model_ema_steps"] / full_epochs
    alpha = 1.0 - config["model_ema_decay"]
    alpha = min(1.0, alpha * adjust)

    # Load or resume model if available.

    pipeline = DDPMPipeline(
        unet=UNet2DModel(
            **config["unet_config"]
        ).to(device),
        scheduler=DDPMScheduler(**config["scheduler_config"])
    )

    model = pipeline.unet
    noise_scheduler = pipeline.scheduler

    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    start_epoch = 0

    # Determine the checkpoint file path
    ckpt_file = get_max_step_file(model_outdir) if args.resume else args.load

    if ckpt_file:
        ckpt = torch.load(ckpt_file, map_location=device)
        model.load_state_dict(ckpt["model"])

        if args.resume:
            model_ema.load_state_dict(ckpt["model_ema"])
            start_epoch = ckpt.get("epoch", 0)
            print(f"Model and EMA resumed from {ckpt_file}")

    elif not args.resume and args.load:

        print(f"Checkpoint file not found. Starting fresh with model loaded from {args.load}.")


    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=1e-4)
    scheduler = OneCycleLR(
        optimizer,
        config["lr"],
        total_steps=full_epochs * len(train_dataloader),  # Use the full model setting.
        pct_start=0.25,
        anneal_strategy="cos",
    )

    # Load frozen model

    if args.method == "esd":
        
        pipeline_frozen = DDPMPipeline.from_pretrained(os.path.join(constants.DATASET_DIR, "pretrained_models/cifar"))


    loss_fn = nn.MSELoss(reduction="mean")

    global_steps = start_epoch * len(train_dataloader)

    for epoch in range(start_epoch, epochs):

        model.train()

        steps_start_time = time.time()

        for j, ((image_r, _), (image_f, _)) in enumerate(zip(train_dataloader, forget_dataloader)):
            
            image_r=image_r.to(device)

            noise=torch.randn_like(image_r).to(device)

            timesteps = torch.randint(
                0,
                config["timesteps"],
                (len(image_r),),
                device=image_r.device
            ).long()

            timesteps = torch.cat([timesteps, noise_scheduler.config.num_train_timesteps - timesteps - 1], dim=0)[:len(image_r)]

            noisy_images_r = noise_scheduler.add_noise(image_r, noise, timesteps)

            optimizer.zero_grad()

            eps_r = model(noisy_images_r, timesteps).sample

            loss=loss_fn(eps_r,noise)

            if args.method == "ga":
                loss *= -1.0

            elif args.method == "esd":
            
                image_f=image_f.to(device)
                
                with torch.no_grad():

                    noisy_images_f = noise_scheduler.add_noise(image_f, noise, timesteps)

                    frozen_unet = pipeline_frozen.unet.to(device)

                    eps_r_frozen = frozen_unet(noisy_images_r, timesteps).sample
                    eps_f_frozen = frozen_unet(noisy_images_f, timesteps).sample

                loss += loss_fn(eps_r, (eps_r_frozen - 1e-4*eps_f_frozen))


            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            if global_steps % config["model_ema_steps"] == 0:
                model_ema.update_parameters(model)

            if (j + 1) % args.log_freq == 0:
                steps_time = time.time() - steps_start_time
                info = f"Epoch[{epoch + 1}/{epochs}]"
                info += f", Step[{j + 1}/{len(train_dataloader)}]"
                info += f", steps_time: {steps_time:.3f}"
                info += f", loss: {loss.detach().cpu().item():.5f}"
                info += f", lr: {scheduler.get_last_lr()[0]:.6f}"
                print(info, flush=True)
                steps_start_time = time.time()

            global_steps += 1

        # Generate samples for evaluation.
        if (epoch + 1) == 1 or (epoch + 1) % config["sample_freq"][args.method] == 0 or (
            epoch + 1
        ) == epochs:
            model_ema.eval()

            sampling_start_time = time.time()

            with torch.no_grad():
                pipeline = DDPMPipeline(
                    unet=model_ema.module,
                    scheduler=DDPMScheduler(**config["scheduler_config"])
                )
                samples = pipeline(
                    batch_size=config["n_samples"],
                    num_inference_steps=config["timesteps"],
                ).images

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
                info_dict["lr"] = f"{scheduler.get_last_lr()[0]:.6f}"
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

            # Rescale images from [-1, 1] to [0, 1] and save

            if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
                samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]


            image_grid = make_image_grid(
                samples,
                rows=int(math.sqrt(config["n_samples"])),
                cols=int(math.sqrt(config["n_samples"]))
            )
            image_grid.save(os.path.join(sample_outdir, f"steps_{global_steps:0>8}.png"))

        # Checkpoints for training.
        if  (epoch + 1) % config["ckpt_freq"][args.method] == 0 or (epoch + 1) == epochs:

            ckpt = {
                "model": model.state_dict(),
                "model_ema": model_ema.state_dict(),
                "epoch": epoch + 1,
            }

            ckpt_file = os.path.join(model_outdir, f"steps_{global_steps:0>8}.pt")
            torch.save(ckpt, ckpt_file)
            print(f"Checkpoint saved at step {global_steps} at {ckpt_file}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
