import argparse
import itertools
import math
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torchvision.utils import save_image

from ddpm_config import DDPMConfig
from diffusion.diffusions import DDPM

# from lora_diffusion import  inject_trainable_lora_extended, save_lora_weight
from diffusion.models import CNN
from utils import *


def parse_args():

    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")

    # Training params
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--ckpt", type=str, help="define checkpoint path", default="")
    parser.add_argument("--cpu", action="store_true", help="cpu training")
    parser.add_argument("--device", type=str, help="gpu training", default="cuda:0")
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=10,
    )
    parser.add_argument("--dataset", type=str, help="name of the dataset", default="")

    ## Diffusion params

    parser.add_argument(
        "--n_samples",
        type=int,
        help="define sampling amounts after every epoch trained",
        default=36,
    )
    parser.add_argument(
        "--model_base_dim", type=int, help="base dim of Unet", default=64
    )
    parser.add_argument(
        "--timesteps", type=int, help="sampling steps of DDPM", default=1000
    )
    parser.add_argument(
        "--model_ema_steps", type=int, help="ema model evaluation interval", default=10
    )
    parser.add_argument(
        "--model_ema_decay", type=float, help="ema model decay", default=0.995
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )

    ## Loss related params

    parser.add_argument(
        "--loss_type", type=str, help="define loss type", default="type1"
    )
    parser.add_argument("--alpha1", type=float, help="loss params: alpha1", default=1)
    parser.add_argument(
        "--alpha2", type=float, help="loss params: alpha2", default=1e-1
    )
    parser.add_argument(
        "--keep_digits",
        action="store_true",
        help="whether to keep other digits in the remaining dataset",
    )
    parser.add_argument(
        "--weight_reg",
        action="store_true",
        help="whether to use weight as regularization.",
    )
    parser.add_argument(
        "--fine_tune_att",
        action="store_true",
        help="whether to fine tune only attentiona layers.",
    )
    parser.add_argument(
        "--fine_tune_lora", action="store_true", help="whether to fine tune with LORA."
    )

    args = parser.parse_args()

    return args


def load_checkpoint(directory):

    all_files = [
        f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))
    ]
    sorted_files = sorted(all_files, key=lambda f: int(f.split("_")[1].split(".")[0]))
    max_step = int(sorted_files[-1].split("_")[1].split(".")[0])
    checkpoint_path = os.path.join(directory, f"steps_{max_step:0>8}.pt")
    ckpt = torch.load(checkpoint_path)

    return ckpt


def main(args):

    device = args.device

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
    else:
        raise ValueError(
            f"Unknown dataset {config['dataset']}, choose 'cifar' or 'mnist'."
        )

    ## loss params

    alpha1 = args.alpha1
    alpha2 = args.alpha2

    # torchvision ema setting - https://github.com/pytorch/vision/blob/main/references/classification/train.py#

    adjust = 1 * args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)

    ckpt = torch.load(config["trained_model"])

    # ckpt = load_checkpoint(f"results/{args.dataset}/retrain/models/full/")

    for excluded_class in range(2, 10):

        path = f"/projects/leelab/mingyulu/data_att/results/{args.dataset}/unlearning/"
        exp_settings = f"/{excluded_class}/epochs={args.epochs}_lr={args.lr}_loss={args.loss_type}:alpha1={alpha1}_alpha2={alpha2}_weight_reg={args.weight_reg}_fine_tune_att={args.fine_tune_att}"

        ## Init new model for unlearning.

        model = DDPM(
            timesteps=config["timesteps"],
            base_dim=config["base_dim"],
            channel_mult=config["channel_mult"],
            image_size=config["image_size"],
            in_channels=config["in_channels"],
            out_channels=config["out_channels"],
            attn=config["attn"],
            attn_layer=config["attn_layer"],
            num_res_blocks=config["num_res_blocks"],
            dropout=config["dropout"],
        ).to(device)

        ckpt = torch.load(config["trained_model"])
        model.load_state_dict(ckpt["model"])

        cnn = CNN(10).to(device)
        cnn.load_state_dict(torch.load("eval/models/cnn_mnist_noisy.pth"))

        samples = model.sampling_guided(
            args.n_samples,
            clipped_reverse_diffusion=not args.no_clip,
            device=device,
            classifier=cnn,
            target_class=7,
        )

        samples = torch.clamp(((samples + 1.0) / 2.0), 0.0, 1.0)

        save_image(
            samples,
            f"results/{args.dataset}/unlearning/samples/classifier_guided.png",
            nrow=int(math.sqrt(args.n_samples)),
        )


if __name__ == "__main__":
    args = parse_args()
    main(args)
