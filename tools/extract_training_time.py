import os
import numpy as np
import torch
import diffusers
import argparse
import matplotlib.pyplot as plt
import seaborn as sns

import src.constants as constants
from src.utils import get_max_steps
from src.ddpm_config import DDPMConfig

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=constants.DATASET,
        required=True,
    )
    parser.add_argument(
        "--excluded_class",
        type=int,
        help="dataset class to exclude for class-wise data removal",
        default=None,
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["uniform", "datamodel", "shapley"],
        default=None,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="proportion of full dataset to keep in the datamodel distribution",
        default=0.5,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=constants.METHOD,
        required=True,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    return parser.parse_args()


def main(args):
    seed_range = 5000
    training_time = []

    for seed in range(seed_range):

        removal_dir = "full"
        if args.excluded_class is not None:
            removal_dir = f"excluded_{args.excluded_class}"
        if args.removal_dist is not None:
            removal_dir = f"{args.removal_dist}/{args.removal_dist}"
            if args.removal_dist == "datamodel":
                removal_dir += f"_alpha={args.datamodel_alpha}"
            removal_dir += f"_seed={seed}"

        model_outdir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            "models",
            removal_dir,
        )
        existing_steps = get_max_steps(model_outdir)

        if existing_steps is not None:
            ckpt_path = os.path.join(model_outdir, f"ckpt_steps_{existing_steps:0>8}.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            training_time.append(ckpt["total_steps_time"])

    fig, axs = plt.subplots(1, 1, figsize=(20, 6))

    sns.histplot(training_time,  bins=30, alpha=0.5)
    axs.set_xlabel('Training Time')
    axs.set_title(f'Histogram Plot of Training Time for {args.method}: {np.mean(training_time):.2f}; {np.median(training_time):.2f}')

    plt.tight_layout()
    plt.savefig(f"results/training_time_{args.method}.png")

if __name__ == "__main__":
    args = parse_args()
    main(args)
