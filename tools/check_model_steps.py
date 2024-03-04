"""Train or perform unlearning on a diffusion model."""
import argparse
import json
import os

import numpy as np
import torch

import constants
from utils import (
    create_dataset,
    get_max_steps,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba", "imagenette"],
        default="mnist",
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
        "--removal_seed",
        type=int,
        help="random seed for sampling from the removal distribution",
        default=0,
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
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )

    return parser.parse_args()


def main(args):

    results = {}

    for i in range(300):
        removal_seed = i
        removal_dir = "full"
        if args.excluded_class is not None:
            removal_dir = f"excluded_{args.excluded_class}"
        if args.removal_dist is not None:
            removal_dir = f"{args.removal_dist}/{args.removal_dist}"
            if args.removal_dist == "datamodel":
                removal_dir += f"_alpha={args.datamodel_alpha}"
            removal_dir += f"_seed={removal_seed}"

        model_outdir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            "models",
            removal_dir,
        )
        steps = get_max_steps(model_outdir)

        results[model_outdir] = steps

    with open(
        f"/gscratch/aims/diffusion-attr/results_ming/{args.dataset}/train_steps.jsonl",
        "w",
    ) as f:
        f.write(json.dumps(results) + "\n")

    steps = [v if v else 0 for _, v in results.items()]
    values, counts = np.unique(steps, return_counts=True)

    count_dict = {}

    for _, (value, count) in enumerate(zip(values, counts)):
        count_dict[value] = count

    print(count_dict)


if __name__ == "__main__":
    args = parse_args()
    main(args)
