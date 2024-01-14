import argparse
import os

import numpy as np
import torch

import constants
from ddpm_config import DDPMConfig
from utils import (
    create_dataset,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Computing D-TRAK and TRAK")
    parser.add_argument(
        "--load",
        type=str,
        help="directory path for loading pre-trained model",
        default=None,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
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
        "--projector_dim",
        type=int,
        default=1024,
        help="Dimension for TRAK projector",
    )
    parser.add_argument(
        "--model_behavior",
        type=str,
        default=None,
        choices=[
            "mean",
            "mean-squared-l2-norm",
            "l1-norm",
            "l2-norm",
            "linf-norm",
        ],
        help="Specification for D-TRAK model behavior.",
    )

    parser.add_argument(
        "--t_strategy",
        type=str,
        default=None,
        help="strategy for sampling time steps",
    )
    return parser.parse_args()


def main(args):
    """Main function for computing D-TRAK and TRAK."""

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
    elif args.dataset == "imagenette":
        config = {**DDPMConfig.imagenette_config}
    else:
        raise ValueError(
            (
                f"dataset={args.dataset} is not one of "
                "['cifar', 'mnist', 'celeba', 'imagenette']"
            )
        )
    removal_dir = "full"
    if args.excluded_class is not None:
        removal_dir = f"excluded_{args.excluded_class}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)

    if args.excluded_class is not None:
        removal_dir = f"excluded_{args.excluded_class}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"

    if args.excluded_class is not None:
        remaining_idx, removed_idx = remove_data_by_class(
            train_dataset, excluded_class=args.excluded_class
        )
    elif args.removal_dist is not None:
        if args.removal_dist == "uniform":
            remaining_idx, removed_idx = remove_data_by_uniform(
                train_dataset, seed=args.removal_seed
            )
        elif args.removal_dist == "datamodel":
            remaining_idx, removed_idx = remove_data_by_datamodel(
                train_dataset, alpha=args.datamodel_alpha, seed=args.removal_seed
            )
        elif args.removal_dist == "shapley":
            remaining_idx, removed_idx = remove_data_by_shapley(
                train_dataset, seed=args.removal_seed
            )
        else:
            raise NotImplementedError
    else:
        remaining_idx = np.arange(len(train_dataset))
        removed_idx = np.array([], dtype=int)

    save_dir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        "d_track",
        removal_dir,
        f"f={args.model_behavior}_t={args.t_strategy}",
    )

    dstore_keys = np.memmap(
        save_dir,
        dtype=np.float32,
        mode="r",
        shape=(len(remaining_idx), args.projector_dim),
    )

    dstore_keys = torch.from_numpy(dstore_keys).cuda()

    print(dstore_keys.size())

    kernel = dstore_keys.T @ dstore_keys
    kernel = kernel + 5e-1 * torch.eye(kernel.shape[0]).cuda()

    kernel = torch.linalg.inv(kernel)

    print(kernel.shape)
    print(torch.mean(kernel.diagonal()))

    # scores = gen_dstore_keys.dot((dstore_keys@kernel_).T)
    scores = dstore_keys @ ((dstore_keys @ kernel).T)
    print(scores.size())
    scores = scores.cpu().numpy()


if __name__ == "__main__":
    args = parse_args()
    main(args)
