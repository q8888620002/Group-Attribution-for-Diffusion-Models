"""Functions for calculating attribution scores including D-TRAK, TRAK, Datamodel, Data Shapley"""
import argparse
import os

import numpy as np
import torch

import constants
from attributions.attribution_utils import (
    clip_score,
    data_shapley,
    datamodel,
    pixel_distance,
)
from utils import create_dataset, remove_data_by_datamodel, remove_data_by_shapley


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="Computing data attribution.")
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
        "--val_sample_dir",
        type=str,
        help="directory for validation samples",
        default=None,
    )
    parser.add_argument(
        "--train_sample_dir",
        type=str,
        help="directory for training samples",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba", "imagenette"],
        default="mnist",
    )
    parser.add_argument(
        "--n_subset",
        type=int,
        help="Number of subsests for attribution score calculation",
        default=None,
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        help="Ratio of subsets for shapley & datamodel calculation.",
        default=0.8,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for splitting train and validation set.",
        default=42,
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
        "--num_runs",
        type=int,
        default=1000,
        help="Number of runs for obtain confidence interval",
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
        "--attribution_method",
        type=str,
        default=None,
        choices=["shapley", "d-trak", "datamodel", "clip_score", "pixel_dist", "if"],
        help="Specification for attribution score methods",
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
            "unlearn_retrain_fid",
        ],
        help="Specification for D-TRAK model behavior.",
    )

    parser.add_argument(
        "--t_strategy",
        type=str,
        default=None,
        help="strategy for sampling time steps for D-TRAK.",
    )
    return parser.parse_args()


def main(args):
    """Main function for computing D-TRAK, TRAK, Datamodel, and Data Shapley."""

    full_dataset = create_dataset(dataset_name=args.dataset, train=True)
    dataset_size = len(full_dataset)

    n_subset = args.n_subset
    all_idx = np.arange(n_subset)
    num_selected = int(args.train_ratio * n_subset)

    # Train and test split for datamodel and data shapley.
    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_idx)

    train_idx = all_idx[:num_selected]
    val_idx = all_idx[num_selected:]

    # Initializing masking array (n, d); n is the subset number and d is number of data is the original dataset.

    X = np.zeros((n_subset, dataset_size))
    Y = np.zeros(n_subset)

    coeff = []

    if args.attribution_method == "d-trak":

        scores = np.zeros((n_subset, 1))

        # Iterating only through validation set for D-TRAK/TRAK.

        for i in enumerate(val_idx):

            # Step1: load computed gradients for each subset.
            if args.removal_dist == "datamodel":
                removal_dir = f"{args.removal_dist}/{args.removal_dist}_alpha={args.datamodel_alpha}_seed={i}"
                remaining_idx, _ = remove_data_by_datamodel(
                    full_dataset, alpha=args.datamodel_alpha, seed=i
                )
            elif args.removal_dist == "shpaley":
                removal_dir = f"{args.removal_dist}/{args.removal_dist}_seed={i}"
                remaining_idx, _ = remove_data_by_shapley(full_dataset, seed=i)

            grad_result_dir = os.path.join(
                args.outdir,
                args.dataset,
                args.method,
                "d_track",
                removal_dir,
                f"f={args.model_behavior}_t={args.t_strategy}",
            )

            dstore_keys = np.memmap(
                grad_result_dir,
                dtype=np.float32,
                mode="r",
                shape=(len(remaining_idx), args.projector_dim),
            )

            dstore_keys = torch.from_numpy(dstore_keys).cuda()

            print(dstore_keys.size())

            # Step2: calculate D-TRAK or TRAK for each subset as in https://github.com/sail-sg/d-trak.

            kernel = dstore_keys.T @ dstore_keys
            kernel = kernel + 5e-1 * torch.eye(kernel.shape[0]).cuda()

            kernel = torch.linalg.inv(kernel)

            print(kernel.shape)
            print(torch.mean(kernel.diagonal()))

            # scores = gen_dstore_keys.dot((dstore_keys@kernel_).T)
            score = dstore_keys @ ((dstore_keys @ kernel).T)
            print(score.size())

            scores[i] = score.cpu().numpy()

    elif args.attribution_method == "datamodel":

        # Load and set input, subset masking indicator, and output, pre-calculated model behavior.

        for i in range(0, n_subset):
            removal_dir = f"{args.removal_dist}/{args.removal_dist}_alpha={args.datamodel_alpha}_seed={i}"
            model_behavior_dir = os.path.join(
                args.outdir,
                args.dataset,
                args.method,
                removal_dir,
                "model_behavior.npy",
            )
            model_output = np.load(model_behavior_dir)

            remaining_idx, _ = remove_data_by_datamodel(
                full_dataset, alpha=args.datamodel_alpha, seed=i
            )
            X[i, remaining_idx] = 1
            Y[i] = model_output[args.model_behavior]

        # Train datamodels and calculate score for validation sets.
        coeff = datamodel(X[train_idx], Y[train_idx], args.num_runs)
        scores = X[val_idx] @ coeff.T

    elif args.attribution_method == "shapley":

        null_behavior_dir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            removal_dir,
            "null/model_behavior.npy",
        )
        full_behavior_dir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            removal_dir,
            "full/model_behavior.npy",
        )

        # Load v(1) and v(0)
        v0 = np.load(null_behavior_dir)
        v1 = np.load(full_behavior_dir)

        # Load pre-calculated model behavior
        for i in range(0, n_subset):

            # Load and set input, subset masking indicator, and output, model behavior eg. FID score.
            removal_dir = f"{args.removal_dist}/{args.removal_dist}_seed={i}"
            remaining_idx, _ = remove_data_by_shapley(full_dataset, seed=i)

            X[i, remaining_idx] = 1

            # Load pre-calculated model behavior
            model_behavior_dir = os.path.join(
                args.outdir,
                args.dataset,
                args.method,
                removal_dir,
                "model_behavior.npy",
            )
            model_output = np.load(model_behavior_dir)
            Y[i] = model_output[args.model_behavior]

        coeff = data_shapley(
            dataset_size, X[train_idx], Y[train_idx], v1, v0, args.num_runs
        )
        scores = X[val_idx] @ coeff.T

    elif args.attribution_method == "clip_score":
        if not args.val_samples_dir or not args.train_samples_dir:
            raise FileNotFoundError(
                "Specify both val_samples_dir and train_samples_dir for clip score."
            )
        # Find the highest score for each image in val_samples
        scores = clip_score(args.val_samples_dir, args.train_samples_dir)

    elif args.attribution_method == "pixel_dist":
        if not args.val_samples_dir or not args.train_samples_dir:
            raise FileNotFoundError(
                "Specify both val_samples_dir and train_samples_dir for pixel distance."
            )

        # Calculate L2 distances and find the highest for each val image
        scores = pixel_distance(args.val_samples_dir, args.train_samples_dir)

    elif args.attribution_method == "if":
        raise NotImplementedError

    else:
        raise NotImplementedError((f"{args.attribution_method} is not implemented."))

    return scores


if __name__ == "__main__":
    args = parse_args()
    main(args)
