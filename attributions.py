"""Functions for calculating attribution scores including D-TRAK, TRAK, Datamodel, Data Shapley"""
import argparse
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

import constants
from ddpm_config import DDPMConfig
from utils import (
    create_dataset,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    shapley_kernel,
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
        "--train_size",
        type=int,
        help="Number of sampling subsets for attribution score calculation",
        default=None,
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
        "--attribution_method",
        type=str,
        default=None,
        choices=[
            "shapley",
            "d-trak",
            "datamodel",
        ],
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
        help="strategy for sampling time steps",
    )
    return parser.parse_args()


def main(args):
    """Main function for computing D-TRAK and TRAK."""
    # TODO: return score of validation set.

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)
    x_train = np.zeros((args.train_size, len(train_dataset)))
    y_train = np.zeros(args.train_size)

    if args.attribution_method == "d-trak":

        scores = np.zeros((len(train_dataset), 1))

        for i in range(0, args.train_size):

            # Step1: load computed gradients for each subset.
            if args.removal_dist == "datamodel":
                removal_dir = f"{args.removal_dist}/{args.removal_dist}_alpha={args.datamodel_alpha}_seed={args.removal_seed}"
                remaining_idx, _ = remove_data_by_datamodel(
                    train_dataset, alpha=args.datamodel_alpha, seed=i
                )
            elif args.removal_dist == "shpaley":
                removal_dir = (
                    f"{args.removal_dist}/{args.removal_dist}_seed={args.removal_seed}"
                )
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed
                )

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

        datamodel_coeff = []

        for i in range(0, args.train_size):
            # Load and set input, subset masking indicator, and output, model behavior eg. FID score.
            removal_dir = f"{args.removal_dist}/{args.removal_dist}_alpha={args.datamodel_alpha}_seed={args.removal_seed}"
            remaining_idx, _ = remove_data_by_datamodel(
                train_dataset, alpha=args.datamodel_alpha, seed=i
            )
            x_train[i, remaining_idx] = 1

            # Load pre-calculated model behavior
            model_behavior_dir = os.path.join(
                args.outdir, args.dataset, args.method, removal_dir, "model_behavior"
            )
            model_output = np.load(model_behavior_dir)
            y_train[i] = model_output[args.model_behavior]

            # Train a datamodel
            reg = RidgeCV(
                cv=5,
                alphas=[0.1, 1.0, 1e1],
                random_state=42,
            ).fit(x_train[i], y_train[i])
            datamodel_coeff.append(reg.coef_)

        scores = x_train @ datamodel_coeff.T

    elif args.attribution_method == "shapley":

        # calculate shapley value e.g. shapley sampling with each subset until each player's value converge.

        kernelshap_coeff = []

        for i in range(0, args.train_size):
            # Load and set input, subset masking indicator, and output, model behavior eg. FID score.
            removal_dir = (
                f"{args.removal_dist}/{args.removal_dist}_seed={args.removal_seed}"
            )
            remaining_idx, removed_idx = remove_data_by_shapley(
                train_dataset, seed=args.removal_seed
            )

            x_train[i, remaining_idx] = 1

            # Load pre-calculated model behavior
            model_behavior_dir = os.path.join(
                args.outdir, args.dataset, args.method, removal_dir, "model_behavior"
            )
            model_output = np.load(model_behavior_dir)
            y_train[i] = model_output[args.model_behavior]

            # Get kernel weights
            weight = shapley_kernel(train_dataset, args.removal_seed)

            # Train a linear regression.
            reg = RidgeCV(
                cv=5,
                alphas=[0.1, 1.0, 1e1],
                random_state=42,
            ).fit(x_train[i], y_train[i])
            kernelshap_coeff.append(weight * reg.coef_)

        scores = x_train @ kernelshap_coeff.T

    else:
        raise NotImplementedError((f"{args.attribution_method} is not implemented."))

    return scores


if __name__ == "__main__":
    args = parse_args()
    main(args)
