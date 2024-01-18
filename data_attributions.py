"""Functions for calculating attribution scores including D-TRAK, TRAK, Datamodel, Data Shapley"""
import argparse
import os

import numpy as np
import torch
from sklearn.linear_model import RidgeCV

import constants
from utils import create_dataset, remove_data_by_datamodel, remove_data_by_shapley


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
        "--calibation_num",
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
    """Main function for computing D-TRAK, TRAK, Datamodel, and Data Shapley."""
    # TODO: return score of validation set.

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)
    train_size = args.train_size
    dataset_size = len(train_dataset)

    # Initializing masking array.
    x_train = np.zeros((train_size, dataset_size))
    y_train = np.zeros(train_size)
    coeff = []

    if args.attribution_method == "d-trak":

        scores = np.zeros((len(train_dataset), 1))

        for i in range(0, train_size):

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
                remaining_idx, _ = remove_data_by_shapley(
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

        for i in range(0, train_size):
            # Load and set input, subset masking indicator, and output, model behavior eg. FID score.
            removal_dir = f"{args.removal_dist}/{args.removal_dist}_alpha={args.datamodel_alpha}_seed={args.removal_seed}"
            remaining_idx, _ = remove_data_by_datamodel(
                train_dataset, alpha=args.datamodel_alpha, seed=i
            )
            x_train[i, remaining_idx] = 1

            # Load pre-calculated model behavior
            model_behavior_dir = os.path.join(
                args.outdir,
                args.dataset,
                args.method,
                removal_dir,
                "model_behavior.npy",
            )
            model_output = np.load(model_behavior_dir)
            y_train[i] = model_output[args.model_behavior]

        # Train datamodels

        for i in range(args.calibation_num):

            bootstrapped_indices = np.random.choice(train_size, train_size)
            reg = RidgeCV(cv=5, alphas=[0.1, 1.0, 1e1]).fit(
                x_train[bootstrapped_indices], y_train[bootstrapped_indices]
            )
            coeff.append(reg.coef_)

        coeff = np.stack(coeff)
        scores = x_train @ coeff.T

    elif args.attribution_method == "shapley":

        # calculate shapley value e.g. shapley sampling with each subset until each player's value converge.

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
        null_model_output = np.load(null_behavior_dir)
        full_model_output = np.load(full_behavior_dir)

        # Load pre-calculated model behavior
        for i in range(0, train_size):

            # Load and set input, subset masking indicator, and output, model behavior eg. FID score.
            removal_dir = (
                f"{args.removal_dist}/{args.removal_dist}_seed={args.removal_seed}"
            )
            remaining_idx, _ = remove_data_by_shapley(
                train_dataset, seed=args.removal_seed
            )

            x_train[i, remaining_idx] = 1

            # Load pre-calculated model behavior
            model_behavior_dir = os.path.join(
                args.outdir,
                args.dataset,
                args.method,
                removal_dir,
                "model_behavior.npy",
            )
            model_output = np.load(model_behavior_dir)
            y_train[i] = model_output[args.model_behavior]

        # Closed form solution of Shapley from equation (7) in https://proceedings.mlr.press/v130/covert21a/covert21a.pdf

        train_size = x_train.shape[0]
        dataset_size = x_train.shape[1]

        for i in range(args.calibation_num):

            bootstrapped_indices = np.random.choice(train_size, train_size)

            a_hat = np.zeros((dataset_size, dataset_size))
            b_hat = np.zeros((dataset_size, 1))

            for j in range(train_size):
                a_hat += np.outer(
                    x_train[bootstrapped_indices][j], x_train[bootstrapped_indices][j]
                )
                b_hat += (
                    x_train[bootstrapped_indices][j]
                    * (y_train[bootstrapped_indices][j] - null_model_output)
                )[:, None]

            a_hat /= train_size
            b_hat /= train_size

            # Using np.linalg.pinv instead of np.linalg.inv in case of singular matrix
            a_hat_inv = np.linalg.pinv(a_hat)
            one = np.ones((dataset_size, 1))

            c = one.T @ a_hat_inv @ b_hat - full_model_output + null_model_output
            d = one.T @ a_hat_inv @ one

            coef = a_hat_inv @ (b_hat - one @ (c / d))

            coeff.append(coef)

        coeff = np.stack(coeff)
        scores = x_train @ coeff

    else:
        raise NotImplementedError((f"{args.attribution_method} is not implemented."))

    return scores


if __name__ == "__main__":
    args = parse_args()
    main(args)
