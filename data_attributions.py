"""Functions for calculating attribution scores including D-TRAK, TRAK, Datamodel, Data Shapley"""
import argparse
import glob
import os

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV

import constants
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
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba", "imagenette"],
        default="mnist",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for splitting train and validation set.",
        default=42,
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
        help="strategy for sampling time steps",
    )
    return parser.parse_args()


def main(args):
    """Main function for computing D-TRAK, TRAK, Datamodel, and Data Shapley."""

    full_dataset = create_dataset(dataset_name=args.dataset, train=True)
    dataset_size = len(full_dataset)

    n_subset = args.n_subset
    all_idx = np.arange(n_subset)
    num_selected = int(args.train_ratio * n_subset)

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

        # Load and set input, subset masking indicator, and output, model behavior eg. FID score.
        for i in range(0, n_subset):
            removal_dir = f"{args.removal_dist}/{args.removal_dist}_alpha={args.datamodel_alpha}_seed={i}"
            remaining_idx, _ = remove_data_by_datamodel(
                full_dataset, alpha=args.datamodel_alpha, seed=i
            )
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

        # Train datamodels
        x_train = X[train_idx]
        y_train = Y[train_idx]

        for i in range(args.num_runs):

            bootstrapped_indices = np.random.choice(n_subset, n_subset)
            reg = RidgeCV(cv=5, alphas=[0.1, 1.0, 1e1]).fit(
                x_train[bootstrapped_indices], y_train[bootstrapped_indices]
            )
            coeff.append(reg.coef_)

        coeff = np.stack(coeff)
        scores = X[val_idx] @ coeff.T

    elif args.attribution_method == "shapley":

        # Calculate shapley value e.g. shapley sampling with each subset until each player's value converge.

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

        x_train = X[train_idx]
        y_train = Y[train_idx]

        # Closed form solution of Shapley from equation (7) in https://proceedings.mlr.press/v130/covert21a/covert21a.pdf

        train_size = x_train.shape[0]

        for i in range(args.num_runs):

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
        scores = X[val_idx] @ coeff.T

    elif args.attribution_method == "clip_score":

        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_transform = clip.load("ViT-B/32", device=device)

        val_samples = []
        for filename in glob.glob("path-to-val_samples"):
            im = Image.open(filename)
            val_samples.append(im)

        train_samples = []
        for filename in glob.glob("path-to-val_samples"):
            im = Image.open(filename)
            train_samples.append(im)

        # Find the most similar images w.r.t. clip score (dot product or cosine similarity)

        with torch.no_grad():
            features1 = clip_model.encode_image(val_samples)
            features2 = clip_model.encode_image(train_samples)

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)

        scores = (features1 @ features2.T).cpu().numpy()
    #     TODO

    # elif args.attribution_method == "pixel_dist":
    #      Find the most similar images w.r.t. l2 distance, dot product or cosine similarity.
    # elif args.attribution_method == "if":
    #     raise NotImplementedError

    else:
        raise NotImplementedError((f"{args.attribution_method} is not implemented."))

    return scores


if __name__ == "__main__":
    args = parse_args()
    main(args)
