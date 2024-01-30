"""Functions for calculating attribution scores"""
import argparse
import json
import os

import numpy as np

import constants
from attributions.attribution_utils import CLIPScore, pixel_distance
from attributions.datamodel import compute_datamodel_scores
from attributions.datashapley import compute_shapley_scores
from attributions.trak import compute_dtrak_trak_scores


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
        "--attribution_method",
        type=str,
        default=None,
        choices=[
            "shapley",
            "d-trak",
            "relative_if",
            "randomized_if",
            "datamodel",
            "clip_score",
            "pixel_dist",
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
        help="strategy for sampling time steps for D-TRAK.",
    )
    return parser.parse_args()


def main(args):
    """Main function for computing D-TRAK, TRAK, Datamodel, and Data Shapley."""

    model_behavior_path = os.path.join(
        args.dataset, constants.GLOBAL_MODEL_BEHAVIOR_DIR, "full_model_db.json"
    )

    with open(model_behavior_path, "r") as f:
        model_behavior = json.loads(f)
        model_behavior = [
            row for row in model_behavior if row.get("exp_name") == args.exp_name
        ]

    # Train and test split for datamodel and data shapley.
    all_idx = [i for i in len(model_behavior)]

    rng = np.random.RandomState(args.seed)
    rng.shuffle(all_idx)
    train_ratio = 0.8
    train_idx = all_idx[: train_ratio * len(all_idx)]
    val_idx = all_idx[train_ratio * len(all_idx) :]

    if args.attribution_method in ["d-trak", "relative_if", "randomized_if", "trak"]:

        scores = compute_dtrak_trak_scores(args, train_idx, val_idx)

    elif args.attribution_method == "shapley":

        scores = compute_shapley_scores(args, model_behavior, train_idx, val_idx)

    elif args.attribution_method == "datamodel":

        scores = compute_datamodel_scores(args, model_behavior, train_idx, val_idx)

    elif args.attribution_method == "clip_score":
        if not args.val_samples_dir or not args.train_samples_dir:
            raise FileNotFoundError(
                "Specify both val_samples_dir and train_samples_dir for clip score."
            )
        # Find the highest score for each image in val_samples

        clip = CLIPScore(args.device)
        scores = clip.clip_score(args.val_samples_dir, args.train_samples_dir)

    elif args.attribution_method == "pixel_dist":
        if not args.val_samples_dir or not args.train_samples_dir:
            raise FileNotFoundError(
                "Specify both val_samples_dir and train_samples_dir for pixel distance."
            )

        # Calculate L2 distances and find the highest for each val image
        scores = pixel_distance(args.val_samples_dir, args.train_samples_dir)
    else:
        raise NotImplementedError((f"{args.attribution_method} is not implemented."))

    return scores


if __name__ == "__main__":
    args = parse_args()
    main(args)
