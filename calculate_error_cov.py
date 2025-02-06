"""Calculating Shapley value covergence for various methods."""

import argparse
import json

import numpy as np
from tqdm import tqdm

import src.constants as constants
from src.attributions.methods.datashapley import data_shapley
from src.datasets import create_dataset, remove_data_by_shapley
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate data attribution methods using the linear model score"
    )
    parser.add_argument(
        "--train_db",
        type=str,
        help="filepath of database for recording training model behaviors",
        required=True,
    )
    parser.add_argument(
        "--test_db",
        type=str,
        help="filepath of database for recording testing model behaviors",
        required=True,
    )
    parser.add_argument(
        "--full_db",
        type=str,
        help="filepath of database for recording full model behaviors",
        required=None,
    )
    parser.add_argument(
        "--null_db",
        type=str,
        help="filepath of database for recording null model behaviors",
        required=None,
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for evaluation",
        choices=constants.DATASET,
        default="cifar",
    )
    parser.add_argument(
        "--train_exp_name",
        type=str,
        help="experiment name of records to extract as part of the training set",
        default=None,
    )
    parser.add_argument(
        "--test_exp_name",
        type=str,
        help="experiment name of records to extract as part of the test set",
        default=None,
    )
    parser.add_argument(
        "--max_train_size",
        type=int,
        help="maximum number of subsets for training removal-based data attributions",
        required=True,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        default="retrain",
        choices=constants.METHOD,
    )
    parser.add_argument(
        "--model_behavior_key",
        type=str,
        help="key to query model behavior in the database",
        default="fid_value",
        choices=[
            "is",
            "fid_value",
            "entropy",
            "mse",
            "nrmse",
            "ssim",
            "diffusion_loss",
            "precision",
            "recall",
            "avg_mse",
            "avg_ssim",
            "avg_nrmse",
            "avg_total_loss",
        ],
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated images to consider for local model behaviors",
        default=None,
    )
    parser.add_argument(
        "--by_class",
        help="whether to remove subset by class",
        action="store_true",
        default=False,
    )

    return parser.parse_args()


def removed_by_classes(index_to_class, remaining_idx):
    """Function that maps data index to subgroup index"""
    remaining_classes = set(index_to_class[idx] for idx in remaining_idx)
    all_classes = set(index_to_class.values())
    removed_classes = all_classes - remaining_classes

    return np.array(list(remaining_classes)), np.array(list(removed_classes))


def collect_data(
    db, condition_dict, dataset_name, model_behavior_key, n_samples, by_class
):
    """Collect data for fitting and evaluating data attribution scores."""
    dataset = create_dataset(dataset_name=dataset_name, train=True)

    unique_labels = sorted(set(data[1] for data in dataset))
    value_to_number = {label: i for i, label in enumerate(unique_labels)}

    index_to_class = {i: value_to_number[data[1]] for i, data in enumerate(dataset)}
    # else:
    #     index_to_class = {i: label for i, (_, label) in enumerate(dataset)}

    train_size = len(dataset)

    remaining_masks = []
    model_behaviors = []
    removal_seeds = []

    with open(db, "r") as handle:
        for line in handle:
            record = json.loads(line)

            keep = all(
                [record[key] == condition_dict[key] for key in condition_dict.keys()]
            )

            if keep:
                seed = int(record["removal_seed"])
                # method = record["method"]

                # Check if remaining_idx is empty or incorrect indices.
                if (
                    "remaining_idx" not in record
                    or len(record["remaining_idx"]) > train_size
                ):
                    remaining_idx, removed_idx = remove_data_by_shapley(dataset, seed)
                else:
                    remaining_idx = record["remaining_idx"]

                if by_class:
                    # return class labels as indices if removed by subclasses.
                    remaining_idx, removed_idx = removed_by_classes(
                        index_to_class, remaining_idx
                    )
                    mask_size = len(remaining_idx) + len(removed_idx)
                else:
                    mask_size = train_size

                remaining_mask = np.zeros(mask_size)
                remaining_mask[remaining_idx] = 1

                if n_samples is None:
                    model_behavior = [float(record[model_behavior_key])]
                else:
                    model_behavior = [
                        float(record[f"generated_image_{i}_{model_behavior_key}"])
                        for i in range(n_samples)
                    ]

                # avoid duplicated records

                if seed not in removal_seeds:
                    if record["method"] == "gd":
                        if int(record["gd_steps"]) == 500:
                            remaining_masks.append(remaining_mask)
                            model_behaviors.append(model_behavior)
                            removal_seeds.append(seed)
                    else:
                        remaining_masks.append(remaining_mask)
                        model_behaviors.append(model_behavior)
                        removal_seeds.append(seed)

    remaining_masks = np.stack(remaining_masks)
    model_behaviors = np.stack(model_behaviors)
    removal_seeds = np.array(removal_seeds)

    return remaining_masks, model_behaviors, removal_seeds


def main(args):
    """Main function."""

    # Extract subsets for estimating data attribution scores.
    print(f"Loading training data from {args.train_db}")

    train_condition_dict = {
        "dataset": args.dataset,
        "removal_dist": args.removal_dist,
        "method": args.method,
        "exp_name": args.train_exp_name,
    }
    train_masks, train_targets, train_seeds = collect_data(
        args.train_db,
        train_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )
    print("loading full and null data.")
    _, null_targets, _ = collect_data(
        args.null_db,
        {"dataset": args.dataset, "method": "retrain"},
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )

    _, full_targets, _ = collect_data(
        args.full_db,
        {"dataset": args.dataset, "method": "retrain"},
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )

    test_condition_dict = {
        "dataset": args.dataset,
        "removal_dist": "shapley",
        "method": "retrain",
        "exp_name": f"{args.test_exp_name}",
    }
    print("loading retraining data...")

    test_masks, test_targets, train_seeds = collect_data(
        args.test_db,
        test_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )

    oracle_coeff = data_shapley(
        test_masks.shape[-1],
        test_masks,
        test_targets[:, 0],
        full_targets.flatten()[0],
        null_targets.flatten()[0],
    )

    train_indices = [i for i in range(len(train_masks))]

    np.random.shuffle(train_indices)
    start = train_masks.shape[-1]

    step = 50 if args.max_train_size > 100 else 10

    subset_sizes = [start] + list(range(step, args.max_train_size + 1, step))

    conv_errors = np.zeros((len(subset_sizes), 2))

    for subset_idx, n in enumerate(tqdm(subset_sizes)):
        train_masks_fold = train_masks[train_indices[:n]]
        train_targets_fold = train_targets[train_indices[:n]]

        coeff = data_shapley(
            train_masks_fold.shape[-1],
            train_masks_fold,
            train_targets_fold[:, 0],
            full_targets.flatten()[0],
            null_targets.flatten()[0],
        )
        np.set_printoptions(suppress=True)
        conv_errors[subset_idx] = (n, np.mean(np.sqrt((coeff - oracle_coeff) ** 2)))

    print(conv_errors)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
