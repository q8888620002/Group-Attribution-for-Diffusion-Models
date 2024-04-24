"""
Evaluate data attributions using the linear datamodel score (LDS).

LDS calculateion for D-TRAK/TRAK is based on
https://github.com/sail-sg/D-TRAK/blob/main/CIFAR2/methods/04_if/01_IF_val_5000-0.5.ipynb
"""

import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import bootstrap, spearmanr
from sklearn.linear_model import RidgeCV

# from sklearn.model_selection import KFold
from tqdm import tqdm

import src.constants as constants
from src.attributions.methods.datashapley import (  # kernel_shap,; kernel_shap_ridge,
    data_shapley,
)
from src.datasets import create_dataset, remove_data_by_shapley
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate data attribution methods using the linear model score"
    )
    parser.add_argument(
        "--test_db",
        type=str,
        help="filepath of database for recording test model behaviors",
        required=True,
    )
    parser.add_argument(
        "--train_db",
        type=str,
        help="filepath of database for recording training model behaviors",
        required=True,
    )
    parser.add_argument(
        "--full_db",
        type=str,
        help="filepath of database for recording training model behaviors",
        required=None,
    )
    parser.add_argument(
        "--null_db",
        type=str,
        help="filepath of database for recording training model behaviors",
        required=None,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for evaluation",
        choices=constants.DATASET,
        default="cifar",
    )
    parser.add_argument(
        "--test_exp_name",
        type=str,
        help="experiment name of records to extract as part of the test set",
        default=None,
    )
    parser.add_argument(
        "--train_exp_name",
        type=str,
        help="experiment name of records to extract as part of the training set",
        default=None,
    )
    parser.add_argument(
        "--max_train_size",
        type=int,
        help="maximum number of subsets for training removal-based data attributions",
        required=True,
    )
    parser.add_argument(
        "--num_test_subset",
        type=int,
        help="number of testing subsets",
        default=32,
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
        default="retrain",
        choices=constants.METHOD,
    )
    parser.add_argument(
        "--model_behavior_key",
        type=str,
        help="key to query model behavior in the database",
        default="fid_value",
        choices=[
            "fid_value",
            "mse",
            "nrmse",
            "ssim",
            "diffusion_loss",
            "precision",
            "recall",
            "is",
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
    parser.add_argument(
        "--bootstrapped",
        help="whether to calculate CI with bootstrapped sampling",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_bootstrap_iters",
        type=int,
        help="number of bootstrapped iterations",
        default=100,
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
    index_to_class = {i: label for i, (_, label) in enumerate(dataset)}

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
                method = record["method"]

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
                    if method == "gd":
                        if record["trained_steps"] == 4000:
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
    # Extract subsets for LDS test evaluation.
    test_condition_dict = {
        "exp_name": args.test_exp_name,
        "dataset": args.dataset,
        # "removal_dist": args.removal_dist,
        "method": "retrain",  # The test set should pertain only to retrained models.
    }

    print(f"Loading testing data from {args.test_db}")
    test_masks, test_targets, test_seeds = collect_data(
        args.test_db,
        test_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )

    # Extract subsets for estimating data attribution scores.
    print(f"Loading training data from {args.train_db}")

    train_condition_dict = {
        "dataset": args.dataset,
        "removal_dist": args.removal_dist,
        "method": "retrain"
        if args.method in ["trak", "clip_score", "pixel_dist"]
        else args.method,
        "exp_name": "retrain_vs_retrain"
        if args.method in ["trak", "clip_score", "pixel_dist"]
        else args.train_exp_name,
    }
    train_masks, train_targets, train_seeds = collect_data(
        args.train_db,
        train_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )

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

    random.seed(42)
    np.random.seed(42)

    # Filtering testing sets

    if args.train_db == args.test_db:
        # If testing subsests within the same distribution.

        common_seeds = list(set(train_seeds) & set(test_seeds))
        test_seeds_filtered = random.sample(common_seeds, args.num_test_subset)

        test_indices_bool = np.isin(test_seeds, test_seeds_filtered)
        test_indices = np.where(test_indices_bool)[0]
        test_masks = test_masks[test_indices]
        test_targets = test_targets[test_indices]

    else:
        test_seeds_filtered = []
        test_indices = np.arange(len(test_masks))

    if args.num_test_subset is not None:
        test_indices = test_indices[: args.num_test_subset]
        test_masks = test_masks[test_indices]
        test_targets = test_targets[test_indices]

    # Select training instances & filtering overlapped subset

    overlap_bool = np.isin(train_seeds, test_seeds_filtered)
    train_indices = np.where(~overlap_bool)[0]

    matches = np.all(
        train_masks[train_indices, None, :] == test_masks[None, :, :], axis=2
    )

    matching_train_indices = np.any(matches, axis=1)
    train_indices = np.where(~matching_train_indices)[0]

    np.random.shuffle(train_indices)

    num_targets = train_targets.shape[-1]

    start = train_masks.shape[-1]
    step = 100

    subset_sizes = [start] + list(range(step, args.max_train_size + 1, step))

    for n in tqdm(subset_sizes):
        train_masks_fold = train_masks[train_indices[:n]]
        train_targets_fold = train_targets[train_indices[:n]]

        data_attr_list = []

        for i in tqdm(range(num_targets)):

            if args.removal_dist == "datamodel":

                datamodel = RidgeCV(alphas=np.linspace(0.01, 10, 100)).fit(
                    train_masks_fold, train_targets_fold[:, i]
                )
                datamodel_str = "Ridge"
                print("Datamodel parameters")
                print(f"\tmodel={datamodel_str}")
                print(f"\talpha={datamodel.alpha_:.8f}")

                coeff = datamodel.coef_

            elif args.removal_dist == "shapley":

                coeff = data_shapley(
                    train_masks_fold.shape[-1],
                    train_masks_fold,
                    train_targets_fold[:, i],
                    full_targets.flatten()[i],
                    null_targets.flatten()[i],
                )
            else:
                raise ValueError(
                    (f"Removal distribution: {args.removal_dist} does not exist.")
                )

            data_attr_list.append(coeff)

        # Calculate LDS for different training subsets.

        def my_lds(idx):
            boot_masks = test_masks[idx, :]
            lds_list = []
            for i in range(num_targets):
                boot_targets = test_targets[idx, i]
                lds_list.append(
                    spearmanr(boot_masks @ data_attr_list[i], boot_targets).statistic
                    * 100
                )
            return np.mean(lds_list)

        print(f"Estimating scores with {len(train_masks_fold)} subsets with bootstrap.")
        boot_result = bootstrap(
            data=(list(range(len(test_targets))),),
            statistic=my_lds,
            n_resamples=args.num_bootstrap_iters,
            random_state=42,
        )
        boot_mean = np.mean(boot_result.bootstrap_distribution.mean())
        boot_se = boot_result.standard_error
        boot_ci_low = boot_result.confidence_interval.low
        boot_ci_high = boot_result.confidence_interval.high

        print(f"Mean: {boot_mean:.2f}")
        print(f"Standard error: {boot_se:.2f}")
        print(f"Confidence interval: ({boot_ci_low:.2f}, {boot_ci_high:.2f})")

        coeff = np.array(data_attr_list).flatten()

        plt.figure(figsize=(20, 10))
        bin_edges = np.histogram_bin_edges(coeff, bins="auto")
        sns.histplot(coeff, bins=bin_edges, alpha=0.5)

        plt.xlabel("Shapley Value")
        plt.ylabel("Frequency")
        plt.title(
            f"{args.dataset} with {len(train_masks_fold)} training set\n"
            f"Mean: {boot_mean:.3f};"
            f"Confidence interval: ({boot_ci_low:.2f}, {boot_ci_high:.2f})\n"
            f"Max coeff: {np.max(coeff):.3f}; Min coeff: {np.min(coeff):.3f}"
        )

        result_path = f"results/lds/{args.dataset}/{args.method}/"

        os.makedirs(result_path, exist_ok=True)
        plt.savefig(
            os.path.join(
                result_path, f"{args.model_behavior_key}_{len(train_masks_fold)}.png"
            )
        )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
