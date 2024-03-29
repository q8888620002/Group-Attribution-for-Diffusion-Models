"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import json
import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import bootstrap, spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from tqdm import tqdm

import src.constants as constants
from src.attributions.methods.datashapley import data_shapley
from src.datasets import create_dataset
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
        default=None,
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
        choices=["retrain", "gd"],
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


def collect_data(db, condition_dict, dataset, model_behavior_key, n_samples):
    """Collect data for fitting and evaluating data attribution scores."""
    train_size = len(create_dataset(dataset_name=dataset, train=True))

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
                remaining_idx = record["remaining_idx"]
                if record["method"] == "ga":
                    remaining_idx = record["removal_idx"]
                remaining_mask = np.zeros(train_size)
                remaining_mask[remaining_idx] = 1

                if n_samples is None:
                    model_behavior = [float(record[model_behavior_key])]
                else:
                    model_behavior = [
                        float(record[f"generated_image_{i}_{model_behavior_key}"])
                        for i in range(n_samples)
                    ]

                if int(record["removal_seed"]) not in removal_seeds:
                    # avoid duplicated records
                    remaining_masks.append(remaining_mask)
                    model_behaviors.append(model_behavior)
                    removal_seeds.append(int(record["removal_seed"]))

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
        "removal_dist": args.removal_dist,
        "datamodel_alpha": args.datamodel_alpha,
        "method": "retrain",  # The test set should pertain only to retrained models.
    }
    test_masks, test_targets, test_seeds = collect_data(
        args.test_db,
        test_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
    )
    # Extract subsets for estimating data attribution scores.
    train_condition_dict = {
        "exp_name": args.train_exp_name,
        "dataset": args.dataset,
        "removal_dist": args.removal_dist,
        "datamodel_alpha": args.datamodel_alpha,
        "method": args.method,
    }
    train_masks, train_targets, train_seeds = collect_data(
        args.train_db,
        train_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
    )

    common_seeds = list(set(train_seeds) & set(test_seeds))
    common_seeds.sort()

    random.seed(42)
    test_seeds_filtered = random.sample(common_seeds, 5 * args.num_test_subset)

    # Select training instances.
    overlap_bool = np.isin(train_seeds, test_seeds_filtered)
    train_indices = np.where(~overlap_bool)[0]

    if args.max_train_size is not None and len(train_indices) > args.max_train_size:
        train_indices = train_indices[: args.max_train_size]

    train_masks = train_masks[train_indices]
    train_targets = train_targets[train_indices]

    data_attr_list = []

    num_targets = train_targets.shape[-1]

    for i in tqdm(range(num_targets)):

        if args.removal_dist == "datamodel":

            datamodel = RidgeCV(alphas=np.linspace(0.01, 10, 100)).fit(
                train_masks, train_targets[:, i]
            )
            datamodel_str = "Ridge"
            print("Datamodel parameters")
            print(f"\tmodel={datamodel_str}")
            print(f"\talpha={datamodel.alpha_:.8f}")

            coeff = datamodel.coef_

        elif args.removal_dist == "shapley":

            v1 = 8.54
            v0 = 348.45

            coeff = data_shapley(
                train_masks.shape[-1],
                train_masks,
                train_targets[:, i],
                v1,
                v0,
            )

        # plots for sanity check
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))
        bin_edges = np.histogram_bin_edges(coeff, bins='auto') 
        sns.histplot(coeff, bins=bin_edges, alpha=0.5)
        # plt.xscale('symlog')  # Apply symmetric log scale to the x-axis
        plt.xlabel('Shapley Value (log scale)')
        plt.ylabel('Frequency')
        plt.title(f'data shapley: {np.max(coeff)}; {np.min(coeff)}')
        plt.savefig(f"results/data_shapley_{args.method}_{args.max_train_size}.png")

        data_attr_list.append(coeff)

        # Calculate LDS with K-Folds

        num_folds = 5
        k_fold_lds = []

        kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

        for fold_idx, (_, test_index) in enumerate(kf.split(test_seeds_filtered)):
            test_seeds_fold = [test_seeds_filtered[i] for i in test_index]

            test_indices_bool = np.isin(test_seeds, test_seeds_fold)
            test_indices = np.where(test_indices_bool)[0]

            test_masks_fold = test_masks[test_indices]
            test_targets_fold = test_targets[test_indices]

            print(
                f"Estimating scores with {len(train_masks)} subsets"
                f" in {fold_idx+1}-fold"
            )

            k_fold_lds.append(
                spearmanr(
                    test_masks_fold @ data_attr_list[i], test_targets_fold[:, i]
                ).statistic
                * 100
            )

        print(f"Mean: {np.mean(k_fold_lds):.3f}")
        print(f"Standard error: {1.96*np.std(k_fold_lds)/np.sqrt(num_folds):.3f}")

    # Calculate test LDS with bootstrapping.
    if args.bootstrapped:

        test_indices_bool = np.isin(test_seeds, test_seeds_filtered)
        test_indices = np.where(test_indices_bool)[0]

        test_masks = test_masks[test_indices]
        test_targets = test_targets[test_indices]

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

        print(f"Estimating scores with {len(train_targets)} subsets with bootstrap.")
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


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
