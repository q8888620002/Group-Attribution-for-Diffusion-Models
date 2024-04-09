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
from sklearn.model_selection import KFold
from tqdm import tqdm

import src.constants as constants
from src.attributions.methods.attribution_utils import CLIPScore, pixel_distance
from src.attributions.methods.compute_trak_score import compute_dtrak_trak_scores
from src.attributions.methods.datashapley import (  # kernel_shap,; kernel_shap_ridge,
    data_shapley,
)
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
    # Data shapley args.
    parser.add_argument(
        "--v1",
        type=float,
        help="full model behavior, required for data shapley",
        default=None,
    )
    parser.add_argument(
        "--v0",
        type=float,
        help="null model behavior, required for data shapley",
        default=None,
    )

    # TRAK/D-TRAK args.
    parser.add_argument(
        "--trak_behavior",
        type=str,
        choices=[
            "loss",
            "mean",
            "mean-squared-l2-norm",
            "l1-norm",
            "l2-norm",
            "linf-norm",
        ],
        default=None,
        help="Specification for D-TRAK model behavior.",
    )
    parser.add_argument(
        "--t_strategy",
        type=str,
        choices=["uniform", "cumulative"],
        help="strategy for sampling time steps",
    )
    parser.add_argument(
        "--k_partition",
        type=int,
        default=None,
        help="Partition for embeddings across time steps.",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=1024,
        help="Dimension for TRAK projector",
    )
    # file path for local model behavior, e.g. pixel_distance, clip score

    parser.add_argument(
        "--sample_dir",
        type=str,
        help="filepath of sample (generated) images ",
    )
    parser.add_argument(
        "--training_dir",
        type=str,
        help="filepath of training data ",
    )
    return parser.parse_args()


def removed_by_classes(index_to_class, remaining_idx):
    """Function that maps data index to subgroup index"""
    remaining_classes = set(index_to_class[idx] for idx in remaining_idx)
    all_classes = set(range(20))  # Assuming 20 classes
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
                remaining_idx = record["remaining_idx"]
                if record["method"] == "ga":
                    remaining_idx = record["removal_idx"]

                if by_class:
                    remaining_idx, removed_idx = removed_by_classes(
                        index_to_class, remaining_idx
                    )
                    remaining_mask = np.zeros(len(remaining_idx) + len(removed_idx))
                else:
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
                    if record["method"] == "gd":
                        if record["trained_steps"] == 4000:
                            remaining_masks.append(remaining_mask)
                            model_behaviors.append(model_behavior)
                            removal_seeds.append(int(record["removal_seed"]))
                    else:
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
        args.by_class,
    )
    # Extract subsets for estimating data attribution scores.
    train_condition_dict = {
        "exp_name": args.train_exp_name,
        "dataset": args.dataset,
        "removal_dist": args.removal_dist,
        "datamodel_alpha": args.datamodel_alpha,
        "method": "retrain" if args.method == "trak" else args.method,
    }
    # Set random states

    random.seed(42)
    np.random.seed(42)

    train_masks, train_targets, train_seeds = collect_data(
        args.train_db,
        train_condition_dict,
        args.dataset,
        args.model_behavior_key,
        args.n_samples,
        args.by_class,
    )

    common_seeds = list(set(train_seeds) & set(test_seeds))
    common_seeds.sort()

    test_seeds_filtered = random.sample(common_seeds, 5 * args.num_test_subset)

    # Select training instances.
    overlap_bool = np.isin(train_seeds, test_seeds_filtered)
    train_indices = np.where(~overlap_bool)[0]

    if args.max_train_size is not None and len(train_indices) > args.max_train_size:
        np.random.shuffle(train_indices)
        train_indices = train_indices[: args.max_train_size]

    train_masks = train_masks[train_indices]
    train_targets = train_targets[train_indices]

    data_attr_list = []

    num_targets = train_targets.shape[-1]

    for i in tqdm(range(num_targets)):

        if args.method == "trak":
            coeff = compute_dtrak_trak_scores(
                args, retraining=False, training_seeds=train_seeds[train_indices]
            )
        elif args.method == "pixel_dist":
            coeff = pixel_distance(args.sample_dir, args.training_dir)
        elif args.method == "clip_score":
            coeff = CLIPScore.clip_score(args.sample_dir, args.training_dir)

        elif args.removal_dist == "datamodel":

            datamodel = RidgeCV(alphas=np.linspace(0.01, 10, 100)).fit(
                train_masks, train_targets[:, i]
            )
            datamodel_str = "Ridge"
            print("Datamodel parameters")
            print(f"\tmodel={datamodel_str}")
            print(f"\talpha={datamodel.alpha_:.8f}")

            coeff = datamodel.coef_

        elif args.removal_dist == "shapley":

            # cifar100: Fid: v1 = 20.0, v0 = 348.45

            coeff = data_shapley(
                train_masks.shape[-1],
                train_masks,
                train_targets[:, i],
                args.v1,
                args.v0,
            )

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
            np.set_printoptions(suppress=True)
            print((test_masks_fold @ data_attr_list[i]).reshape(-1))
            print(test_targets_fold[:, i])
            k_fold_lds.append(
                spearmanr(
                    test_masks_fold @ data_attr_list[i], test_targets_fold[:, i]
                ).statistic
                * 100
            )

        print(f"Mean: {np.mean(k_fold_lds):.3f}")
        print(f"Standard error: {1.96*np.std(k_fold_lds)/np.sqrt(num_folds):.3f}")

        # plots for sanity check
        fig, axs = plt.subplots(1, 1, figsize=(20, 10))
        bin_edges = np.histogram_bin_edges(coeff, bins="auto")
        sns.histplot(coeff, bins=bin_edges, alpha=0.5)

        plt.xlabel("Shapley Value")
        plt.ylabel("Frequency")
        plt.title(
            f"{args.dataset} with {args.max_train_size} training set\n"
            f"Mean: {np.mean(k_fold_lds):.3f};"
            f"Standard error: {1.96*np.std(k_fold_lds)/np.sqrt(num_folds):.3f}\n"
            f"Max coeff: {np.max(coeff):.3f}; Min coeff: {np.min(coeff):.3f}"
        )

        result_path = f"results/lds/{args.dataset}/{args.method}/"

        os.makedirs(result_path, exist_ok=True)
        plt.savefig(
            os.path.join(
                result_path, f"{args.model_behavior_key}_{args.max_train_size}.png"
            )
        )

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
