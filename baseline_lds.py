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
import torch
from scipy.stats import bootstrap, spearmanr

import src.constants as constants
from src.attributions.global_scores.diversity_score import calcualte_embedding_dist
from src.attributions.methods.attribution_utils import CLIPScore, pixel_distance
from src.attributions.methods.compute_gradient_score import compute_gradient_scores
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
        help="filepath of database for recording LOO model behaviors",
        required=None,
    )
    parser.add_argument(
        "--loo_db",
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
        help="experiment name of records to extract as part of the test set",
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
            "is",
            "entropy",
            "fid_value",
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
            "entropy",
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

    # TRAK/D-TRAK args.
    parser.add_argument(
        "--gradient_type",
        type=str,
        choices=[
            "vanilla_gradient",
            "trak",
            "relative_if",
            "renormalized_if",
            "journey_trak",
            "d_trak",
        ],
        default=None,
        help="Specification for gradient-based model behavior.",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=1024,
        help="Dimension for TRAK projector",
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
    # file path for local model behavior, e.g. pixel_distance, clip score
    parser.add_argument(
        "--by",
        type=str,
        help="aggregation according to mean or max",
        default="mean",
        choices=["mean", "max"],
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=None,
        help="Number of samples for local model behavior",
    )
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


def evaluate_lds(attrs_all, test_data_list, num_model_behaviors):
    """Evaluate LDS mean and CI across a list of test data."""
    lds_list = []
    for (x_test, y_test) in test_data_list:
        model_behavior_lds_list = []
        for k in range(num_model_behaviors):
            model_behavior_lds_list.append(
                spearmanr(x_test @ attrs_all[k], y_test[:, k]).statistic * 100
            )
        lds_list.append(np.mean(model_behavior_lds_list))
    lds_mean = np.mean(lds_list)
    lds_ci = np.std(lds_list) / np.sqrt(len(lds_list)) * 1.96
    return lds_mean, lds_ci


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

    unique_values = sorted(set(data[1] for data in dataset))
    value_to_number = {value: i for i, value in enumerate(unique_values)}

    index_to_class = {i: value_to_number[data[1]] for i, data in enumerate(dataset)}

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

                # Avoid duplicated records based on seed or class.

                if (
                    record["excluded_class"] is not None
                    and record["removal_dist"] is None
                ):
                    seed = int(record["excluded_class"])

                if seed not in removal_seeds:
                    if method == "gd":
                        # if record["trained_steps"] == 500:
                        # Only extract record when trained steps == 4000 for gd.
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
        "removal_dist": "datamodel",
        "datamodel_alpha": args.datamodel_alpha,
        "method": "retrain",  # The test set should pertain only to retrained models.
    }

    if args.dataset == "cifar100":
        test_db_list = [
            os.path.join(
                "/gscratch/aims/mingyulu/results_ming",
                args.dataset,
                "datamodel",
                f"retrain_global_behavior_seed{seed}.jsonl",
            )
            for seed in [42, 43, 44]
        ]
    elif args.dataset == "celeba":
        test_db_list = [
            os.path.join(
                "/gscratch/aims/mingyulu/results_ming",
                args.dataset,
                "datamodel",
                f"diversity_datamodel_"
                f"{str(args.datamodel_alpha).replace('.', '_')}"
                f"_seed{seed}.jsonl",
            )
            for seed in [42, 43, 44]
        ]
    else:
        raise ValueError

    test_data_list = []

    for db_path in test_db_list:
        print(f"Loading testing data from {db_path}")

        test_masks, test_targets, test_seeds = collect_data(
            db_path,
            test_condition_dict,
            args.dataset,
            args.model_behavior_key,
            args.n_samples,
            args.by_class,
        )
        test_data_list.append((test_masks, test_targets))

    random.seed(42)
    np.random.seed(42)

    # Filtering testing sets
    num_targets = test_targets.shape[-1]

    test_indices = np.arange(len(test_masks))

    if args.num_test_subset is not None:
        test_indices = test_indices[: args.num_test_subset]
        test_masks = test_masks[test_indices]
        test_targets = test_targets[test_indices]

    if args.method == "trak":
        coeff = compute_gradient_scores(
            args,
            retraining=False,
        )
    elif args.method == "emb_dist":
        coeff = calcualte_embedding_dist(
            args.dataset, args.training_dir, num_cluster=20, use_cache=False, by=args.by
        )
    elif args.method == "pixel_dist":
        coeff = pixel_distance(
            args.model_behavior_key,
            args.by,
            args.dataset,
            args.n_samples if args.sample_size is None else args.sample_size,
            args.sample_dir,
            args.training_dir,
        )
    elif args.method == "clip_score":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip = CLIPScore(device)
        coeff = clip.clip_score(
            args.model_behavior_key,
            args.by,
            args.dataset,
            args.n_samples if args.sample_size is None else args.sample_size,
            args.sample_dir,
            args.training_dir,
        )

        # print(np.argsort(coeff.flatten())[: int(len(coeff)* 0.05)])
        # print(np.argsort(coeff.flatten())[:int(len(coeff)* 0.1)])
        # print(np.argsort(coeff.flatten())[:int(len(coeff)* 0.2)])
        # print(np.argsort(coeff.flatten())[:int(len(coeff)* 0.3)])
        # print(np.argsort(coeff.flatten())[:int(len(coeff)* 0.4)])
    elif args.method == "loo":
        loo_condition_dict = {
            "exp_name": args.train_exp_name,
            "dataset": args.dataset,
            "method": "retrain",
        }

        loo_masks, loo_targets, _ = collect_data(
            args.loo_db,
            loo_condition_dict,
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

        coeff = np.zeros((num_targets, loo_masks.shape[-1]))

        for i in range(num_targets):
            coeff[i] = np.dot(
                (full_targets[:, i] - loo_targets[:, i]).T, 1.0 - loo_masks
            )
    assert (
        coeff.shape[0] == num_targets
    ), "number of target should match number of samples in sample_dir."

    data_attr_list = coeff

    lds_mean, lds_ci = evaluate_lds(data_attr_list, test_data_list, num_targets)

    print(f"Mean: {lds_mean:.2f} ({lds_ci:.2f})")
    print(f"Confidence interval: ({lds_mean - lds_ci:.2f}, {lds_mean + lds_ci:.2f})")

    if args.bootstrapped:

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
            f"{args.dataset} with {args.method}\n"
            f"Mean: {boot_mean:.3f};"
            f"Confidence interval: ({boot_ci_low:.2f}, {boot_ci_high:.2f})\n"
            f"Max coeff: {np.max(coeff):.3f}; Min coeff: {np.min(coeff):.3f}"
        )

        result_path = f"results/lds/{args.dataset}/{args.method}/"

        os.makedirs(result_path, exist_ok=True)
        plt.savefig(os.path.join(result_path, f"{args.model_behavior_key}.png"))


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
