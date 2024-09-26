"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.ddpm_config import DatasetStats
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate LOO values using the linear model score"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
        choices=["artbench_post_impressionism"],
        default="artbench_post_impressionism",
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="datamodel alpha for the test set",
        default=0.5,
    )
    parser.add_argument(
        "--fit_db",
        type=str,
        help="database with model behaviors for fitting LOO values",
        required=True,
    )
    parser.add_argument(
        "--full_db",
        type=str,
        help="database with model behaviors for the fully trained model",
        required=True,
    )
    parser.add_argument(
        "--model_behavior_key",
        type=str,
        help="key to query model behavior in the databases",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated images to consider for local model behaviors",
        default=None,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="number of subsets used for evaluating data attributions",
        default=100,
    )
    return parser.parse_args()


def collect_data(
    df, num_groups, model_behavior_key, n_samples, collect_remaining_masks=True
):
    """Collect data for fitting and evaluating attribution scores from a data frame."""

    model_behavior_array = []
    if collect_remaining_masks:
        remaining_mask_array = []

    for _, row in df.iterrows():
        if collect_remaining_masks:
            remaining_idx = row["remaining_idx"]
            remaining_mask = np.zeros(num_groups)
            remaining_mask[remaining_idx] = 1
            remaining_mask_array.append(remaining_mask)

        if n_samples is None:
            model_behavior = [row[model_behavior_key]]
        else:
            model_behavior = [
                row[f"generated_image_{i}_{model_behavior_key}"]
                for i in range(n_samples)
            ]
        model_behavior_array.append(model_behavior)

    model_behavior_array = np.stack(model_behavior_array)
    if collect_remaining_masks:
        remaining_mask_array = np.stack(remaining_mask_array)
        return remaining_mask_array, model_behavior_array
    else:
        return model_behavior_array


def evaluate_lds(attrs_all, test_data_list, num_model_behaviors):
    """Evaluate LDS mean and CI across a list of test data."""
    lds_list = []
    for (x_test, y_test) in test_data_list:
        model_behavior_lds_list = []
        for k in range(num_model_behaviors):
            model_behavior_lds_list.append(
                spearmanr(x_test @ attrs_all[:, k], y_test[:, k]).statistic * 100
            )
        lds_list.append(np.mean(model_behavior_lds_list))
    lds_mean = np.mean(lds_list)
    lds_ci = np.std(lds_list) / np.sqrt(len(lds_list)) * 1.96
    return lds_mean, lds_ci


def main(args):
    """Main function."""
    if args.dataset == "artbench_post_impressionism":
        dataset_stats = DatasetStats.artbench_post_impressionism_stats
        num_groups = dataset_stats["num_groups"]
        test_db_list = [
            os.path.join(
                "/gscratch/aims/diffusion-attr",
                f"seed{seed}",
                args.dataset,
                f"retrain_artist_datamodel_alpha={args.datamodel_alpha}.jsonl",
            )
            for seed in [42, 43, 44]
        ]
    else:
        raise ValueError

    # Collect test data.
    test_data_list = []
    for test_db in test_db_list:
        test_df = pd.read_json(test_db, lines=True)
        test_df["subset_seed"] = (
            test_df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
        )
        test_df = test_df.sort_values(by="subset_seed")
        test_subset_seeds = [i for i in range(args.test_size)]
        test_df = test_df[test_df["subset_seed"].isin(test_subset_seeds)]
        assert len(test_df) == args.test_size
        x_test, y_test = collect_data(
            df=test_df,
            num_groups=num_groups,
            model_behavior_key=args.model_behavior_key,
            n_samples=args.n_samples,
        )
        test_data_list.append((x_test, y_test))
    num_model_behaviors = y_test.shape[-1]

    # Collect full model behaviors.
    full_df = pd.read_json(args.full_db, lines=True)
    y_full = collect_data(
        df=full_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
        collect_remaining_masks=False,
    )
    y_full = y_full.flatten()

    # Read in data frames with fitting data.
    fit_df = pd.read_json(args.fit_db, lines=True)
    fit_df["idx"] = fit_df["exp_name"].str.split("loo_idx_", expand=True)[1].astype(int)
    fit_df = fit_df.sort_values(by="idx")
    _, y_fit = collect_data(
        df=fit_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
    )

    # Get LOO scores and evaluate LDS.
    attrs_all = y_full - y_fit

    lds_mean, lds_ci = evaluate_lds(
        attrs_all=attrs_all,
        test_data_list=test_data_list,
        num_model_behaviors=num_model_behaviors,
    )
    print(f"\tLDS: {lds_mean:.2f} ({lds_ci:.2f})")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
