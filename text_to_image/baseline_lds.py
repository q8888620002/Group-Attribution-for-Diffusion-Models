"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, spearmanr

from src.ddpm_config import DatasetStats
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate data attribution methods using the linear model score"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset name",
        choices=["artbench_post_impressionism"],
        default="artbench_post_impressionism",
    )
    parser.add_argument(
        "--test_db",
        type=str,
        help="database with model behaviors for evaluating data attributions",
        required=True,
    )
    parser.add_argument(
        "--group",
        type=str,
        default="artist",
        choices=["artist", "filename"],
        help="unit for how to group images",
    )
    parser.add_argument(
        "--baseline_dir",
        type=str,
        help="directory containing baseline attribution values",
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="number of subsets used for evaluating data attributions",
        default=150,
    )
    parser.add_argument(
        "--model_behavior_key",
        type=str,
        help="key to query model behavior in the test database",
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
        "--bootstrap_size",
        type=int,
        help="number of bootstraps for reporting test statistics",
        default=150,
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


def main(args):
    """Main function."""
    if args.dataset == "artbench_post_impressionism":
        dataset_stats = DatasetStats.artbench_post_impressionism_stats
        num_groups = dataset_stats["num_groups"]
    else:
        raise ValueError

    # Read in and preprocess databases as data frames.
    test_df = pd.read_json(args.test_db, lines=True)
    test_df["subset_seed"] = (
        test_df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
    )
    test_df = test_df.sort_values(by="subset_seed")

    test_subset_seeds = [i for i in range(args.test_size)]
    test_df = test_df[test_df["subset_seed"].isin(test_subset_seeds)]
    assert len(test_df) == args.test_size

    # Collect test data.
    x_test, y_test = collect_data(
        df=test_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
    )
    if args.model_behavior_key in ["simple_loss", "nrmse"]:
        # The directionality of these model behaviors is opposite of pixel and CLIP
        # similarity, so we flip their signs.
        y_test *= -1.0
    num_model_behaviors = y_test.shape[-1]

    baseline_list = [
        "avg_pixel_similarity",
        "max_pixel_similarity",
        "avg_clip_similarity",
        "max_clip_similarity",
    ]
    baseline_list = [f"{args.group}_{baseline}" for baseline in baseline_list]

    for baseline in baseline_list:
        baseline_file = os.path.join(args.baseline_dir, f"{baseline}.npy")
        with open(baseline_file, "rb") as handle:
            attrs_all = np.load(handle)
            assert attrs_all.shape[0] == num_groups
            assert attrs_all.shape[-1] >= num_model_behaviors
            if num_model_behaviors == 1:
                attrs_all = np.mean(attrs_all, axis=-1, keepdims=True)

        # LDS with bootstrap.
        def my_lds(idx):
            x_boot = x_test[idx, :]
            lds_list = []
            for k in range(num_model_behaviors):
                y_boot = y_test[idx, k]
                lds_list.append(
                    spearmanr(x_boot @ attrs_all[:, k], y_boot).statistic * 100
                )
            return np.mean(lds_list)

        boot_result = bootstrap(
            data=(list(range(len(x_test))),),
            statistic=my_lds,
            n_resamples=args.bootstrap_size,
            random_state=0,
        )
        boot_mean = boot_result.bootstrap_distribution.mean()
        boot_se = boot_result.standard_error
        print(f"{baseline}")
        print(f"\tLDS: {boot_mean:.3f} ({boot_se:.3f})")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
