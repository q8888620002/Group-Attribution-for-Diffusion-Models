"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse
import math
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.attributions.methods.datashapley import data_shapley
from src.ddpm_config import DatasetStats
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="evaluate Shapley values using the linear model score"
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
        "--baseline_fit_db",
        type=str,
        help="database with model behaviors for fitting baseline Shapley values",
        required=True,
    )
    parser.add_argument(
        "--fit_db",
        type=str,
        help="database with model behaviors for fitting Shapley values",
        required=True,
    )
    parser.add_argument(
        "--fit_size_factor",
        type=float,
        help="factor for scaling the baseline fitting size",
        default=1.0,
    )
    parser.add_argument(
        "--null_db",
        type=str,
        help="database with model behaviors for the null model",
        required=True,
    )
    parser.add_argument(
        "--full_db",
        type=str,
        help="database with model behaviors for the fully trained model",
        required=True,
    )
    parser.add_argument(
        "--test_size",
        type=int,
        help="number of subsets used for evaluating data attributions",
        default=100,
    )
    parser.add_argument(
        "--fit_size",
        type=int,
        nargs="*",
        help="number of subsets used for fitting baseline data attributions",
        default=[300],
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
        "--output_dir",
        type=str,
        help="directory to save the Shapley values",
        default=None,
    )
    parser.add_argument(
        "--outfile_prefix",
        type=str,
        help="output file prefix for saving the Shapley values",
        default=None,
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

    # Collect null and full model behaviors.
    null_df = pd.read_json(args.null_db, lines=True)
    y_null = collect_data(
        df=null_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
        collect_remaining_masks=False,
    )
    y_null = y_null.flatten()

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
    baseline_fit_df = pd.read_json(args.baseline_fit_db, lines=True)
    baseline_fit_df["subset_seed"] = (
        baseline_fit_df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
    )
    baseline_fit_df = baseline_fit_df.sort_values(by="subset_seed")

    fit_df = pd.read_json(args.fit_db, lines=True)
    fit_df["subset_seed"] = (
        fit_df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
    )
    fit_df = fit_df.sort_values(by="subset_seed")

    # Evaluate Shapley values with varying fitting sizes.
    baseline_lds_mean_list, baseline_lds_ci_list = [], []
    lds_mean_list, lds_ci_list = [], []
    fit_size_list = []
    for baseline_fit_size in args.fit_size:
        baseline_x_fit, baseline_y_fit = collect_data(
            df=baseline_fit_df[:baseline_fit_size],
            num_groups=num_groups,
            model_behavior_key=args.model_behavior_key,
            n_samples=args.n_samples,
        )

        fit_size = math.floor(baseline_fit_size * args.fit_size_factor)
        fit_size_list.append(fit_size)
        x_fit, y_fit = collect_data(
            df=fit_df[:fit_size],
            num_groups=num_groups,
            model_behavior_key=args.model_behavior_key,
            n_samples=args.n_samples,
        )

        baseline_attrs_all, attrs_all = [], []
        for k in range(num_model_behaviors):
            v0 = y_null[k]
            v1 = y_full[k]
            baseline_attrs = data_shapley(
                dataset_size=baseline_x_fit.shape[-1],
                x_train=baseline_x_fit,
                y_train=baseline_y_fit[:, k],
                v0=v0,
                v1=v1,
            )
            baseline_attrs_all.append(baseline_attrs.flatten())

            attrs = data_shapley(
                dataset_size=x_fit.shape[-1],
                x_train=x_fit,
                y_train=y_fit[:, k],
                v0=v0,
                v1=v1,
            )
            attrs_all.append(attrs.flatten())
        baseline_attrs_all = np.stack(baseline_attrs_all, axis=1)
        attrs_all = np.stack(attrs_all, axis=1)

        baseline_lds_mean, baseline_lds_ci = evaluate_lds(
            attrs_all=baseline_attrs_all,
            test_data_list=test_data_list,
            num_model_behaviors=num_model_behaviors,
        )
        baseline_lds_mean_list.append(baseline_lds_mean)
        baseline_lds_ci_list.append(baseline_lds_ci)

        lds_mean, lds_ci = evaluate_lds(
            attrs_all=attrs_all,
            test_data_list=test_data_list,
            num_model_behaviors=num_model_behaviors,
        )
        lds_mean_list.append(lds_mean)
        lds_ci_list.append(lds_ci)

        print(f"Baseline fit size: {baseline_fit_size}, fit size: {fit_size}")
        print(f"\tBaseline LDS: {baseline_lds_mean:.2f} ({baseline_lds_ci:.2f})")
        print(f"\tLDS: {lds_mean:.2f} ({lds_ci:.2f})")

        if args.output_dir is not None:
            outfile = f"artist_{args.outfile_prefix}_fit_size={fit_size}.npy"
            with open(os.path.join(args.output_dir, outfile), "wb") as handle:
                np.save(handle, attrs_all)

            global_rank = np.argsort(-attrs_all.mean(axis=-1), kind="stable")
            rank_file = "all_generated_images_artist_rank"
            rank_file += f"_{args.outfile_prefix}_fit_size={fit_size}.npy"
            with open(os.path.join(args.output_dir, rank_file), "wb") as handle:
                np.save(handle, global_rank)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
