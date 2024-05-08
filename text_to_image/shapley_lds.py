"""Evaluate data attributions using the linear datamodel score (LDS)."""

import argparse

import numpy as np
import pandas as pd
from scipy.stats import bootstrap, spearmanr
from tqdm import tqdm

from src.attributions.methods.datashapley import data_shapley
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
        "--fit_db",
        type=str,
        help="database with model behaviors for fitting data attributions",
        required=True,
    )
    parser.add_argument(
        "--remove_overlap",
        action="store_true",
        help="whether to remove overlap in removal seeds in the test and fit databases",
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
        help="number of subsets used for fitting data attributions",
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
        "--bootstrap_size",
        type=int,
        help="number of bootstraps for reporting test statistics",
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

    fit_df = pd.read_json(args.fit_db, lines=True)
    fit_df["subset_seed"] = (
        fit_df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
    )
    fit_df = fit_df.sort_values(by="subset_seed")

    test_subset_seeds = [i for i in range(args.test_size)]
    test_df = test_df[test_df["subset_seed"].isin(test_subset_seeds)]
    assert len(test_df) == args.test_size

    if args.remove_overlap:
        fit_df = fit_df[~fit_df["subset_seed"].isin(test_subset_seeds)]

    null_df = pd.read_json(args.null_db, lines=True)
    full_df = pd.read_json(args.full_db, lines=True)

    # Collect null and full model behaviors.
    y_null = collect_data(
        df=null_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
        collect_remaining_masks=False,
    )
    y_null = y_null.flatten()
    y_full = collect_data(
        df=full_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
        collect_remaining_masks=False,
    )
    y_full = y_full.flatten()

    # Collect test data.
    x_test, y_test = collect_data(
        df=test_df,
        num_groups=num_groups,
        model_behavior_key=args.model_behavior_key,
        n_samples=args.n_samples,
    )
    num_model_behaviors = y_test.shape[-1]

    # Collect fitting data with varying sizes.
    x_fit_list, y_fit_list = [], []
    for n in args.fit_size:
        x_fit, y_fit = collect_data(
            df=fit_df[:n],
            num_groups=num_groups,
            model_behavior_key=args.model_behavior_key,
            n_samples=args.n_samples,
        )
        x_fit_list.append(x_fit)
        y_fit_list.append(y_fit)

    # Evaluate LDS with varying fitting sizes.
    for i in range(len(args.fit_size)):

        # Fit data shapley values for all the model behaviors.
        x_fit = x_fit_list[i]
        y_fit_all = y_fit_list[i]
        attrs_all = []
        for k in tqdm(range(num_model_behaviors)):
            v0 = y_null[k]
            v1 = y_full[k]
            y_fit = y_fit_all[:, k]
            attrs = data_shapley(
                dataset_size=x_fit.shape[-1], x_train=x_fit, y_train=y_fit, v0=v0, v1=v1
            )
            attrs_all.append(attrs)
        attrs_all = np.stack(attrs_all, axis=1)

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
        print(f"Fitting size = {args.fit_size[i]}")
        print(f"\tLDS: {boot_mean:.3f} ({boot_se:.3f})")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
