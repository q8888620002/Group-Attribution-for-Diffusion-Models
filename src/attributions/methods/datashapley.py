"""Function that calculate data shapley"""
import os

import numpy as np

from src.attributions.methods.attribution_utils import load_filtered_behaviors
from src.datasets import create_dataset


def data_shapley(dataset_size, x_train, y_train, v1, v0):
    """
    Function to compute kernel shap coefficients with closed form solution
    of Shapley from equation (7) in
    https://proceedings.mlr.press/v130/covert21a/covert21a.pdf

    Args:
    ----
        dataset_size: length of reference dataset size
        x_train: indices of subset, n x d
        y_train: model behavior, n x 1
        v1: model behavior with all data presented
        v0: model behavior of null subset

    Return:
    ------
        coef: coefficients for kernel shap
    """

    train_size = len(x_train)

    a_hat = np.dot(x_train.T, x_train) / train_size
    b_hat = np.dot(x_train.T, (y_train - v0).reshape(-1, 1)) / train_size

    # Using np.linalg.pinv instead of np.linalg.inv in case of singular matrix
    a_hat_inv = np.linalg.pinv(a_hat)
    one = np.ones((dataset_size, 1))

    c = one.T @ a_hat_inv @ b_hat - v1 + v0
    d = one.T @ a_hat_inv @ one

    coef = a_hat_inv @ (b_hat - one @ (c / d))

    return coef


def compute_shapley_scores(args, model_behavior_all, train_idx, val_idx):
    """
    Compute scores for the data shapley.

    Args:
    ----
        args: Command line arguments.
        model_behavior_all: pre_calculated model behavior for each subset.
        train_idx: Indices for the training subset.
        val_idx: Indices for the validation subset.

    Returns
    -------
        Scores calculated using the data shapley.
    """

    total_data_num = len(create_dataset(dataset_name=args.dataset, train=True))

    train_val_index = train_idx + val_idx
    X = np.zeros((len(train_val_index), total_data_num))
    Y = np.zeros(len(train_val_index))

    # Load v(0) and v(1) for Shapley values
    null_behavior_path = os.path.join(
        args.outdir, args.dataset, args.method, "null_model_behavior.jsonl"
    )
    full_behavior_dir = os.path.join(
        args.outdir, args.dataset, args.method, "full_model_behavior.jsonl"
    )

    model_behavior_full = load_filtered_behaviors(full_behavior_dir, args.exp_name)
    model_behavior_null = load_filtered_behaviors(null_behavior_path, args.exp_name)

    v0 = (
        model_behavior_null[0].get(args.model_behavior)
        if model_behavior_null and args.model_behavior in model_behavior_null[0]
        else None
    )
    v1 = (
        model_behavior_full[0].get(args.model_behavior)
        if model_behavior_full and args.model_behavior in model_behavior_full[0]
        else None
    )

    if v0 is None or v1 is None:
        raise ValueError("Warning: full or null behaviors were not found.")

    for i in train_val_index:
        try:

            remaining_idx = model_behavior_all[i].get("remaining_idx", [])
            removed_idx = model_behavior_all[i].get("removed_idx", [])

            assert total_data_num == len(remaining_idx) + len(
                removed_idx
            ), "Total data number mismatch."

            X[i, remaining_idx] = 1
            Y[i] = model_behavior_all[i].get(args.model_behavior)

        except AssertionError as e:
            # Handle cases where total_data_num does not match the sum of indices
            print(f"AssertionError for index {i}: {e}")

    coeff = data_shapley(
        total_data_num, X[train_idx, :], Y[train_idx], v1, v0, args.num_runs
    )
    return X[val_idx, :] @ coeff.T
