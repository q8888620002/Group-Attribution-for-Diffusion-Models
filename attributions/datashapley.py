"""Function that calculate data shapley"""
import os

import numpy as np


def data_shapley(dataset_size, x_train, y_train, v1, v0, num_runs):
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
        num_runs: number of bootstrapped times.

    Return:
    ------
        coef: coefficients for kernel shap
    """

    train_size = len(x_train)
    coeff = []

    for _ in range(num_runs):

        bootstrapped_indices = np.random.choice(train_size, train_size, replace=True)

        x_train_boot = x_train[bootstrapped_indices]
        y_train_boot = y_train[bootstrapped_indices]

        a_hat = np.zeros((dataset_size, dataset_size))
        b_hat = np.zeros((dataset_size, 1))

        for j in range(train_size):
            a_hat += np.outer(x_train_boot[j], x_train_boot[j])
            b_hat += (x_train_boot[j] * (y_train_boot[j] - v0))[:, None]

        a_hat /= train_size
        b_hat /= train_size

        # Using np.linalg.pinv instead of np.linalg.inv in case of singular matrix
        a_hat_inv = np.linalg.pinv(a_hat)
        one = np.ones((dataset_size, 1))

        c = one.T @ a_hat_inv @ b_hat - v1 + v0
        d = one.T @ a_hat_inv @ one

        coef = a_hat_inv @ (b_hat - one @ (c / d))
        coeff.append(coef)

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

    total_data_num = len(model_behavior_all[0]["remaining_idx"] + model_behavior_all[0]["removed_idx"])

    train_val_index = train_idx + val_idx
    X = np.zeros((len(train_val_index), total_data_num))
    Y = np.zeros(len(train_val_index))

    # Load v(0) and v(1) for Shapley values
    # null_behavior_dir = os.path.join(
    #     args.outdir, args.dataset, args.method, "wo_train"
    # )
    # full_behavior_dir = os.path.join(
    #     args.outdir, args.dataset, args.method,
    # )
    # v0 = None
    # v1 = None
    
    v1, v0 = 1, 20

    if v0 is None or v1 is None:
        print("Warning: full or null behaviors were not found.")

    for i in train_val_index:

        remaining_idx = model_behavior_all[i].get("remaining_idx")
        X[i, remaining_idx] = 1
        Y[i] = model_behavior_all[i].get(args.model_behavior)

    coeff = data_shapley(
        total_data_num, X[train_idx, :], Y[train_idx], v1, v0, args.num_runs
    )
    return X[val_idx, :] @ coeff.T
