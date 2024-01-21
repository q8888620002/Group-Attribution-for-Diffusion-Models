"""Functions that calculate datamodel score"""
import os

import numpy as np
from sklearn.linear_model import RidgeCV

from attributions.attribution_utils import load_model_behavior
from utils import create_dataset, remove_data_by_datamodel


def datamodel(x_train, y_train, num_runs):
    """
    Function to compute datamodel coefficients with linear regression.

    Args:
    ----
        x_train: indices of subset, n x d
        y_train: model behavior, n x 1
        num_runs: number of bootstrapped times.

    Return:
    ------
        coef: stacks of coefficients for regression.
    """

    train_size = len(x_train)
    coeff = []

    for _ in range(num_runs):
        bootstrapped_indices = np.random.choice(train_size, train_size, replace=True)
        reg = RidgeCV(cv=5, alphas=[0.1, 1.0, 1e1]).fit(
            x_train[bootstrapped_indices],
            y_train[bootstrapped_indices],
        )
        coeff.append(reg.coef_)

    coeff = np.stack(coeff)

    return coeff


def compute_datamodel_scores(args, train_idx, val_idx):
    """
    Compute scores for the datamodel method.

    Args:
    ----
        args: Command line arguments.
        train_idx: Indices for the training subset.
        val_idx: Indices for the validation subset.

    Returns
    -------
        Scores calculated using the datamodel method.
    """
    full_dataset = create_dataset(dataset_name=args.dataset, train=True)
    dataset_size = len(full_dataset)
    n_subset = len(train_idx) + len(val_idx)

    X = np.zeros((n_subset, dataset_size))
    Y = np.zeros(n_subset)

    for i in range(n_subset):

        remaining_idx, _ = remove_data_by_datamodel(
            full_dataset, alpha=args.datamodel_alpha, seed=i
        )
        model_output = load_model_behavior(args, remaining_idx)

        X[i, remaining_idx] = 1
        Y[i] = model_output[args.model_behavior]

    coefficients = datamodel(X[train_idx], Y[train_idx], args.num_runs)
    return X[val_idx] @ coefficients.T
