"""Functions that calculate datamodel score"""
import numpy as np
from sklearn.linear_model import RidgeCV

from src.datasets import create_dataset


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


def compute_datamodel_scores(args, model_behavior_all, train_idx, val_idx):
    """
    Compute scores for the datamodel method.

    Args:
    ----
        args: Command line arguments.
        model_behavior_all: pre_calculated model behavior for each subset.
        train_idx: Indices for the training subset.
        val_idx: Indices for the validation subset.

    Returns
    -------
        Scores calculated using the datamodel method.
    """
    total_data_num = len(create_dataset(dataset_name=args.dataset, train=True))

    train_val_index = train_idx + val_idx

    X = np.zeros((len(train_val_index), total_data_num))
    Y = np.zeros(len(train_val_index))

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

    coeff = datamodel(X[train_idx, :], Y[train_idx], args.num_runs)

    return X[val_idx, :] @ coeff.T
