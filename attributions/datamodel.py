"""Functions that calculate datamodel score"""
import numpy as np
from sklearn.linear_model import RidgeCV


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


def compute_datamodel_scores(args, model_behavior, train_idx, val_idx):
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
    total_num_data = len(model_behavior[0].get(["remaining_idx"])) + len(
        model_behavior[0].get(["removed_idx"])
    )

    train_val_index = train_idx + val_idx
    X = np.zeros((len(train_val_index), total_num_data))
    Y = np.zeros(len(train_val_index))

    for i in train_val_index:

        remaining_idx = model_behavior[i].get("remaining_idx")
        X[i, remaining_idx] = 1
        Y[i] = model_behavior[i].get(args.model_behavior)

    coefficients = datamodel(X[train_idx, :], Y[train_idx], args.num_runs)
    return X[val_idx, :] @ coefficients.T
