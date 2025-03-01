"""Functions for Data Banzhaf."""
import numpy as np


def data_banzhaf(x_train, y_train):
    """
    Function to compute kernel shap coefficients with closed form solution
    of Shapley from equation (7) in
    https://proceedings.mlr.press/v130/covert21a/covert21a.pdf

    Args:
    ----
        x_train: indices of subset, n x d
        y_train: model behavior, n x 1

    Return:
    ------
        coef: coefficients for KernelBanzhaf.
    """
    shifted_x_train = x_train - 0.5  # Convert {0, 1} to {-1/2, 1/2}
    coef = np.linalg.lstsq(
        shifted_x_train.T @ shifted_x_train,
        shifted_x_train.T @ y_train,
        rcond=None,
    )[0]
    return coef
