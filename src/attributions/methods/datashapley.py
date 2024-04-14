"""Function that calculate data shapley"""
import warnings

import numpy as np
from sklearn.linear_model import RidgeCV


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
    # rond: Cutoff for small singular values. The default is 1e-15.
    # Singular values less than or equal to rcond * largest_singular_value
    # are set to zero.

    a_hat_inv = np.linalg.pinv(a_hat)
    one = np.ones((dataset_size, 1))

    c = one.T @ a_hat_inv @ b_hat - v1 + v0
    d = one.T @ a_hat_inv @ one

    coef = a_hat_inv @ (b_hat - one @ (c / d))

    coef[np.abs(coef) < 1e-10] = 0
    # coef[np.abs(coef) > 50] = 50

    return coef


def kernel_shap_ridge(dataset_size, x_train, y_train, v1, v0):
    """
    Function to compute kernel shap coefficients
    following, https://github.com/shap/shap/blob/master/shap/explainers/_kernel.py

    Args:
    ----
        dataset_size: length of reference dataset size
        x_train: indices of subset, n x d
        y_train: model behavior, n x 1
        v1: model behavior with all data presented
        v0: model behavior of null subset

    Return:
    ------
        w: coefficients for kernel shap
    """

    ones = np.ones((1, dataset_size))
    zeros = np.zeros((1, dataset_size))

    X = np.concatenate((x_train, ones, zeros), axis=0)
    y = np.concatenate((y_train, np.asarray([v1, v0])), axis=0)

    # init sample kernel weights
    kernel_weights = np.concatenate(
        (np.ones(len(x_train)), np.asarray([10000.0, 10000.0])), axis=0
    )

    WX = kernel_weights[:, None] * X

    model = RidgeCV(alphas=np.linspace(1e-20, 1e-15, 5)).fit(WX, y)

    return model.coef_


def kernel_shap(dataset_size, x_train, y_train, v1, v0):
    """
    Function to compute kernel shap coefficients
    following, https://github.com/shap/shap/blob/master/shap/explainers/_kernel.py

    Args:
    ----
        dataset_size: length of reference dataset size
        x_train: indices of subset, n x d
        y_train: model behavior, n x 1
        v1: model behavior with all data presented
        v0: model behavior of null subset

    Return:
    ------
        w: coefficients for kernel shap
    """

    ones = np.ones((1, dataset_size))
    zeros = np.zeros((1, dataset_size))

    X = np.concatenate((x_train, ones, zeros), axis=0)
    y = np.concatenate((y_train, np.asarray([v1, v0])), axis=0)

    # init sample kernel weights
    kernel_weights = np.concatenate(
        (np.ones(len(x_train)), np.asarray([1e10, 1e10])), axis=0
    )

    WX = kernel_weights[:, None] * X

    try:
        w = np.linalg.solve(X.T @ WX, WX.T @ y)
    except np.linalg.LinAlgError:
        warnings.warn(
            "Linear regression equation is singular, a least squares solutions is used instead.\n"
            "To avoid this situation and get a regular matrix do one of the following:\n"
            "1) turn up the number of samples,\n"
            "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
            "3) group features together to reduce the number of inputs that need to be explained."
        )
        # XWX = np.linalg.pinv(X.T @ WX)
        # w = np.dot(XWX, np.dot(np.transpose(WX), y))
        sqrt_W = np.sqrt(kernel_weights)
        w = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]

    return w
