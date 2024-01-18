"""This script servse for testing data attribution calculation"""
import numpy as np
from sklearn.linear_model import RidgeCV


def __kernel_shap_test(
    x_train: np.array,
    y_train: np.array,
):
    """
    Method for calculating shapley value with kernel shap closed form solution.
    Confidence interval is calculated based on bootstrapped sampling
    Args:
        x_train: np array data indices
        y_train: np arraycorresponding model behavior

    Return:
        estimated data shapley value with confidence interval.

    """
    kernelshap_coeff = []
    train_size = x_train.shape[0]
    dataset_size = x_train.shape[1]

    null_model_output = 0
    full_model_output = 3.0

    for i in range(5):
        bootstrapped_indices = np.random.choice(train_size, train_size)

        a_hat = np.zeros((dataset_size, dataset_size))
        b_hat = np.zeros((dataset_size, 1))

        for j in range(train_size):
            a_hat += np.outer(x_train[bootstrapped_indices][j], x_train[bootstrapped_indices][j])
            b_hat += (x_train[bootstrapped_indices][j] * (y_train[bootstrapped_indices][j] - null_model_output))[:, None]

        a_hat /= train_size
        b_hat /= train_size

        # Using np.linalg.pinv instead of np.linalg.inv in case of singular matrix
        a_hat_inv = np.linalg.pinv(a_hat)
        one = np.ones((dataset_size, 1))

        c = one.T@a_hat_inv@b_hat - full_model_output + null_model_output
        d = one.T@a_hat_inv@one

        coef = a_hat_inv@(b_hat - one@(c/d))

        kernelshap_coeff.append(coef)

    kernelshap_coeff = np.stack(kernelshap_coeff)
    scores = x_train @ kernelshap_coeff

    return scores

def __datamodel_test(
    x_train: np.array,
    y_train: np.array,
):
    train_size = x_train.shape[0]
    dataset_size = x_train.shape[1]
    datamodel_coeff = []

    for i in range(5):
        bootstrapped_indices = np.random.choice(train_size, train_size)
        reg = RidgeCV(
            cv=5,
            alphas=[0.1, 1.0, 1e1],
        ).fit(x_train[bootstrapped_indices], y_train[bootstrapped_indices])
        datamodel_coeff.append(reg.coef_)

    datamodel_coeff = np.stack(datamodel_coeff)
    scores = x_train @ datamodel_coeff.T

    return scores



if __name__ == "__main__":

    dataset_size = 1000
    train_size = 100

    x_train = np.random.randint(2, size=(train_size, dataset_size))
    y_train = np.random.random(train_size)
    data_score = __kernel_shap_test(x_train, y_train)
