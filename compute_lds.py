"""Calculate LDS for global model behavior"""
import json

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold


def datamodel_simple(x_train, y_train):
    """
    Function to compute datamodel coefficients with linear regression.

    Args:
    ----
        x_train: Training data, n x d.
        y_train: Training targets, n x 1.
        alphas: List of alpha values to try.
        k_folds: Number of folds for cross-validation.

    Return:
    ------
        best_coef: The coefficients for regression.
    """

    model = RidgeCV(alphas=np.linspace(0.01, 10, 100))

    # Fit the model to the training data
    model.fit(x_train, y_train)

    # Best alpha and coefficients
    best_alpha = model.alpha_
    coeffs = model.coef_

    return best_alpha, coeffs


def compute_datascore(subset_indices, global_behavior, train_seeds, test_seeds):
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
    total_data_num = 50000

    all_seeds = train_seeds + test_seeds

    X = np.zeros((len(all_seeds), total_data_num))
    Y = np.zeros((len(all_seeds)))

    for idx, seed in enumerate(all_seeds):

        X[idx, subset_indices[seed]] = 1
        Y[idx] = global_behavior[seed]

    _, coeffs = datamodel_simple(X[: len(train_seeds), :], Y[: len(train_seeds)])

    x_test = X[len(train_seeds) :, :]

    return x_test @ coeffs.T


if __name__ == "__main__":

    results = {"retrain_vs_train": [], "gd_vs_train": []}
    subset_indices = {"retrain_vs_train": {}, "gd_vs_train": {}}
    global_behavior = {"retrain_vs_train": {}, "gd_vs_train": {}}

    model_behavior = (
        "/gscratch/aims/diffusion-attr/results_ming/cifar/global_score_2_24.jsonl"
    )

    with open(model_behavior, "r") as f:
        for line in f:
            row = json.loads(line)

            exp_name = row.get("exp_name")
            seed = row.get("removal_seed", "")

            if (
                exp_name in subset_indices
                and seed not in subset_indices[exp_name].keys()
            ):
                subset_indices[exp_name][seed] = np.asarray(row.get("remaining_idx"))
                global_behavior[exp_name][seed] = float(row["fid_value"])

    retrain_seeds = list(subset_indices["retrain_vs_train"].keys())
    gd_seets = list(subset_indices["gd_vs_train"].keys())

    common_seeds = list(set(gd_seets) & set(retrain_seeds))

    # Split test set into 3 folds.

    kf = KFold(n_splits=3, shuffle=True, random_state=42)

    for _, test_index in kf.split(common_seeds):

        test_seeds = [common_seeds[i] for i in test_index]

        train_seeds = {
            "retrain_vs_train": [
                seed for seed in retrain_seeds if seed not in test_seeds
            ],
            "gd_vs_train": [seed for seed in gd_seets if seed not in test_seeds],
        }

        test_values = [global_behavior["retrain_vs_train"][seed] for seed in test_seeds]

        for method in ["retrain", "gd"]:
            pre_fix = f"{method}_vs_train"

            pred_score = compute_datascore(
                subset_indices[pre_fix],
                global_behavior[pre_fix],
                train_seeds[pre_fix],
                test_seeds,
            )

            results[pre_fix].append(spearmanr(pred_score, test_values).statistic)

    print(
        f"LDS: retrain, Mean:{np.mean(results['retrain_vs_train'])}"
        f"SE:{1.96*np.std(results['retrain_vs_train'])/np.sqrt(len(test_values))}"
    )
    print(
        f"LDS: gd, mean:{np.mean(results['gd_vs_train'])}"
        f"SE:{1.96*np.std(results['gd_vs_train'])/np.sqrt(len(test_values))}"
    )
