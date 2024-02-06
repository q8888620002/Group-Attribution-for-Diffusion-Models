import json
import re
import numpy as np

from sklearn.linear_model import RidgeCV
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

def datamodel_modified(x_train, y_train, num_runs, alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1], k_folds=5):
    """
    Function to compute datamodel coefficients with linear regression and select the best
    based on Spearman rank correlation across a range of alphas.

    Args:
    ----
        x_train: Training data, n x d.
        y_train: Training targets, n x 1.
        num_runs: Number of bootstrapped times (not used in this version).
        alphas: List of alpha values to try.
        k_folds: Number of folds for cross-validation.

    Return:
    ------
        best_coef: The coefficients for regression resulting in the best Spearman rank correlation.
    """
    coeff = []
    train_size = len(x_train)

    for _ in range(num_runs):

        bootstrapped_indices = np.random.choice(train_size, train_size, replace=True)
        bootstrapped_x = x_train[bootstrapped_indices]
        bootstrapped_y = y_train[bootstrapped_indices]

        kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
        best_alpha = alphas[0]
        best_corr = -1

        for alpha in alphas:
            corr_sum = 0

            for train_index, val_index in kf.split(bootstrapped_x):
                x_train_k, x_val_k = bootstrapped_x[train_index], bootstrapped_x[val_index]
                y_train_k, y_val_k = bootstrapped_y[train_index], bootstrapped_y[val_index]

                model = Ridge(alpha=alpha)
                model.fit(x_train_k, y_train_k)
                predictions = model.predict(x_val_k)
                corr, _ = spearmanr(predictions, y_val_k)
                corr_sum += corr

            avg_corr = corr_sum / k_folds

            if avg_corr > best_corr:
                best_corr = avg_corr
                best_alpha = alpha

        # Train final model on the entire training set with the best alpha
        best_model = Ridge(alpha=best_alpha)
        best_model.fit(bootstrapped_x, bootstrapped_y)
        best_coef = best_model.coef_
        coeff.append(best_coef)

    return np.stack(coeff)

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
        reg = RidgeCV(cv=5, alphas=[1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]).fit(
            x_train[bootstrapped_indices],
            y_train[bootstrapped_indices],
        )
        coeff.append(reg.coef_)

    coeff = np.stack(coeff)

    return coeff


def compute_datamodel_scores(subset_indices, global_behavior ,train_seeds, test_seeds):
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

    coeff = datamodel_modified(X[:len(train_seeds), :], Y[:len(train_seeds)], 5)

    return np.mean(X[len(train_seeds):, :] @ coeff.T, axis=1)


if __name__ == "__main__":

    random_states = [42, 10, 5, 2, 20]

    results  = {
        "retrain_vs_train_fid": [],
        "gd_vs_train_fid" : []
    }

    for random_state in random_states:
        subset_indices = {
            'retrain_vs_train_fid': {}, 
            'gd_vs_train_fid': {}
        }
        global_behavior = {
            'retrain_vs_train_fid': {}, 
            'gd_vs_train_fid': {}
        }

        model_behavior = "/gscratch/aims/diffusion-attr/results_ming/cifar/full_model_db.jsonl"

        with open(model_behavior, "r") as f:
            for line in f:
                row = json.loads(line)

                exp_name = row.get("exp_name")
                sample_dir = row.get("sample_dir", "")
                match = re.search(r"_seed=(\d+)", str(sample_dir))

                if exp_name in subset_indices and match:
                    seed = int(match.group(1))

                    if seed not in subset_indices[exp_name].keys():
                        subset_indices[exp_name][seed] = np.asarray(row.get("remaining_idx"))
                        global_behavior[exp_name][seed] = float(row['fid_value'])

        retrain_seeds =  list(subset_indices["retrain_vs_train_fid"].keys())
        gd_seets = list(subset_indices["gd_vs_train_fid"].keys())

        common_seeds = list(set(gd_seets) & set(retrain_seeds))

        rng = np.random.RandomState(random_state)
        rng.shuffle(common_seeds)

        test_seeds = [seed for seed in common_seeds[:64]]
        train_seeds = { 
            "retrain_vs_train_fid": [seed for seed in retrain_seeds if seed not in test_seeds][:30],
            "gd_vs_train_fid": [seed for seed in gd_seets if seed not in test_seeds]              
        }

        test_values = [global_behavior["retrain_vs_train_fid"][seed] for seed in test_seeds]

        for method in ["retrain", "gd"]:
            pre_fix = f"{method}_vs_train_fid"

            predicted_behavior = compute_datamodel_scores(
                subset_indices[pre_fix],
                global_behavior[pre_fix], 
                train_seeds[pre_fix], 
                test_seeds
            )
            rho = spearmanr(predicted_behavior , test_values).statistic

            results[pre_fix].append(rho)
        
    print(f"LDS: retrain, {np.mean(results['retrain_vs_train_fid']), np.std(results['retrain_vs_train_fid'])}")
    print(f"LDS: gd, {np.mean(results['gd_vs_train_fid']), np.std(results['gd_vs_train_fid'])}")
