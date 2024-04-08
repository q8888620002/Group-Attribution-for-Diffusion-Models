"""Calcuation D-TRAK, relative IF, randomized IF"""
import os

import numpy as np
import torch

import src.constants as constants
from src.datasets import create_dataset


def sum_scores_by_class(scores, dataset):
    """
    Sum scores by classes and return group-based scores

    :param scores: sample based coefficients
    :param dataset: dataset
    :return: Numpy array with summed scores, indexed by label.
    """
    # Initialize a dictionary to accumulate scores for each label
    score_sum_by_label = {}
    for score, (_, label) in zip(scores, dataset):
        score_sum_by_label[label] = score_sum_by_label.get(label, 0) + score

    result_array = np.zeros(max(score_sum_by_label.keys()) + 1)
    for label, sum_score in score_sum_by_label.items():
        result_array[label] = sum_score

    return result_array


def compute_dtrak_trak_scores(args, retraining=False, training_seeds=None):
    """Compute scores for D-TRAK, TRAK, and influence function."""
    dataset = create_dataset(dataset_name=args.dataset, train=True)

    if retraining:
        # Retraining based
        scores = np.zeros(len(dataset))

        for seed in training_seeds:
            removal_dir = f"{args.removal_dist}/{args.removal_dist}"
            removal_dir += f"_seed={seed}"

            grad_result_dir = os.path.join(
                constants.OUTDIR,
                args.dataset,
                "d_track",
                removal_dir,
                f"f={args.trak_behavior}_t={args.t_strategy}",
            )
            print(f"Loading pre-calculated gradients from {grad_result_dir}...")

            dstore_keys = np.memmap(
                grad_result_dir,
                dtype=np.float32,
                mode="r",
                shape=(len(dataset), args.projector_dmi),
            )
            dstore_keys = torch.from_numpy(dstore_keys).cuda()

            kernel = dstore_keys.T @ dstore_keys
            kernel = kernel + 5e-1 * torch.eye(kernel.shape[0]).cuda()
            kernel = torch.linalg.inv(kernel)

            scores += dstore_keys @ ((dstore_keys @ kernel).T) / len(training_seeds)
    else:
        # retraining free TRAK/D-TRAK
        grad_result_dir = os.path.join(
            constants.OUTDIR,
            args.dataset,
            "d_track",
            "full",
            f"f={args.trak_behavior}_t={args.t_strategy}",
        )
        print(f"Loading pre-calculated gradients from {grad_result_dir}...")

        dstore_keys = np.memmap(
            grad_result_dir,
            dtype=np.float32,
            mode="r",
            shape=(len(dataset), args.projector_dim),
        )
        # dstore_keys = torch.from_numpy(dstore_keys).cuda()

        kernel = dstore_keys.T @ dstore_keys
        kernel = kernel + 5e-1 * np.eye(kernel.shape[0])

        kernel = np.linalg.inv(kernel)

        scores = dstore_keys @ ((dstore_keys @ kernel).T)

        # TBD
        #   Normalize based on the meganitude.

        #     if args.attribution_method == "relative_if":
        #         magnitude = np.linalg.norm(dstore_keys @ kernel)
        #     elif args.attribution_method == "randomized_if":
        #         magnitude = np.linalg.norm(dstore_keys)
        #     else:
        #         magnitude = 1

        #     scores[i] = score.cpu().numpy() / magnitude
        # scores = np.ones(len(dataset))

    if args.dataset == "cifar100":
        coeff = -sum_scores_by_class(scores, dataset)
    else:
        coeff = -scores

    return coeff
