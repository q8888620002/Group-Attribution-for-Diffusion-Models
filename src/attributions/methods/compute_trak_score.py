"""Calcuation D-TRAK, relative IF, randomized IF"""
import os

import numpy as np
import torch

import src.constants as constants
from src.attributions.methods.attribution_utils import sum_scores_by_class
from src.datasets import create_dataset


def compute_dtrak_trak_scores(args, retraining=False, training_seeds=None):
    """Compute scores for D-TRAK, TRAK, and influence function."""
    dataset = create_dataset(dataset_name=args.dataset, train=True)

    if retraining:
        # Retraining based
        scores = np.zeros(len(dataset))

        val_grad_path = os.path.join(
            constants.OUTDIR,
            args.dataset,
            "d_track",
            "full",
            f"val_f={args.trak_behavior}_t={args.t_strategy}",
        )
        val_phi = np.memmap(
            val_grad_path,
            dtype=np.float32,
            mode="r",
            shape=(len(dataset), args.projector_dim),
        )

        for seed in training_seeds:
            removal_dir = f"{args.removal_dist}/{args.removal_dist}"
            removal_dir += f"_seed={seed}"

            train_grad_path = os.path.join(
                constants.OUTDIR,
                args.dataset,
                "d_track",
                removal_dir,
                f"train_f={args.trak_behavior}_t={args.t_strategy}",
            )
            train_phi = np.memmap(
                train_grad_path,
                dtype=np.float32,
                mode="r",
                shape=(len(dataset), args.projector_dim),
            )
            train_phi = torch.from_numpy(train_phi).cuda()

            kernel = train_phi.T @ train_phi
            kernel = kernel + 5e-1 * torch.eye(kernel.shape[0]).cuda()
            kernel = torch.linalg.inv(kernel)

            scores += val_phi @ ((train_phi @ kernel).T) / len(training_seeds)
    else:
        # retraining free TRAK/D-TRAK
        print(
            f"Loading pre-calculated grads for training set from {train_grad_path}..."
        )

        train_grad_path = os.path.join(
            constants.OUTDIR,
            args.dataset,
            "d_track",
            "full",
            f"train_f={args.trak_behavior}_t={args.t_strategy}",
        )
        train_phi = np.memmap(
            train_grad_path,
            dtype=np.float32,
            mode="r",
            shape=(len(dataset), args.projector_dim),
        )
        print(
            f"Loading pre-calculated grads for validation set from {val_grad_path}..."
        )

        val_grad_path = os.path.join(
            constants.OUTDIR,
            args.dataset,
            "d_track",
            "full",
            f"val_f={args.trak_behavior}_t={args.t_strategy}",
        )
        val_phi = np.memmap(
            val_grad_path,
            dtype=np.float32,
            mode="r",
            shape=(len(dataset), args.projector_dim),
        )

        # dstore_keys = torch.from_numpy(dstore_keys).cuda()

        kernel = train_phi.T @ train_phi
        kernel = kernel + 5e-1 * np.eye(kernel.shape[0])

        kernel = np.linalg.inv(kernel)

        scores = val_phi @ ((train_phi @ kernel).T)

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

    if args.by_class:
        coeff = -sum_scores_by_class(scores, dataset)
    else:
        coeff = -scores

    # using average of local model behavior as global score

    coeff = np.mean(coeff, axis=0)

    return coeff
