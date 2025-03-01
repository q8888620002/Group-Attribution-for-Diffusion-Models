"""Calcuation D-TRAK, relative IF, randomized IF"""
import os

import numpy as np
import torch

import time
import src.constants as constants
from src.attributions.methods.attribution_utils import aggregate_by_class
from src.datasets import ImageDataset, create_dataset


def compute_gradient_scores(args, retraining=False, training_seeds=None):
    """Compute scores for D-TRAK, TRAK, and influence function."""
    dataset = create_dataset(dataset_name=args.dataset, train=True)

    sample_dataset = ImageDataset(args.sample_dir)

    if args.gradient_type == "d_trak":
        model_behavior = "mean-squared-l2-norm"
        t_strategy = "uniform"
    else:
        model_behavior = "loss"
        t_strategy = "uniform"

    if args.gradient_type == "journey_trak":
        val_grad_path = os.path.join(
            constants.OUTDIR,
            args.dataset,
            "d_trak",
            "full",
            f"gen_f=loss_t=uniform_k={args.k_partition}_d={args.projector_dim}",
        )
    else:
        val_grad_path = os.path.join(
            args.sample_dir,
            "d_trak",
            f"reference_f={model_behavior}_t={t_strategy}_k={args.k_partition}_d={args.projector_dim}",
        )
    print(f"Loading pre-calculated grads for validation set from {val_grad_path}...")

    n_val_samples =  args.sample_size if args.gradient_type == "journey_trak" else len(sample_dataset)

    val_phi = np.memmap(
        val_grad_path,
        dtype=np.float32,
        mode="r",
        shape=(n_val_samples, args.projector_dim),
    )
    val_phi = val_phi[: args.sample_size]

    if retraining:
        # Retraining based TODO
        scores = np.zeros(len(dataset))

        for seed in training_seeds:
            removal_dir = f"{args.removal_dist}/{args.removal_dist}"
            removal_dir += f"_seed={seed}"

            train_grad_path = os.path.join(
                constants.OUTDIR,
                args.dataset,
                "d_track",
                removal_dir,
                f"train_f={args.trak_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
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
        # retraining free gradient methods

        train_grad_dir = os.path.join(constants.OUTDIR, args.dataset, "d_trak", "full")
        train_grad_path = os.path.join(
            train_grad_dir,
            f"train_f={model_behavior}_t={t_strategy}_k={args.k_partition}_d={args.projector_dim}",
        )
        kernel_path = os.path.join(
            train_grad_dir, f"kernel_train_f={model_behavior}_t={t_strategy}_k={args.k_partition}_d={args.projector_dim}.npy"
        )

        print(
            f"Loading pre-calculated grads for training set from {train_grad_path}..."
        )
        train_phi = np.memmap(
            train_grad_path,
            dtype=np.float32,
            mode="r",
            shape=(len(dataset), args.projector_dim),
        )

        if os.path.isfile(kernel_path):
            # Check if the kernel file exists
            print("Kernel file exists. Loading...")
            kernel = np.load(kernel_path)
        else:
            starttime = time.time()
            kernel = train_phi.T @ train_phi
            kernel = kernel + 5e-1 * np.eye(kernel.shape[0])
            kernel = np.linalg.inv(kernel)
            np.save(kernel_path, kernel)
            print(time.time() - starttime)

        if args.gradient_type == "vanilla_gradient":
            train_phi = train_phi / np.linalg.norm(train_phi, axis=1, keepdims=True)
            val_phi = val_phi / np.linalg.norm(val_phi, axis=1, keepdims=True)
            scores = np.dot(val_phi, train_phi.T)
        else:
            if args.gradient_type == "relative_if":
                magnitude = np.linalg.norm((train_phi @ kernel).T, axis=0)
            elif args.gradient_type == "renormalized_if":
                magnitude = np.linalg.norm(train_phi.T, axis=0)
            else:
                magnitude = 1.0

            scores = val_phi @ ((train_phi @ kernel).T) / magnitude

    # Using the average as coefficients
    if args.model_behavior_key not in ["ssim", "nrmse", "diffusion_loss"]:
        coeff = np.mean(scores, axis=0)
    else:
        coeff = scores

    if args.by_class:
        coeff = aggregate_by_class(coeff, dataset, args.by)
    else:
        coeff = scores

    return coeff
