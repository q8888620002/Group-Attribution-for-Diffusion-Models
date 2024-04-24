"""
FID calculation based on pytorch-fid[1]

[1]: https://github.com/mseitzer/pytorch-fid
"""
import os
import pickle as pkl

import numpy as np
import torch
from pytorch_fid.fid_score import (
    calculate_fid_given_paths,
    calculate_frechet_distance,
    compute_statistics_of_path,
)
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

import src.constants as constants


def calculate_fid(dataset, images_dataset, batch_size, device, reference_dir=None):
    """Calculate fid given a set of generated images."""

    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    inceptionNet = InceptionV3([block_idx]).to(device)
    inceptionNet.eval()  # Important: .eval() is needed to turn off dropout
    pre_computed_path = os.path.join(constants.DATASET_DIR, dataset, "precomputed")

    # Calculate mu and sigma for reference images
    # Fix bugs when overwritting pkl when specifying  reference_dir
    if reference_dir is not None:
        mu, sigma = compute_statistics_of_path(
            reference_dir, inceptionNet, batch_size, dims, device
        )
        stats = {}
        stats["mu"] = mu
        stats["sigma"] = sigma

        os.makedirs(pre_computed_path, exist_ok=True)

        with open(os.path.join(pre_computed_path, "stats.pkl"), "wb") as file:
            pkl.dump(stats, file)

        print(
            f"Pre-calculated mean and mu from {reference_dir} are saved at "
            f"{pre_computed_path}."
        )
    else:
        try:
            with open(
                os.path.join(pre_computed_path, "stats.pkl"),
                "rb",
            ) as file:
                cifar_train = pkl.load(file)
            mu, sigma = cifar_train["mu"], cifar_train["sigma"]

        except FileNotFoundError:
            raise FileNotFoundError(
                f"No pre-calculated stats at {pre_computed_path} found."
            )

    mu1, sigma1 = compute_features_stats(
        images_dataset, inceptionNet, batch_size, dims, device
    )

    fid = calculate_frechet_distance(mu1, sigma1, mu, sigma)

    return fid


def compute_features_stats(images, model, batch_size, dims, device):
    """Function to extract InceptionNet Features"""

    batch_size_list = [batch_size] * (len(images) // batch_size)
    remaining_sample_size = len(images) % batch_size

    if remaining_sample_size > 0:
        batch_size_list.append(remaining_sample_size)

    dims = 2048
    pred_arr = np.empty((len(images), dims))

    start_idx = 0

    for batch_size in tqdm(batch_size_list):
        batch = images[start_idx : start_idx + batch_size, :, :, :].to(device)

        with torch.no_grad():
            pred = model(batch)[0]

        # If model output is not scalar, apply global spatial average pooling.
        # This happens if you choose a dimensionality not equal 2048.

        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()
        pred_arr[start_idx : start_idx + pred.shape[0]] = pred
        start_idx = start_idx + pred.shape[0]

    mu = np.mean(pred_arr, axis=0)
    sigma = np.cov(pred_arr, rowvar=False)

    return mu, sigma
