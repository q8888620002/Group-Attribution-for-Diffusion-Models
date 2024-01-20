"""Utility functions for data attribution calculation."""
import glob

import clip
import numpy as np
import torch
from PIL import Image
from sklearn.linear_model import RidgeCV

# from .utils import remove_data_by_datamodel, remove_data_by_shapley

device = "cpu"
clip_model, clip_transform = clip.load("ViT-B/32", device=device)


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


def data_shapley(dataset_size, x_train, y_train, v1, v0, num_runs):
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
        num_runs: number of bootstrapped times.

    Return:
    ------
        coef: coefficients for kernel shap
    """

    train_size = len(x_train)
    coeff = []

    for _ in range(num_runs):

        bootstrapped_indices = np.random.choice(train_size, train_size, replace=True)

        x_train_boot = x_train[bootstrapped_indices]
        y_train_boot = y_train[bootstrapped_indices]

        a_hat = np.zeros((dataset_size, dataset_size))
        b_hat = np.zeros((dataset_size, 1))

        for j in range(train_size):
            a_hat += np.outer(x_train_boot[j], x_train_boot[j])
            b_hat += (x_train_boot[j] * (y_train_boot[j] - v0))[:, None]

        a_hat /= train_size
        b_hat /= train_size

        # Using np.linalg.pinv instead of np.linalg.inv in case of singular matrix
        a_hat_inv = np.linalg.pinv(a_hat)
        one = np.ones((dataset_size, 1))

        c = one.T @ a_hat_inv @ b_hat - v1 + v0
        d = one.T @ a_hat_inv @ one

        coef = a_hat_inv @ (b_hat - one @ (c / d))
        coeff.append(coef)

    return coef


def process_images_clip(file_list):
    """Function to load and process images with clip transform"""
    images = []
    for filename in file_list:
        image = Image.open(filename)
        image = clip_transform(image).unsqueeze(0).to(device)
        images.append(image)
    return torch.cat(images, dim=0)


def process_images_np(file_list):
    """Function to load and process images into numpy"""
    images = []
    for filename in file_list:
        image = Image.open(filename).convert("RGB")
        image = np.array(image).astype(np.float32)
        images.append(image)
    return np.stack(images)


def clip_score(sample_dir, reference_dir):
    """
    Function that calculate CLIP score, cosine similarity between images1 and images2

    Args:
    ----
        sample_dir: directory of the first set of images.
        reference_dir: directory of the second set of images.

    Return:
    ------
        Mean pairwise CLIP score between the two sets of images.
    """

    # Get the model's visual features (without text features)
    sample_image = process_images_clip(glob.glob(sample_dir))
    ref_image = process_images_clip(glob.glob(reference_dir))

    with torch.no_grad():
        features1 = clip_model.encode_image(sample_image)
        features2 = clip_model.encode_image(ref_image)

    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)
    similarity = (features1 @ features2.T).cpu().numpy()
    similarity = np.max(similarity, axis=1)

    return similarity


def pixel_distance(sample_dir, reference_dir):
    """
    Function that calculate CLIP score, cosine similarity between images1 and images2

    Args:
    ----
        sample_dir: directory of the first set of images.
        reference_dir: directory of the second set of images.

    Return:
    ------
        Mean pairwise CLIP score between the two sets of images.
    """
    sample_images = process_images_np(glob.glob(sample_dir))
    ref_images = process_images_np(glob.glob(reference_dir))

    sample_images = sample_images.reshape(sample_images.shape[0], -1)
    ref_images = ref_images.reshape(ref_images.shape[0], -1)

    scores = np.zeros(sample_images.shape[0])

    for i, sample in enumerate(sample_images):
        distances = np.sqrt(np.sum((ref_images - sample) ** 2, axis=1))
        scores[i] = np.max(distances)
