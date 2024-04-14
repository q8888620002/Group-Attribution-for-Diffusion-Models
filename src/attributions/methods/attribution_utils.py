"""Utility functions for data attribution calculation."""
import glob
import json

import clip
import numpy as np
import torch
from PIL import Image
from scipy.spatial.distance import cdist
from tqdm import tqdm

from src.datasets import (
    create_dataset,
    remove_data_by_datamodel,
    remove_data_by_shapley,
)


class CLIPScore:
    """Class for initializing CLIP model and calculating clip score."""

    def __init__(self, device):
        self.device = device
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=device)

    def process_images_clip(self, file_list, max_size=None):
        """Function to load and process images with clip transform"""

        images = []
        if max_size is not None:
            file_list = file_list[:max_size]

        for filename in tqdm(file_list):
            image = Image.open(filename)
            image = self.clip_transform(image).unsqueeze(0).to(self.device)
            images.append(image)
        return torch.cat(images, dim=0)

    def clip_score(self, args, sample_size, sample_dir, training_dir):
        """
        Function that calculate CLIP score between generated and training data

        Args:
        ----
            args: argument for calculating lds.
            sample_size: number of samples
            sample_dir: directory of the first set of images.
            training_dir: directory of the second set of images.

        Return:
        ------
            Mean pairwise CLIP score as data attribution.
        """

        # Get the model's visual features (without text features)

        sample_image = self.process_images_clip(
            glob.glob(sample_dir + "/*"), sample_size
        )
        training_image = self.process_images_clip(glob.glob(training_dir + "/*"))

        with torch.no_grad():
            features1 = self.clip_model.encode_image(sample_image)
            features2 = self.clip_model.encode_image(training_image)

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        similarity = (features1 @ features2.T).cpu().numpy()
        coeff = np.mean(similarity, axis=0).astype(np.float32)

        if args.by_class:
            dataset = create_dataset(dataset_name=args.dataset, train=True)
            coeff = mean_scores_by_class(coeff, dataset)

        return coeff


def load_filtered_behaviors(file_path, exp_name):
    """Define function to load and filter model behaviors based on experiment name"""

    filtered_behaviors = []
    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            if row.get("exp_name") == exp_name:
                filtered_behaviors.append(row)
    return filtered_behaviors


def create_removal_path(args, seed_index):
    """Create removal directory based on removal distribution and subset index."""

    full_dataset = create_dataset(dataset_name=args.dataset, train=True)

    if args.removal_dist == "datamodel":
        removal_dir = (
            f"{args.removal_dist}/"
            f"{args.removal_dist}_"
            f"alpha={args.datamodel_alpha}_seed={seed_index}"
        )
        remaining_idx, _ = remove_data_by_datamodel(
            full_dataset, alpha=args.datamodel_alpha, seed=seed_index
        )
    elif args.removal_dist == "shapley":
        removal_dir = f"{args.removal_dist}/{args.removal_dist}_seed={seed_index}"
        remaining_idx, _ = remove_data_by_shapley(full_dataset, seed=seed_index)
    else:
        raise NotImplementedError(f"{args.removal_dist} does not exist.")

    return removal_dir, remaining_idx

    raise ValueError(f"No record found for sample_dir: {removal_dir}")


def mean_scores_by_class(scores, dataset):
    """
    Compute mean scores by classes and return group-based means.

    :param scores: sample-based coefficients
    :param dataset: dataset, each entry should be a tuple or list with the label as the last element
    :return: Numpy array with mean scores, indexed by label.
    """
    # Initialize dictionaries to accumulate scores and counts for each label
    scores_and_counts = {}

    # Gather sums and counts per label
    for score, (_, label) in zip(scores, dataset):
        if label in scores_and_counts:
            scores_and_counts[label][0] += score
            scores_and_counts[label][1] += 1
        else:
            scores_and_counts[label] = [score, 1]  # [sum, count]

    # Prepare the result array
    num_labels = max(scores_and_counts.keys()) + 1
    result_array = np.zeros(num_labels)

    # Compute the mean for each label
    for label, (total_score, count) in scores_and_counts.items():
        result_array[label] = total_score / count

    return result_array


def process_images_np(file_list, max_size=None):
    """Function to load and process images into numpy"""
    images = []

    if max_size is not None:
        file_list = file_list[:max_size]

    for filename in tqdm(file_list):
        image = Image.open(filename).convert("RGB")
        image = np.array(image).astype(np.float32)
        images.append(image)
    return np.stack(images)


def pixel_distance(args, sample_size, generated_dir, training_dir):
    """
    Function that calculate the pixel distance between two image sets,
    generated images and training images. Using the average distance
    across generated images as attribution value for training data.

    Args:
    ----
        args: argument for calculating lds.
        sample_size: number of generated samples.
        generated_dir: directory of the generated images.
        training_dir: directory of the training set images.

    Return:
    ------
        Mean of pixel distance as data attribution.

    """
    print(f"Loading images from {generated_dir}..")

    generated_images = process_images_np(glob.glob(generated_dir + "/*"), sample_size)

    print(f"Loading images from {training_dir}..")

    ref_images = process_images_np(glob.glob(training_dir + "/*"))

    generated_images = generated_images.reshape(generated_images.shape[0], -1)
    ref_images = ref_images.reshape(ref_images.shape[0], -1)
    # Normalize the image vectors to unit vectors
    generated_images = generated_images / np.linalg.norm(generated_images, axis=1, keepdims=True)
    ref_images = ref_images / np.linalg.norm(ref_images, axis=1, keepdims=True)

    similarities = np.dot(generated_images, ref_images.T)
    coeff = np.mean(similarities, axis=0)

    if args.by_class:
        dataset = create_dataset(dataset_name=args.dataset, train=True)
        coeff = mean_scores_by_class(coeff, dataset)
    
    return coeff
