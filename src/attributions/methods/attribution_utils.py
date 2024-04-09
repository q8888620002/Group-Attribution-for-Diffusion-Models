"""Utility functions for data attribution calculation."""
import glob
import json
import os

import clip
import numpy as np
import torch
from PIL import Image

from src.datasets import (
    create_dataset,
    remove_data_by_datamodel,
    remove_data_by_shapley
)


class CLIPScore:
    """Class for initializing CLIP model and calculating clip score."""

    def __init__(self, device):
        self.device = device
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=device)

    def process_images_clip(self, file_list):
        """Function to load and process images with clip transform"""
        images = []
        for filename in file_list:
            image = Image.open(filename)
            image = self.clip_transform(image).unsqueeze(0).to(self.device)
            images.append(image)
        return torch.cat(images, dim=0)

    def clip_score(self, sample_dir, reference_dir):
        """
        Function that calculate CLIP score between images1 and images2

        Args:
        ----
            sample_dir: directory of the first set of images.
            reference_dir: directory of the second set of images.

        Return:
        ------
            Mean pairwise CLIP score between the two sets of images.
        """

        # Get the model's visual features (without text features)
        sample_image = self.process_images_clip(glob.glob(sample_dir))
        ref_image = self.process_images_clip(glob.glob(reference_dir))

        with torch.no_grad():
            features1 = self.clip_model.encode_image(sample_image)
            features2 = self.clip_model.encode_image(ref_image)

        features1 = features1 / features1.norm(dim=-1, keepdim=True)
        features2 = features2 / features2.norm(dim=-1, keepdim=True)
        similarity = (features1 @ features2.T).cpu().numpy()
        similarity = np.mean(similarity, axis=1)

        return similarity


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

def process_images_np(file_list):
    """Function to load and process images into numpy"""
    images = []
    for filename in file_list:
        image = Image.open(filename).convert("RGB")
        image = np.array(image).astype(np.float32)
        images.append(image)
    return np.stack(images)


def pixel_distance(sample_dir, reference_dir):
    """
    Function that calculate minimum pixel distance between two image sets.

    Args:
    ----
        sample_dir: directory of the first set of images.
        reference_dir: directory of the second set of images.

    Return:
    ------
        Minimum pixel distance to images in reference dir for each image in sample dir.
    """
    sample_images = process_images_np(glob.glob(sample_dir))
    ref_images = process_images_np(glob.glob(reference_dir))

    sample_images = sample_images.reshape(sample_images.shape[0], -1)
    ref_images = ref_images.reshape(ref_images.shape[0], -1)

    similarity = np.zeros(sample_images.shape[0])

    for i, sample in enumerate(sample_images):
        distance = np.sqrt(np.sum((ref_images - sample) ** 2, axis=1))
        similarity[i] = np.mean(distance)

    return similarity
