"""Utility functions for data attribution calculation."""
import glob
import json

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import (
    ImageDataset,
    create_dataset,
    remove_data_by_datamodel,
    remove_data_by_shapley,
)


class CLIPScore:
    """Class for initializing CLIP model and calculating clip score."""

    def __init__(self, device):
        self.device = device
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=device)

    def clip_score(self, dataset_name, sample_size, sample_dir, training_dir):
        """
        Function that calculate CLIP score between generated and training data

        Args:
        ----
            dataset_name: name of the dataset.
            sample_size: number of samples to calculate local model behavior
            sample_dir: directory of the first set of images.
            training_dir: directory of the second set of images.

        Return:
        ------
            Mean pairwise CLIP score as data attribution.
        """

        all_sample_features = []
        all_training_features = []
        num_workers = 4 if torch.get_num_threads() >= 4 else torch.get_num_threads()

        sample_dataset = ImageDataset(sample_dir, self.clip_transform, sample_size)
        sample_loader = DataLoader(
            sample_dataset, batch_size=64, num_workers=num_workers, pin_memory=True
        )
        train_dataset = ImageDataset(training_dir, transform=self.clip_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=64, num_workers=num_workers, pin_memory=True
        )

        # Assuming clip_transform is your image transformation pipeline

        with torch.no_grad():
            print(f"Calculating CLIP embeddings for {sample_dir}...")
            for sample_batch in tqdm(sample_loader):
                features = self.clip_model.encode_image(sample_batch.to(self.device))
                all_sample_features.append(features.cpu().numpy())

            print(f"Calculating CLIP embeddings for {training_dir}...")
            for training_batch in tqdm(train_loader):
                features = self.clip_model.encode_image(training_batch.to(self.device))
                all_training_features.append(features.cpu().numpy())

        # Concatenate all batch features
        all_sample_features = np.concatenate(all_sample_features, axis=0)
        all_training_features = np.concatenate(all_training_features, axis=0)

        all_sample_features = all_sample_features / np.linalg.norm(
            all_sample_features, axis=1, keepdims=True
        )
        all_training_features = all_training_features / np.linalg.norm(
            all_training_features, axis=1, keepdims=True
        )

        similarity = all_sample_features @ all_training_features.T
        coeff = np.mean(similarity, axis=0)

        if dataset_name in ["cifar100", "cifar100_f"]:
            dataset = create_dataset(dataset_name=dataset_name, train=True)
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


def mean_scores_by_class(scores, dataset):
    """
    Compute mean scores by classes and return group-based means.

    :param scores: sample-based coefficients
    :param dataset:
        dataset, each entry should be a tuple or list with the label as the last element
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

        # Convert PIL Image to NumPy array and scale from 0 to 1
        image_np = np.array(image, dtype=np.float32) / 255.0

        # Normalize: shift and scale the image to have pixel values in range [-1, 1]
        image_np = (image_np - 0.5) / 0.5

        images.append(image_np)

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
    generated_images = generated_images / np.linalg.norm(
        generated_images, axis=1, keepdims=True
    )
    ref_images = ref_images / np.linalg.norm(ref_images, axis=1, keepdims=True)

    similarities = np.dot(generated_images, ref_images.T)
    coeff = np.mean(similarities, axis=0)

    if args.by_class:
        dataset = create_dataset(dataset_name=args.dataset, train=True)
        coeff = mean_scores_by_class(coeff, dataset)

    return coeff
