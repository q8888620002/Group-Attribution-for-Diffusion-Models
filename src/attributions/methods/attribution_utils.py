"""Utility functions for data attribution calculation."""
import glob

import clip
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets import ImageDataset, create_dataset



def aggregate_by_class(scores, dataset, by="mean"):
    """
    Compute mean scores by classes and return group-based means.

    :param scores: sample-based coefficients
    :param dataset:
        dataset, each entry should be a tuple or list with the label as the last element
    :return: Numpy array with mean scores, indexed by label.
    """
    if scores.ndim == 1:
        scores = scores.reshape(1, -1)

    n, _ = scores.shape


    unique_values = sorted(set(data[1] for data in dataset))
    value_to_number = {value: i  for i, value in enumerate(unique_values)}
    labels = np.array([value_to_number[entry[1]] for entry in dataset])
    num_labels = len(np.unique(labels))

    result = np.zeros((n, num_labels))

    for i in range(num_labels):
        # Create a mask for columns corresponding to the current label
        label_mask = labels == i
        # Average scores across groups
        if by == "mean":
            result[:, i] = np.divide(
                scores[:, label_mask].sum(axis=1), np.sum(label_mask)
            )
        elif by == "max":
            result[:, i] = np.max(scores[:, label_mask])

    return result


def process_images_np(file_list, max_size=None):
    """Function to load and process images into numpy"""

    valid_extensions = {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
    images = []
    filtered_files = [
        file for file in file_list if file.split(".")[-1].lower() in valid_extensions
    ]

    if max_size is not None:
        filtered_files = filtered_files[:max_size]

    for filename in tqdm(filtered_files):
        try:
            image = Image.open(filename).convert("RGB")
            image = np.array(image).astype(np.float32)

            # Convert PIL Image to NumPy array and scale from 0 to 1
            image_np = np.array(image, dtype=np.float32) / 255.0

            # Normalize: shift and scale the image to have pixel values in range [-1, 1]
            image_np = (image_np - 0.5) / 0.5

            images.append(image_np)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    return np.stack(images) if images else np.array([])

class CLIPScore:
    """Class for initializing CLIP model and calculating clip score."""

    def __init__(self, device):
        self.device = device
        self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=device)

    def clip_score(
        self,
        model_behavior_key,
        by,
        dataset_name,
        sample_size,
        sample_dir,
        training_dir,
    ):
        """
        Function that calculate CLIP score between generated and training data

        Args:
        ----
            model_behavior_key: model behavior
            by: aggregate class based coefficients based on mean or max
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
            for sample_batch, _ in tqdm(sample_loader):
                features = self.clip_model.encode_image(sample_batch.to(self.device))
                all_sample_features.append(features.cpu().numpy())

            print(f"Calculating CLIP embeddings for {training_dir}...")
            for training_batch, _ in tqdm(train_loader):
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

        if model_behavior_key not in ["ssim", "nrmse", "diffusion_loss"]:
            # average across all samples

            coeff = np.mean(similarity, axis=0)
        else:
            coeff = similarity

        if dataset_name in ["cifar100", "cifar100_f", "celeba", "cifar100_new"]:
            dataset = create_dataset(dataset_name=dataset_name, train=True)
            coeff = aggregate_by_class(coeff, dataset, by)

        return coeff


def pixel_distance(
    model_behavior_key, by, dataset_name, sample_size, generated_dir, training_dir
):
    """
    Function that calculate the pixel distance between two image sets,
    generated images and training images. Using the average distance
    across generated images as attribution value for training data.

    Args:
    ----
        model_behavior_key: model behavior
        by: aggregated class based coefficients based on mean or max
        dataset_name: dataset
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

    if model_behavior_key not in ["ssim", "nrmse", "diffusion_loss"]:
        coeff = np.mean(similarities, axis=0)
    else:
        coeff = similarities

    if dataset_name in ["cifar100", "cifar100_f", "celeba","cifar100_new"]:
        dataset = create_dataset(dataset_name=dataset_name, train=True)
        # coeff = mean_scores_by_class(coeff, dataset)

        coeff = aggregate_by_class(coeff, dataset, by)

    return coeff
