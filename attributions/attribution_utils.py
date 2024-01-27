"""Utility functions for data attribution calculation."""
import glob
import json
import os

import clip
import numpy as np
import torch
from PIL import Image

from .utils import remove_data_by_datamodel, remove_data_by_shapley


class CLIPScore:
    """
    Class for initializing CLIP model and calculating clip score.
    """
  def __init__(self, device):

    self.clip_model, self.clip_transform = clip.load("ViT-B/32", device=device)

    def process_images_clip(self, file_list):
        """Function to load and process images with clip transform"""
        images = []
        for filename in file_list:
            image = Image.open(filename)
            image = clip_transform(image).unsqueeze(0).to(device)
            images.append(image)
        return torch.cat(images, dim=0)


    def clip_score(self, sample_dir, reference_dir):
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

def create_removal_path(args, seed_index):
    """Create removal directory based on removal distribution and subset index."""
    if args.removal_dist == "datamodel":
        removal_dir = (
            f"{args.removal_dist}/"
            f"{args.removal_dist}_"
            f"alpha={args.datamodel_alpha}_seed={seed_index}"
        )
    elif args.removal_dist == "shapley":
        removal_dir = f"{args.removal_dist}/{args.removal_dist}_seed={seed_index}"
    else:
        raise NotImplementedError(f"{args.removal_dist} does not exist.")

    return removal_dir

def load_model_behavior(args, seed_index):
    """Load model behavior based on the removal distribution and subset index."""

    removal_dir = create_removal_path(args, seed_index)

    model_behavior_path = os.path.join(
        args.outdir, args.dataset, args.method, removal_dir
    )

    with open(os.path.join(model_behavior_path, "model_behavior.json")) as f:
        model_behavior = json.load(f)

    return model_behavior


def load_gradient_data(args, subset_index):
    """Load gradient data based on the removal distribution and subset index."""

    removal_dir = create_removal_path(args, subset_index)

    grad_result_dir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        args.attribution_method,
        removal_dir,
        f"f={args.model_behavior}_t={args.t_strategy}",
    )

    return np.memmap(
        grad_result_dir,
        dtype=np.float32,
        mode="r",
        shape=(len(remaining_idx), args.projector_dim),
    )

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
        similarity[i] = np.min(distance)

    return similarity
