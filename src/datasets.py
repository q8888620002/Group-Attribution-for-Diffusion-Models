"""Dataset related functions and classes"""

import os
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, MNIST, ImageFolder

import src.constants as constants


class CIFAR2(CIFAR10):
    """
    Dataloader for CIFAR2 dataset (automobile and horse)

    Return_
        3x32x32 CIFAR-2 images, and it's corresponding label
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):

        super(CIFAR2, self).__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.classes_to_keep = [1, 7]  # 1 for automobile, 7 for horse

        # Filter the dataset
        filtered_indices = [
            i for i, target in enumerate(self.targets) if target in self.classes_to_keep
        ]
        self.data = self.data[filtered_indices]
        self.targets = [
            self.classes_to_keep.index(target)
            for i, target in enumerate(self.targets)
            if target in self.classes_to_keep
        ]

        # Update class labels and class names
        self.classes = ["automobile", "horse"]
        self.class_to_idx = {"automobile": 0, "horse": 1}


class CIFAR100(CIFAR100):
    """
    Dataloader for CIFAR-100 dataset to include only animal classes.

    Return_
        3x32x32 CIFAR-100 images, and its corresponding label
        (filtered to only animals, 35 classes.)
    """

    def __init__(
        self, root, train=True, transform=None, target_transform=None, download=False
    ):
        super().__init__(
            root,
            train=train,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

        # Update this list based on CIFAR-100 animal class indices
        self.classes_to_keep = [
            # 0, 1, 2, 3, 4,   # Aquatic mammals
            # 5, 6, 7, 8, 9,   # Fish
            # 75, 76, 77, 78, 79,  # Reptiles
            40,
            41,
            42,
            43,
            44,  # Large carnivores
            55,
            56,
            57,
            58,
            59,  # Large omnivores and herbivores
            60,
            61,
            62,
            63,
            64,  # Medium mammals
            80,
            81,
            82,
            83,
            84,  # Small mammals
        ]
        # Filter the dataset

        filtered_indices = [
            i for i, target in enumerate(self.targets) if target in self.classes_to_keep
        ]
        self.data = self.data[filtered_indices]
        self.targets = [
            self.classes_to_keep.index(target)
            for i, target in enumerate(self.targets)
            if target in self.classes_to_keep
        ]


class CelebA(Dataset):
    """
    DataLoader for CelebA 256 x 256. Note that there's no label for this one.

    Return_
        3x256x256 Celeb images, and -1, pseudo-label
    """

    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.train = train

        data_df = pd.read_csv(os.path.join(root, "labels.csv"))

        assert data_df["filename"].nunique() == len(
            data_df
        ), "filename should be unique"
        # self.data_df = data_df[data_df["split"] == ("train" if train else "test")]
        self.data_df = data_df

    def __len__(self):
        """Return the number of dataset"""
        return len(self.data_df)

    def __getitem__(self, idx):
        """Iterate dataloader"""
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.root, row["filename"])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # return image, -1, row["filename"]
        return image, row["celeb"], row["filename"]


class ImageDataset(Dataset):
    """Loads and transforms images from a directory."""

    def __init__(self, img_dir, transform=transforms.PILToTensor()):
        """Initializes dataset with image directory and transform."""
        self.img_dir = img_dir
        self.img_list = [
            img
            for img in os.listdir(img_dir)
            if img.split(".")[-1] in {"jpg", "jpeg", "png", "bmp", "webp", "tiff"}
        ]
        self.transform = transform

    def __getitem__(self, idx):
        """Returns transformed image at index `idx`."""
        with Image.open(os.path.join(self.img_dir, self.img_list[idx])) as im:
            return self.transform(im)

    def __len__(self):
        """Returns total number of images."""
        return len(self.img_list)


class TensorDataset(Dataset):
    """Wraps tensor data for easy dataset operations."""

    def __init__(self, data):
        """Initializes dataset with data tensor."""
        self.data = data

    def __len__(self):
        """Returns dataset size."""
        return len(self.data)

    def __getitem__(self, idx):
        """Retrieves sample at index `idx`."""
        return self.data[idx]


def create_dataset(
    dataset_name: str,
    train: bool,
    dataset_dir: str = constants.DATASET_DIR,
) -> torch.utils.data.Dataset:
    """
    Create a PyTorch Dataset corresponding to a dataset.

    Args:
    ----
        dataset_name: Name of the dataset.
        train: Whether to return the training dataset or the test set.
        dataset_dir: Parent directory for all the datasets.

    Return:
    ------
        A PyTorch Dataset.
    """
    if dataset_name == "cifar":
        preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "cifar")
        dataset = CIFAR10(
            root=root_dir, train=train, download=True, transform=preprocess
        )
    elif dataset_name == "cifar2":
        preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "cifar2")
        dataset = CIFAR2(
            root=root_dir, train=train, download=True, transform=preprocess
        )
    elif dataset_name == "cifar100":
        preprocess = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "cifar100")
        dataset = CIFAR100(
            root=root_dir, train=train, download=True, transform=preprocess
        )
    elif dataset_name == "mnist":
        preprocess = transforms.Compose(
            [
                transforms.Resize((32, 32)),  # Resize to 32x32 for diffusers UNet.
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "mnist")
        dataset = MNIST(root=root_dir, train=train, download=True, transform=preprocess)
    elif dataset_name == "celeba":
        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "celeba_hq_256_50_resized")
        dataset = CelebA(root=root_dir, train=train, transform=preprocess)
    elif dataset_name == "imagenette":
        preprocess = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1,1].
            ]
        )
        root_dir = os.path.join(dataset_dir, "imagenette2", "train" if train else "val")
        dataset = ImageFolder(root_dir, transform=preprocess)
    else:
        raise ValueError(
            f"dataset_name={dataset_name} should be one of ['cifar', 'mnist', 'celeba']"
        )
    return dataset


def remove_data_by_class(
    dataset: torch.utils.data.Dataset, excluded_class: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where
    data corresponding to a class are removed.

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        excluded_class: The class to remove.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    removed_idx = [i for i, (_, label) in enumerate(dataset) if label == excluded_class]
    removed_idx = np.array(removed_idx)
    remaining_idx = np.setdiff1d(np.arange(len(dataset)), removed_idx)
    return remaining_idx, removed_idx


def remove_data_by_uniform(
    dataset: torch.utils.data.Dataset, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where each
    data point has a 0.5 probability of being removed.

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        seed: Random seed for sampling which data points are selected to keep.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    rng = np.random.RandomState(seed)
    selected = rng.normal(size=len(dataset)) > 0
    all_idx = np.arange(len(dataset))
    return all_idx[selected], all_idx[~selected]


def remove_data_by_datamodel(
    dataset: torch.utils.data.Dataset, alpha: float = 0.5, seed: int = 0, by_class: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where
    the remaining dataset is an `alpha` proportion of the full dataset.

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        alpha: The proportion of the full dataset to keep in the remaining set.
        seed: Random seed for sampling which data points are selected to keep.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    rng = np.random.RandomState(seed)
    if by_class:
        # If splitting by class, we need to sample the class to remove.
        possible_classes = np.unique([data[1] for data in dataset])
        all_idx = np.arange(len(possible_classes))
        
        num_selected= int(alpha * len(possible_classes))
        rng.shuffle(all_idx)  # Shuffle in place.
        
        remaining_classes = possible_classes[all_idx[:num_selected]]
        remaining_idx = [
            i for i, data in enumerate(dataset) if data[1] in remaining_classes
        ]
        remaining_idx = np.array(remaining_idx)
        removed_idx = np.setdiff1d(np.arange(len(dataset)), remaining_idx)
        return remaining_idx, removed_idx
    else:
        dataset_size = len(dataset)
        all_idx = np.arange(dataset_size)

        num_selected = int(alpha * dataset_size)
        rng.shuffle(all_idx)  # Shuffle in place.

        remaining_idx = all_idx[:num_selected]
        removed_idx = all_idx[num_selected:]
        return remaining_idx, removed_idx


def remove_data_by_shapley(
    dataset: torch.utils.data.Dataset, seed: int = 0, by_class: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a PyTorch Dataset into indices with the remaining and removed data, where
    the remaining dataset is drawn from the Shapley kernel distribution, which has the
    probability mass function: p(S) = (n - 1) / (|S| * (n - |S|) * (n choose |S|)).

    Reference: https://captum.ai/api/kernel_shap.html#captum.attr.KernelShap.
    kernel_shap_perturb_generator.

    Args:
    ----
        dataset: The PyTorch Dataset to split.
        seed: Random seed for sampling which data points are selected to keep.

    Returns
    -------
        A numpy array with the remaining indices, and another numpy array with the
        indices corresponding to the removed data.
    """
    rng = np.random.RandomState(seed)
    if by_class:
        # If splitting by class, we need to sample the class to remove.
        possible_classes = np.unique([data[1] for data in dataset])
        possible_classes_sizes = np.arange(1, len(possible_classes))
        remaining_size_probs = (len(possible_classes) - 1) / (
            possible_classes_sizes * (len(possible_classes) - possible_classes_sizes)
        )
        remaining_size_probs /= remaining_size_probs.sum()
        remaining_size = rng.choice(
            possible_classes_sizes, size=1, p=remaining_size_probs
        )[0]

        all_idx = np.arange(len(possible_classes))
        rng.shuffle(all_idx)  # Shuffle in place.
        removed_classes = possible_classes[all_idx[remaining_size:]]

        removed_idx = [
            i for i, data in enumerate(dataset) if data[1] in removed_classes
        ]
        removed_idx = np.array(removed_idx)
        remaining_idx = np.setdiff1d(np.arange(len(dataset)), removed_idx)
        return remaining_idx, removed_idx
    else:
        dataset_size = len(dataset)

        # First sample the remaining set size.
        # This corresponds to the term: (n - 1) / (|S| * (n - |S|)).
        possible_remaining_sizes = np.arange(1, dataset_size)
        remaining_size_probs = (dataset_size - 1) / (
            possible_remaining_sizes * (dataset_size - possible_remaining_sizes)
        )
        remaining_size_probs /= remaining_size_probs.sum()
        remaining_size = rng.choice(
            possible_remaining_sizes, size=1, p=remaining_size_probs
        )[0]

        # Then sample uniformly given the remaining set size.
        # This corresponds to the term: 1 / (n choose |S|).
        all_idx = np.arange(dataset_size)
        rng.shuffle(all_idx)  # Shuffle in place.
        remaining_idx = all_idx[:remaining_size]
        removed_idx = all_idx[remaining_size:]
        return remaining_idx, removed_idx


def removed_by_classes(dataset: torch.utils.data.Dataset, seed: int = 0):
    """Function that return remained and removed classes."""
    # Find all possible classes in the dataset

    rng = np.random.RandomState(seed)

    # If splitting by class, we need to sample the class to remove.
    possible_classes = np.unique([data[1] for data in dataset])
    possible_classes_sizes = np.arange(1, len(possible_classes))
    remaining_size_probs = (len(possible_classes) - 1) / (
        possible_classes_sizes * (len(possible_classes) - possible_classes_sizes)
    )
    remaining_size_probs /= remaining_size_probs.sum()
    remaining_size = rng.choice(possible_classes_sizes, size=1, p=remaining_size_probs)[
        0
    ]

    all_idx = np.arange(len(possible_classes))
    rng.shuffle(all_idx)  # Shuffle in place.
    removed_classes = possible_classes[all_idx[remaining_size:]]
    remaining_classes = possible_classes[all_idx[:remaining_size]]

    return remaining_classes, removed_classes
