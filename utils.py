"""Utilities"""

import glob
import os
import sys
from typing import List, Tuple

import numpy as np
import pynvml
import torch
from PIL import Image
from scipy.linalg import sqrtm
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, ImageFolder
from torchvision.transforms import Compose, Lambda, Normalize, Resize, ToPILImage
from transformers import PreTrainedTokenizer

import constants


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def get_memory_free_MiB(gpu_index):
    """Method for monitoring GPU usage when debugging."""
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(int(gpu_index))
    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return mem_info.free // 1024 ** 2


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


class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """
    Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel
    <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


class ImagenetteCaptioner:
    """
    Class to caption Imagenette labels.

    Args:
    ----
        dataset: a torchvision ImageFolder dataset.
    """

    def __init__(self, dataset: ImageFolder):
        self.label_to_synset = dataset.classes  # List of synsets.
        self.num_classes = len(self.label_to_synset)
        self.synset_to_word = {
            "n01440764": "tench",
            "n02102040": "English springer",
            "n02979186": "cassette player",
            "n03000684": "chainsaw",
            "n03028079": "church",
            "n03394916": "French horn",
            "n03417042": "garbage truck",
            "n03425413": "gas pump",
            "n03445777": "golf ball",
            "n03888257": "parachute",
        }

    def __call__(self, labels: torch.LongTensor) -> List[str]:
        """
        Convert integer labels to string captions.

        Args:
        ----
            labels: Tensor of integer labels.

        Returns
        -------
            A list of string captions, with the format of "a photo of a {object}."
        """
        captions = []
        for label in labels:
            synset = self.label_to_synset[label]
            word = self.synset_to_word[synset]
            captions.append(f"a photo of a {word}.")
        return captions


class LabelTokenizer:
    """
    Class to convert integer labels to caption token ids.

    Args:
    ----
        captioner: A class that converts integer labels to string captions.
        tokenizer: A Hugging Face PreTrainedTokenizer.
    """

    def __init__(self, captioner: ImagenetteCaptioner, tokenizer: PreTrainedTokenizer):
        self.captioner = captioner
        self.tokenizer = tokenizer

    def __call__(self, labels: torch.LongTensor) -> torch.LongTensor:
        """
        Converts integer labels to caption token ids.

        Args:
        ----
            labels: Tensor of integer labels.

        Returns
        -------
            Integer tensor of token ids, with padding and truncation if necessary.
        """
        captions = self.captioner(labels)
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids


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

        all_img_names = os.listdir(root)

        rng = np.random.RandomState(42)
        shuffled_indices = rng.permutation([i for i in range(len(all_img_names))])

        train_size = int(0.8 * len(all_img_names))

        if train:
            self.img_names = [all_img_names[i] for i in shuffled_indices[:train_size]]
        else:
            self.img_names = [all_img_names[i] for i in shuffled_indices[train_size:]]

    def __len__(self):
        """Return the number of dataset"""
        return len(self.img_names)

    def __getitem__(self, idx):
        """Iterate dataloader"""
        img_path = os.path.join(self.root, self.img_names[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image, -1


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
        root_dir = os.path.join(dataset_dir, "celeba/celeba_hq_256")
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
    dataset: torch.utils.data.Dataset, alpha: float = 0.5, seed: int = 0
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
    dataset_size = len(dataset)
    all_idx = np.arange(dataset_size)

    num_selected = int(alpha * dataset_size)
    rng.shuffle(all_idx)  # Shuffle in place.

    remaining_idx = all_idx[:num_selected]
    removed_idx = all_idx[num_selected:]
    return remaining_idx, removed_idx


def remove_data_by_shapley(
    dataset: torch.utils.data.Dataset, seed: int = 0
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


def get_max_step_file(folder_path):
    """Get maximum number of training steps for results in a folder."""
    path_pattern = os.path.join(folder_path, "steps_*.pt")
    files = glob.glob(path_pattern)
    if not files:
        return None
    max_step_file = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0])
    )
    return max_step_file


def get_max_steps(folder_path):
    """Get maximum number of training steps for results in a folder."""

    path_pattern = os.path.join(folder_path, "ckpt_steps_*.pt")
    files = glob.glob(path_pattern)

    if not files:
        return None

    max_steps = max(
        files, key=lambda x: int(os.path.basename(x).split("_")[-1].split(".")[0])
    )
    return int(os.path.basename(max_steps).split("_")[-1].split(".")[0])


def get_features(dataloader, mean, std, model, n_samples, device) -> np.ndarray:
    """
    Feature extraction for Inception V3

    Args:
    ----
        dataloader: Dataloader that contains real images e.g. cifar-10
        mean: mean for the preprocessing for a given dataset
        std: std for the preprocessing for a given dataset
        model: this should be an inception_v3 model with pretrained weights.
        n_samples: number of samples to be collected for real images from dataloader
        device: model device

    Return:
    ------
        features converted by inception_v3, should be (n_samples, 2024)
    """

    model.eval()
    features = []

    processed_samples = 0

    for images, _ in dataloader:
        if processed_samples >= n_samples:
            break

        with torch.no_grad():

            images = images.to(device)

            # Convert iamges to [ -1, 1] if use other normalization scales.

            images = (images * std + mean) * 2.0 - 1.0

            # Passing through inception

            batch_features = model(images)[0]
            batch_features = batch_features.squeeze(3).squeeze(2).cpu().numpy()

            features.append(batch_features)
            processed_samples += images.size(0)

    features = np.concatenate(features, axis=0)

    # Trim the features if necessary
    return features[:n_samples]


def calculate_fid(real_features, fake_features):
    """
    Calculating fid score, squared Wasserstein metric between two multidimensional
    Gaussian distributions.

    Args:
    ----
        real_features: Features of real images.
        fake_features: Features of fake generated images.

    Return:
    ------
        fid score between images1 and images2
    """

    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)

    cov_mean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Calculate score
    fid = np.sum((mu1 - mu2) ** 2.0) + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)

    return fid


def preprocess_clip_mnist(batch_images):
    """Preprocess a batch of MNIST images for CLIP."""

    transform = Compose(
        [
            ToPILImage(),
            Resize((224, 224)),
            Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
            transforms.ToTensor(),
            Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        ]
    )

    batch_processed = torch.stack([transform(img) for img in batch_images])

    return batch_processed
