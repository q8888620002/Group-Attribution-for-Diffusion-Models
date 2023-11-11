import torch
import torch.nn.functional as F
import numpy as np
import random
import torchvision.transforms.functional as TF

from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST , CIFAR10
from torchvision import transforms
from torchvision.transforms import Compose, Resize, Lambda, Normalize, ToPILImage
from torchvision.models import inception_v3

from scipy.linalg import sqrtm
from CLIP.clip import clip
from PIL import Image

class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)

def create_dataloaders(
    dataset_name: str,
    batch_size: int,
    image_size: int = 28,
    num_workers: int = 4,
    excluded_class: int = None,
    unlearning: bool = False
):
    """
    Create dataloaders for CIFAR10 and MNIST datasets with options for excluding 
    specific classes and creating subsets for unlearning.

    Args:
        dataset_name (str): Name of the dataset ('cifar' or 'mnist').
        batch_size (int): Batch size for the dataloaders.
        image_size (int): Image size for resizing (used for MNIST).
        num_workers (int): Number of workers for dataloaders.
        excluded_class (int, optional): Class to be excluded (for ablation).
        unlearning (bool, optional): Flag to create subsets for unlearning.

    Returns:
        Tuple[DataLoader, DataLoader]: Train and test dataloaders.
    """

    if dataset_name == 'cifar':
        preprocess = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        DatasetClass = CIFAR10
        root_dir = 'data2/mingyulu/cfair_10'
    
    elif dataset_name == 'mnist':
        preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        DatasetClass = MNIST
        root_dir = 'data2/mingyulu/mnist_data'
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, choose 'cifar' or 'mnist'.")

    train_dataset = DatasetClass(root=root_dir, train=True, download=True, transform=preprocess)
    test_dataset = DatasetClass(root=root_dir, train=False, download=True, transform=preprocess)

    # Exclude specified class if needed
    if not unlearning and excluded_class is not None:
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label != excluded_class]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label != excluded_class]

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Handle unlearning subsets if needed
    if unlearning and excluded_class is not None:
        ablated_indices = [i for i, (_, label) in enumerate(train_dataset) if label == excluded_class]
        remaining_indices = [i for i, (_, label) in enumerate(train_dataset) if label != excluded_class]
        
        remaining_indices = random.sample(remaining_indices, len(ablated_indices))

        remaining_dataset = Subset(train_dataset, remaining_indices)
        ablated_dataset = Subset(train_dataset, ablated_indices)

        # Return dataloaders for unlearning
        return (
            DataLoader(remaining_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(ablated_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        )

    # Regular dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader


def calculate_fid(images1, images2, device):
    """

    Create a mnist dataset with/without filtered labels

    Args:
        images1:
        images2:
    Return:
        fid score between images1 and images2
    """

    model = inception_v3(
        weights='DEFAULT',
        transform_input=False,
        aux_logits=True
    ).to(device)

    model.eval()

    ## resize images for inception_v3 as it expects tensors with a size of (N x 3 x 299 x 299).

    images1 = torch.stack([TF.resize(img, (299, 299), antialias=True) for img in images1])
    images2 = torch.stack([TF.resize(img, (299, 299), antialias=True) for img in images2])

    with torch.no_grad():
        act1 = model(images1)
        act2 = model(images2)

    act1 = act1.cpu().numpy()
    act2 = act2.cpu().numpy()

    mu1, sigma1 = act1.mean(axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), np.cov(act2, rowvar=False)

    ss_diff = np.sum((mu1 - mu2)**2.0)

    cov_mean = sqrtm(sigma1.dot(sigma2))

    # Check and correct imaginary numbers from sqrt
    if np.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Calculate score
    fid = ss_diff + np.trace(sigma1 + sigma2 - 2.0 * cov_mean)

    return fid

# Load CLIP model and transformation outside of the function for efficiency
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, clip_transform = clip.load("ViT-B/32", device=device)

def preprocess_clip_mnist(batch_images):
    """
    Preprocess a batch of MNIST images for CLIP.

    """

    transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    batch_processed = torch.stack([transform(img) for img in batch_images])

    return batch_processed

def clip_score(images1, images2):

    """
    Function that calculate CLIP score, cosine similarity between images1 and images2

    """

    images1 = preprocess_clip_mnist(images1).to(device)
    images2 = preprocess_clip_mnist(images2).to(device)

    # Get the model's visual features (without text features)

    with torch.no_grad():
        features1 = clip_model.encode_image(images1)
        features2 = clip_model.encode_image(images2)

    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)
    similarity = (features1 @ features2.T).cpu().numpy()

    return similarity.mean()