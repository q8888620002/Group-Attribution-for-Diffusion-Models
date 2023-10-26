import torch
import torch.nn.functional as F
import numpy as np
import random

from torch.utils.data import DataLoader, Subset

from torchvision.datasets import MNIST
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.models import inception_v3
from torchvision.transforms import Compose, Resize, Lambda, Normalize, ToPILImage
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


def create_mnist_dataloaders(
        batch_size,image_size=28,
        num_workers=4,
        exclude_label=None
    ):

    """

    Create a mnist dataset with/without filtered labels

    Args:
        batch_size: int
        num_worksrs: int
        exclude_label: int e.g. 1,2
    Return:
        MNIST dataset.

    """

    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess
        )
    test_dataset=MNIST(
        root="./mnist_data",
        train=False,
        download=True,
        transform=preprocess
        )

    if exclude_label is not None:
        train_indices = [i for i, (_, label) in enumerate(train_dataset) if label != exclude_label]
        test_indices = [i for i, (_, label) in enumerate(test_dataset) if label != exclude_label]

        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)


def create_unlearning_dataloaders(
        batch_size,image_size=28,
        num_workers=4,
        keep_digits: bool=True,
        exclude_label=None
    ):

    """

    Create a mnist dataset with/without filtered labels

    Args:
        batch_size: int
        num_worksrs: int
        exclude_label: int e.g. 1,2
    Return:
        ablated_subset: Dataloader that ONLY contains excluded digit.
        remaining_subset: Dataloader that contains digits excluded digit.

        ablated_subset: Dataloader that contains excluded digit. (The original mnist datasets.)
        remaining_subset: Dataloader that contains digits excluded digit.
    """

    preprocess=transforms.Compose(
        [transforms.Resize(image_size),\
        transforms.ToTensor(),\
        transforms.Normalize([0.5],[0.5])
    ]) #[0,1] to [-1,1]

    train_dataset=MNIST(
        root="./mnist_data",
        train=True,
        download=True,
        transform=preprocess
    )

    all_indices = [i for i, (_, label) in enumerate(train_dataset)]

    if exclude_label is not None:

        exclude_indices = [i for i, (_, label) in enumerate(train_dataset) if label == exclude_label]
        remaining_indices = [i for i in all_indices if i not in exclude_indices]

        if keep_digits:
            ablated_indices = random.sample(all_indices, len(remaining_indices))
        else:
            ablated_indices = exclude_indices
            # Resize remaining_indices to match the length of ablated_indices
            remaining_indices = random.sample(remaining_indices, len(ablated_indices))

        ablated_subset = Subset(train_dataset, ablated_indices)
        remaining_subset = Subset(train_dataset, remaining_indices)



    return (DataLoader(remaining_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                DataLoader(ablated_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers))




def calculate_fid_mnist(images1, images2):
    """

    Create a mnist dataset with/without filtered labels

    Args:
        images1:
        images2:
    Return:
        fid score between images1 and images2
    """

    # Define preprocessing steps for MNIST
    preprocess_mnist = Compose([
        Resize((299, 299)),
        Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to RGB format
        Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # Adjust normalization for MNIST
    ])

    model = inception_v3(pretrained=True, transform_input=False, aux_logits=True)

    model = model.cuda() if torch.cuda.is_available() else model
    model.eval()

    # Preprocess images
    images1 = torch.stack([preprocess_mnist(img) for img in images1])
    images2 = torch.stack([preprocess_mnist(img) for img in images2])

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
    """Preprocess a batch of MNIST images for CLIP."""

    transform = Compose([
        ToPILImage(),
        Resize((224, 224)),
        Lambda(lambda x: x.convert("RGB") if x.mode != "RGB" else x),
        transforms.ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    ])

    # Apply the transformations
    batch_processed = torch.stack([transform(img) for img in batch_images])

    return batch_processed

def clip_score(images1, images2):

    images1 = preprocess_clip_mnist(images1).to(device)
    images2 = preprocess_clip_mnist(images2).to(device)

    # Get the model's visual features (without text features)
    with torch.no_grad():
        features1 = clip_model.encode_image(images1)
        features2 = clip_model.encode_image(images2)

    features1 = features1 / features1.norm(dim=-1, keepdim=True)
    features2 = features2 / features2.norm(dim=-1, keepdim=True)
    similarity = (features1 @ features2.T).cpu().numpy()

    return similarity