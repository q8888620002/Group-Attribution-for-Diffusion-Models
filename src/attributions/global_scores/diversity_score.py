"""Converting images to BLIP embedding"""

import glob
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from lightning import seed_everything
from PIL import Image
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import squareform
from transformers import BlipForQuestionAnswering, BlipImageProcessor

from src.attributions.methods.attribution_utils import aggregate_by_class
from src.datasets import create_dataset


class ImageDataset(torch.utils.data.Dataset):
    """Image dataset class for generated images."""

    def __init__(self, image_dir_or_tensor, processor):
        if isinstance(image_dir_or_tensor, torch.Tensor):
            self.image_files_or_tensor = image_dir_or_tensor
        elif isinstance(image_dir_or_tensor, str):
            self.image_files_or_tensor = [
                file
                for file in glob.glob(image_dir_or_tensor + "/*")
                if file.endswith(".jpg")
                or file.endswith(".png")
                or file.endswith(".jpeg")
            ]
            self.image_files_or_tensor = sorted(
                self.image_files_or_tensor,
                key=lambda x: os.path.basename(x).split(".")[0],
            )
        else:
            raise ValueError("Image directory or tensor should be provided")
        self.processor = processor

    def __len__(self):
        """Return the length of dataset."""
        return len(self.image_files_or_tensor)

    def __getitem__(self, idx):
        """Iterate dataset"""
        if isinstance(self.image_files_or_tensor, torch.Tensor):
            # to comply with generate_images function output
            image = Image.fromarray(
                (255 * self.image_files_or_tensor[idx].permute(1, 2, 0).numpy()).astype(
                    np.uint8
                )
            )
        elif isinstance(self.image_files_or_tensor, list):
            file_path = self.image_files_or_tensor[idx]
            image = Image.open(file_path)

        else:
            raise ValueError("Image directory or tensor should be provided")
        image = image.convert("RGB")
        tensor = self.processor(images=image, return_tensors="pt")
        assert len(tensor["pixel_values"]) == 1, "Batch size should be 1"
        tensor["pixel_values"] = tensor["pixel_values"][0]

        return tensor


def image_files_to_tensor(image_files):
    """Convert image file list to tensor"""

    tensor_list = []
    for image in image_files:
        image = Image.open(image)
        image = image.convert("RGB")
        torch_image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
        tensor_list.append(torch_image)
    return torch.stack(tensor_list)


def calculate_diversity_score(
    ref_image_dir_or_tensor,
    generated_images_dir_or_tensor,
    num_cluster,
    use_cache=True,
):
    """Calcualte entropy based on BLIP-VQA embedding of reference images."""
    processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model = model.vision_model.to("cuda")
    model.eval()

    dataset1 = ImageDataset(ref_image_dir_or_tensor, processor)

    if use_cache:
        assert isinstance(
            ref_image_dir_or_tensor, str
        ), "Cache can only be used with image directory"
        if os.path.exists(os.path.join(ref_image_dir_or_tensor, "cache.pt")):
            cache = torch.load(os.path.join(ref_image_dir_or_tensor, "cache.pt"))
            dataset1.image_files_or_tensor = cache["image_files_or_tensor"]
        else:
            dataset1.image_files_or_tensor = image_files_to_tensor(
                dataset1.image_files_or_tensor
            )
            torch.save(
                {"image_files_or_tensor": dataset1.image_files_or_tensor},
                os.path.join(ref_image_dir_or_tensor, "cache.pt"),
            )

    dataloader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=32, shuffle=False, drop_last=False, num_workers=4
    )
    emb1 = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader1):
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
            emb1.append((model(**inputs).pooler_output).detach().cpu().numpy())
    emb1 = np.vstack(emb1)

    sim_mtx = np.dot(emb1, emb1.T)
    distance_matrix = np.max(sim_mtx) - sim_mtx

    np.fill_diagonal(distance_matrix, 0)

    seed_everything(42)
    # Ward's linkage clustering
    # Convert to a condensed distance matrix for ward's linkage (if needed)
    condensed_distance_matrix = squareform(distance_matrix)
    linkage_matrix = ward(condensed_distance_matrix)
    ref_cluster_labels = fcluster(linkage_matrix, num_cluster, criterion="maxclust")

    dataset2 = ImageDataset(generated_images_dir_or_tensor, processor)
    dataloader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=32, shuffle=False, drop_last=False, num_workers=4
    )
    emb2 = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader2):
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
            emb2.append((model(**inputs).pooler_output).detach().cpu().numpy())
    emb2 = np.vstack(emb2)

    sim_to_emb1 = np.dot(emb2, emb1.T)
    dist_to_emb1 = np.max(sim_mtx) - sim_to_emb1

    # Allocate each new image to a cluster
    new_image_labels = []
    for distances in dist_to_emb1:
        cluster_distances = np.zeros(num_cluster)
        for i in range(1, num_cluster + 1):
            cluster_indices = np.where(ref_cluster_labels == i)[0]
            cluster_distances[i - 1] = np.mean(distances[cluster_indices])
        nearest_cluster = (
            np.argmin(cluster_distances) + 1
        )  # Cluster assignment for one new image
        new_image_labels.append(nearest_cluster)

    # Calculate proportions of each cluster
    new_image_labels = np.array(new_image_labels)
    cluster_count = np.zeros(num_cluster)
    for i in range(1, num_cluster + 1):
        cluster_count[i - 1] = np.sum(new_image_labels == i)

    cluster_proportions = cluster_count / len(new_image_labels)

    # Entropy calculation.
    entropy = -np.sum(
        cluster_proportions * np.log2(cluster_proportions + np.finfo(float).eps)
    )

    # Map each reference image to its cluster
    ref_cluster_images = {i: [] for i in range(1, num_cluster + 1)}
    for i, cluster_id in enumerate(ref_cluster_labels):
        ref_cluster_images[cluster_id].append(dataset1.image_files_or_tensor[i])

    new_cluster_images = {i: [] for i in range(1, num_cluster + 1)}
    for i, cluster_id in enumerate(new_image_labels):
        new_cluster_images[cluster_id].append(dataset2.image_files_or_tensor[i])

    return (
        entropy,
        cluster_count.tolist(),
        cluster_proportions.tolist(),
        ref_cluster_images,
        new_cluster_images,
    )


def calcualte_embedding_dist(
    dataset_name, ref_image_dir_or_tensor, num_cluster, use_cache=True, by="mean"
):
    """Function to calculate l2 distance of reference samples"""
    processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model = model.vision_model.to("cuda")
    model.eval()

    dataset1 = ImageDataset(ref_image_dir_or_tensor, processor)

    if use_cache:
        assert isinstance(
            ref_image_dir_or_tensor, str
        ), "Cache can only be used with image directory"
        if os.path.exists(os.path.join(ref_image_dir_or_tensor, "cache.pt")):
            cache = torch.load(os.path.join(ref_image_dir_or_tensor, "cache.pt"))
            dataset1.image_files_or_tensor = cache["image_files_or_tensor"]
        else:
            dataset1.image_files_or_tensor = image_files_to_tensor(
                dataset1.image_files_or_tensor
            )
            torch.save(
                {"image_files_or_tensor": dataset1.image_files_or_tensor},
                os.path.join(ref_image_dir_or_tensor, "cache.pt"),
            )

    dataloader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=32, shuffle=False, drop_last=False, num_workers=1
    )
    emb1 = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader1):
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
            emb1.append((model(**inputs).pooler_output).detach().cpu().numpy())
    emb1 = np.vstack(emb1)

    # L2 distance between training samples and cluster mean.

    dist_to_mean = np.linalg.norm(emb1 - np.mean(emb1, axis=0), axis=1)
    dataset = create_dataset(dataset_name=dataset_name, train=True)
    coeff = aggregate_by_class(dist_to_mean, dataset, by=by)

    return coeff


def plot_cluster_proportions(cluster_proportions, num_cluster):
    """Helper function that plot the histogram of the clusters"""

    fig = plt.figure(figsize=(10, 6))  # Create a figure with specified dimensions
    ax = fig.add_subplot(111)  # Add a subplot to the figure
    ax.bar(
        range(1, num_cluster + 1), cluster_proportions, color="blue"
    )  # Plot the bar chart
    ax.set_xlabel("Cluster")  # Label the x-axis
    ax.set_ylabel("Proportion")  # Label the y-axis
    ax.set_title(
        "Proportion of Generated Images per Cluster"
    )  # Set the title of the plot
    ax.set_xticks(range(1, num_cluster + 1))  # Set the tick marks on the x-axis

    return fig


def plot_cluster_images(ref_cluster_images, new_cluster_images, num_cluster):
    """Helper function that plot images for each cluster"""

    # Plotting the images
    num_sample_ref = 10
    num_sample_new = 10

    fig, axs = plt.subplots(
        num_cluster,
        20,
        figsize=(2.5 * (num_sample_ref + num_sample_new), num_cluster * 2.5),
    )  # 20 columns for ref and new images
    for cluster_id, paths_or_tensors in new_cluster_images.items():
        selected_new_images = (
            random.Random(42).sample(paths_or_tensors, num_sample_new)
            if len(paths_or_tensors) > num_sample_new
            else paths_or_tensors
        )
        selected_ref_images = (
            random.Random(42).sample(ref_cluster_images[cluster_id], num_sample_ref)
            if len(ref_cluster_images[cluster_id]) > num_sample_ref
            else ref_cluster_images[cluster_id]
        )

        # Display reference images
        for col, img_path_or_tensor in enumerate(selected_ref_images):

            if isinstance(img_path_or_tensor, torch.Tensor):
                # print(img_path_or_tensor.shape)
                img = Image.fromarray(
                    (255 * img_path_or_tensor.permute(1, 2, 0).numpy()).astype(np.uint8)
                )
            elif isinstance(img_path_or_tensor, str):
                img = Image.open(img_path_or_tensor)
            else:
                raise ValueError("Image directory or tensor should be provided")
            axs[cluster_id - 1, col].imshow(img)
            axs[cluster_id - 1, col].axis("off")
        axs[cluster_id - 1, 0].set_title(f"Cluster {cluster_id} (Ref)")

        # Display new images
        for col, img_path_or_tensor in enumerate(selected_new_images):
            if isinstance(img_path_or_tensor, torch.Tensor):
                # print(img_path_or_tensor.shape)
                img = Image.fromarray(
                    (255 * img_path_or_tensor.permute(1, 2, 0).numpy()).astype(np.uint8)
                )
            elif isinstance(img_path_or_tensor, str):
                img = Image.open(img_path_or_tensor)
            else:
                raise ValueError("Image directory or tensor should be provided")
            axs[cluster_id - 1, col + 10].imshow(img)  # Start from column 11
            axs[cluster_id - 1, col + 10].axis("off")
        axs[cluster_id - 1, 10].set_title(f"Cluster {cluster_id} (New)")

    plt.tight_layout()

    return fig
