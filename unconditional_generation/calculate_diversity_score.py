"""Converting images to BLIP embedding"""

import argparse
import glob
import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import requests
import torch
import tqdm
from lightning.pytorch import seed_everything
from PIL import Image
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import squareform
from transformers import (
    BlipForQuestionAnswering,
    BlipImageProcessor,
    BlipProcessor,
    BlipVisionModel,
)

import src.constants as constants


def load_images(image_dir):
    """Load images for a given directory"""
    image_files = os.listdir(image_dir)
    images = []

    for file in image_files:
        file_path = os.path.join(image_dir, file)
        try:
            image = Image.open(file_path).convert("RGB")
            images.append(image)
        except IOError:
            print(f"Error loading image: {file_path}")
    return images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--celeba_images_dir",
        type=str,
        help="directory path for based  cluster images",
        default=None,
    )

    parser.add_argument(
        "--generated_images_dir",
        type=str,
        help="directory path for generated images",
        default=None,
    )
    parser.add_argument(
        "--num_cluster",
        type=int,
        help="number of clusters",
        default=None,
    )
    return parser.parse_args()


class ImageDataset(torch.utils.data.Dataset):

    def __init__(self, image_dir, processor):
        self.image_files = [
            file
            for file in glob.glob(image_dir + "/*")
            if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".jpeg")
        ]
        self.processor = processor

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_path = self.image_files[idx]
        image = Image.open(file_path).convert("RGB")
        tensor = self.processor(images=image, return_tensors="pt")
        assert len(tensor["pixel_values"]) == 1, "Batch size should be 1"
        tensor["pixel_values"] = tensor["pixel_values"][0]

        return tensor


def main(args):
    """Main function for calculating entropy based on diversed celebrity images."""

    seed_everything(42)

    processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
    # model = BlipVisionModel.from_pretrained("Salesforce/blip-vqa-base").to("cuda")
    # model.eval()

    # processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
    model = model.vision_model.to("cuda")

    # celeb_images = load_images(args.celeba_images_dir)
    # generated_images = load_images(args.celeba_images_dir)

    # inputs = processor(images=celeb_images, return_tensors="pt").to("cuda")
    # emb1 = (model(**inputs).pooler_output).detach().cpu().numpy()

    # inputs = processor(images=generated_images, return_tensors="pt").to("cuda")
    # emb2 = (model(**inputs).pooler_output).detach().cpu().numpy()

    dataset1 = ImageDataset(args.celeba_images_dir, processor)
    dataloader1 = torch.utils.data.DataLoader(
        dataset1, batch_size=32, shuffle=False, drop_last=False, num_workers=4
    )
    emb1 = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader1):
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
            emb1.append((model(**inputs).pooler_output).detach().cpu().numpy())
    emb1 = np.vstack(emb1)

    dataset2 = ImageDataset(args.generated_images_dir, processor)
    dataloader2 = torch.utils.data.DataLoader(
        dataset2, batch_size=32, shuffle=False, drop_last=False, num_workers=4
    )
    emb2 = []
    with torch.no_grad():
        for inputs in tqdm.tqdm(dataloader2):
            inputs["pixel_values"] = inputs["pixel_values"].to("cuda")
            emb2.append((model(**inputs).pooler_output).detach().cpu().numpy())
    emb2 = np.vstack(emb2)

    sim_mtx = np.dot(emb1, emb1.T)
    distance_matrix = np.max(sim_mtx) - sim_mtx

    np.fill_diagonal(distance_matrix, 0)

    # Ward's linkage clustering
    # Convert to a condensed distance matrix for ward's linkage (if needed)
    condensed_distance_matrix = squareform(distance_matrix)
    linkage_matrix = ward(condensed_distance_matrix)
    cluster_labels = fcluster(linkage_matrix, args.num_cluster, criterion="maxclust")

    sim_to_emb1 = np.dot(emb2, emb1.T)
    dist_to_emb1 = np.max(sim_mtx) - sim_to_emb1

    # Allocate each new image to a cluster
    new_image_clusters = []
    for distances in dist_to_emb1:
        cluster_distances = np.zeros(args.num_cluster)
        for i in range(1, args.num_cluster + 1):
            cluster_indices = np.where(cluster_labels == i)[0]
            cluster_distances[i - 1] = np.mean(distances[cluster_indices])
        nearest_cluster = (
            np.argmin(cluster_distances) + 1
        )  # Cluster assignment for one new image
        new_image_clusters.append(nearest_cluster)

    # Calculate proportions of each cluster
    new_image_clusters = np.array(new_image_clusters)
    cluster_proportions = np.zeros(args.num_cluster)
    for i in range(1, args.num_cluster + 1):
        cluster_proportions[i - 1] = np.sum(new_image_clusters == i) / len(
            new_image_clusters
        )

    # Entropy calculation.
    entropy = -np.sum(
        cluster_proportions * np.log2(cluster_proportions + np.finfo(float).eps)
    )

    print(f"Entropy of the distribution across clusters: {entropy}")

    output_dir = os.path.join(constants.OUTDIR, "celeba/diveristy_score")
    os.makedirs(output_dir, exist_ok=True)
    # time stamp MM_DD_HH_MM_SS

    output_dir = os.path.join(
        constants.OUTDIR, "celeba/diveristy_score", time.strftime("%m_%d_%H_%M_%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    # Plot the histogram of the clusters
    fig = plt.figure(figsize=(10, 6))  # Create a figure with specified dimensions
    ax = fig.add_subplot(111)  # Add a subplot to the figure
    ax.bar(
        range(1, args.num_cluster + 1), cluster_proportions, color="blue"
    )  # Plot the bar chart
    ax.set_xlabel("Cluster")  # Label the x-axis
    ax.set_ylabel("Proportion")  # Label the y-axis
    ax.set_title(
        "Proportion of Generated Images per Cluster"
    )  # Set the title of the plot
    ax.set_xticks(range(1, args.num_cluster + 1))  # Set the tick marks on the x-axis
    fig.savefig(
        os.path.join(output_dir, "cluster_proportions.png")
    )  # Save the plot as a PNG file

    # plot images for each cluster

    # Map each reference image to its cluster
    ref_cluster_images = {i: [] for i in range(1, args.num_cluster + 1)}
    for i, cluster_id in enumerate(cluster_labels):
        ref_cluster_images[cluster_id].append(dataset1.image_files[i])

    cluster_images = {i: [] for i in range(1, args.num_cluster + 1)}
    for i, cluster_id in enumerate(new_image_clusters):
        cluster_images[cluster_id].append(dataset2.image_files[i])

    # Plotting the images
    num_sample_ref = 10
    num_sample_new = 10

    fig, axs = plt.subplots(
        args.num_cluster,
        20,
        figsize=(2.5 * (num_sample_ref + num_sample_new), args.num_cluster * 2.5),
    )  # 20 columns for ref and new images
    for cluster_id, paths in cluster_images.items():
        selected_new_images = (
            random.sample(paths, num_sample_new)
            if len(paths) > num_sample_new
            else paths
        )
        selected_ref_images = (
            random.sample(ref_cluster_images[cluster_id], num_sample_ref)
            if len(ref_cluster_images[cluster_id]) > num_sample_ref
            else ref_cluster_images[cluster_id]
        )

        # Display reference images
        for col, img_path in enumerate(selected_ref_images):
            img = Image.open(img_path)
            axs[cluster_id - 1, col].imshow(img)
            axs[cluster_id - 1, col].axis("off")
        axs[cluster_id - 1, 0].set_title(f"Cluster {cluster_id} (Ref)")

        # Display new images
        for col, img_path in enumerate(selected_new_images):
            img = Image.open(img_path)
            axs[cluster_id - 1, col + 10].imshow(img)  # Start from column 11
            axs[cluster_id - 1, col + 10].axis("off")
        axs[cluster_id - 1, 10].set_title(f"Cluster {cluster_id} (New)")

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "cluster_images.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
