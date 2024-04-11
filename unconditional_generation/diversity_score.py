"""Converting images to BLIP embedding"""
import argparse
import os

import numpy as np
from PIL import Image
from scipy.cluster.hierarchy import fcluster, ward
from scipy.spatial.distance import squareform
from transformers import BlipImageProcessor, BlipVisionModel


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


def main(args):
    """Main function for calculating entropy based on diversed celebrity images."""

    processor = BlipImageProcessor.from_pretrained("Salesforce/blip-vqa-base")
    model = BlipVisionModel.from_pretrained("Salesforce/blip-vqa-base")

    celeb_images = load_images(args.celeba_images_dir)
    generated_images = load_images(args.celeba_images_dir)

    inputs = processor(images=celeb_images, return_tensors="pt")
    emb1 = (model(**inputs).pooler_output).detach().cpu().numpy()

    inputs = processor(images=generated_images, return_tensors="pt")
    emb2 = (model(**inputs).pooler_output).detach().cpu().numpy()

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


if __name__ == "__main__":
    args = parse_args()
    main(args)
