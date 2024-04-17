"""Converting images to BLIP embedding"""

import argparse
import os
import time

from lightning.pytorch import seed_everything
from PIL import Image

import src.constants as constants
from src.attributions.global_scores.diversity_score import (
    calculate_diversity_score,
    plot_cluster_images,
    plot_cluster_proportions,
)


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

    seed_everything(42)

    (
        entropy,
        cluster_proportions,
        ref_cluster_images,
        new_cluster_images,
    ) = calculate_diversity_score(
        ref_image_dir=args.celeba_images_dir,
        generated_images_dir=args.generated_images_dir,
        num_cluster=args.num_cluster,
    )

    print(f"Entropy of the distribution across clusters: {entropy}")

    output_dir = os.path.join(constants.OUTDIR, "celeba/diveristy_score")
    os.makedirs(output_dir, exist_ok=True)
    # time stamp MM_DD_HH_MM_SS

    output_dir = os.path.join(
        constants.OUTDIR, "celeba/diveristy_score", time.strftime("%m_%d_%H_%M_%S")
    )
    os.makedirs(output_dir, exist_ok=True)

    fig = plot_cluster_proportions(
        cluster_proportions=cluster_proportions, num_cluster=args.num_cluster
    )
    fig.savefig(
        os.path.join(output_dir, "cluster_proportions.png")
    )  # Save the plot as a PNG file

    fig = plot_cluster_images(
        ref_cluster_images=ref_cluster_images,
        new_cluster_images=new_cluster_images,
        num_cluster=args.num_cluster,
    )

    fig.savefig(os.path.join(output_dir, "cluster_images.png"))


if __name__ == "__main__":
    args = parse_args()
    main(args)
