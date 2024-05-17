"""Run TRAK-related methods."""

import argparse
import os

import numpy as np
import pandas as pd
import torch

from src.constants import DATASET_DIR
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run TRAK-related methods.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output parent directory",
        required=True,
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="number of timesteps for computing the gradients",
        default=100,
    )
    parser.add_argument(
        "--proj_dim",
        type=int,
        help="projection dimension for the gradients",
        default=32768,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["artbench"],
        default="artbench",
        help="dataset",
    )
    parser.add_argument(
        "--cls",
        type=str,
        default="post_impressionism",
        help="class of images in the dataset",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="artist",
        choices=["artist", "filename"],
        help="unit for how to group images",
    )
    parser.add_argument(
        "--lam",
        type=float,
        help="factor to stablize kernel matrix inversion",
        default=5e-1,
    )
    args = parser.parse_args()
    args.gradient_dir = os.path.join(args.output_dir, "gradients")
    return args


def main(args):
    """Main function."""
    # Load the data frame with mapping between indices and group names.
    if args.dataset == "artbench":
        group_df = pd.read_csv(
            os.path.join(
                DATASET_DIR,
                "artbench-10-imagefolder-split",
                "train",
                f"{args.cls}_{args.group}s.csv",
            )
        )
    else:
        raise ValueError

    grad_file_suffix = f"num_timesteps={args.num_timesteps}_proj_dim={args.proj_dim}.pt"

    # Load the gradients for the training data.
    train_grads = torch.load(
        os.path.join(args.gradient_dir, "train", f"emb_f=loss_{grad_file_suffix}")
    ).to(
        "cuda"
    )  # train_size x proj_dim
    train_dtrak_grads = torch.load(
        os.path.join(
            args.gradient_dir, "train", f"emb_f=mean-squared-l2-norm_{grad_file_suffix}"
        )
    ).to(
        "cuda"
    )  # train_size x proj_dim

    # Find the training image indices for each group.
    train_df = pd.read_csv(os.path.join(args.gradient_dir, "train", "group.csv"))
    group_indices_dict = {}
    for i in group_df.index:
        group_name = group_df.iloc[i].item()
        group_indices = np.where(train_df[args.group] == group_name)[0]
        group_indices_dict[i] = group_indices

    # Load the gradients for the genearted images.
    gen_grads = torch.load(
        os.path.join(args.gradient_dir, "generated", f"emb_f=loss_{grad_file_suffix}")
    ).to(
        "cuda"
    )  # num_images x proj_dim
    gen_dtrak_grads = torch.load(
        os.path.join(
            args.gradient_dir,
            "generated",
            f"emb_f=mean-squared-l2-norm_{grad_file_suffix}",
        )
    ).to(
        "cuda"
    )  # num_images x proj_dim
    journey_grads = torch.load(
        os.path.join(
            args.gradient_dir,
            "generated_journey",
            (
                "emb_f=loss"
                + "_num_journey_points=50"
                + "_num_journey_noises=1"
                + f"_proj_dim={args.proj_dim}"
                + ".pt"
            ),
        )
    ).to(
        "cuda"
    )  # (num_images x num_journey_timesteps) x proj_dim

    # Compute global data attribution scores for each training image.
    sample_output_dict = {}

    # Gradient similarity.
    grad_sim = torch.matmul(gen_grads, train_grads.T)
    grad_sim /= torch.matmul(
        gen_grads.norm(dim=-1, keepdim=True), train_grads.norm(dim=-1, keepdim=True).T
    )
    grad_sim = grad_sim.mean(dim=0).cpu().numpy()
    sample_output_dict["grad_sim"] = grad_sim

    # TRAK.
    trak_inv_hessian_dot_products = torch.matmul(train_grads.T, train_grads)
    trak_inv_hessian_dot_products += args.lam * torch.eye(args.proj_dim).to("cuda")
    trak_inv_hessian_dot_products = torch.inverse(trak_inv_hessian_dot_products)
    trak_inv_hessian_dot_products = torch.matmul(
        trak_inv_hessian_dot_products, train_grads.T
    )  # proj_dim x train_size

    trak = torch.matmul(gen_grads, trak_inv_hessian_dot_products)
    trak = trak.mean(dim=0).cpu().numpy()
    sample_output_dict["trak"] = trak

    # Relative and renormalized influence.
    influence = torch.matmul(gen_grads, trak_inv_hessian_dot_products)
    relative_influence = influence / trak_inv_hessian_dot_products.norm(dim=0)
    relative_influence = relative_influence.mean(dim=0).cpu().numpy()
    sample_output_dict["relative_influence"] = relative_influence

    renorm_influence = influence / train_grads.norm(dim=-1)
    renorm_influence = renorm_influence.mean(dim=0).cpu().numpy()
    sample_output_dict["renorm_influence"] = renorm_influence

    # Journey TRAK.
    journey_trak = torch.matmul(journey_grads, trak_inv_hessian_dot_products)
    journey_trak = journey_trak.mean(dim=0).cpu().numpy()
    sample_output_dict["journey_trak"] = journey_trak

    # D-TRAK.
    dtrak_inv_hessian_dot_products = torch.matmul(
        train_dtrak_grads.T, train_dtrak_grads
    )
    dtrak_inv_hessian_dot_products += args.lam * torch.eye(args.proj_dim).to("cuda")
    dtrak_inv_hessian_dot_products = torch.inverse(dtrak_inv_hessian_dot_products)
    dtrak_inv_hessian_dot_products = torch.matmul(
        dtrak_inv_hessian_dot_products, train_dtrak_grads.T
    )  # proj_dim x train_size
    dtrak = torch.matmul(gen_dtrak_grads, dtrak_inv_hessian_dot_products)
    dtrak = dtrak.mean(dim=0).cpu().numpy()
    sample_output_dict["dtrak"] = dtrak

    # Aggregate attribution scores for each group.
    num_groups = len(group_indices_dict.keys())
    output_dict = {}
    for method, attrs in sample_output_dict.items():
        if method in ["grad_sim"]:
            group_avg_attrs = np.zeros(shape=(num_groups, 1))
            group_max_attrs = np.zeros(shape=(num_groups, 1))
            for i, group_indices in group_indices_dict.items():
                group_avg_attrs[i, :] = attrs[group_indices].mean()
                group_max_attrs[i, :] = attrs[group_indices].max()
            output_dict[f"avg_{method}"] = group_avg_attrs
            output_dict[f"max_{method}"] = group_max_attrs
        else:
            group_attrs = np.zeros(shape=(num_groups, 1))
            for i, group_indices in group_indices_dict.items():
                group_attrs[i, :] = attrs[group_indices].sum()
            output_dict[method] = group_attrs

    for group_attrs in output_dict.values():
        assert group_attrs.shape == (num_groups, 1)

    # Save results.
    output_dir = os.path.join(args.output_dir, "baselines")
    os.makedirs(output_dir, exist_ok=True)
    for name, output in output_dict.items():
        with open(os.path.join(output_dir, f"{args.group}_{name}.npy"), "wb") as handle:
            np.save(handle, output)

    # Rank groups and save ranked group indices.
    for name, output in output_dict.items():
        global_rank = np.argsort(-output.mean(axis=-1), kind="stable")
        with open(
            os.path.join(
                output_dir, f"all_generated_images_{args.group}_rank_{name}.npy"
            ),
            "wb",
        ) as handle:
            np.save(handle, global_rank)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
