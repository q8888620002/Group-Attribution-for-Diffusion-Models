"""Calculate model behavior scores for diffusion models."""

import argparse
import json
import os

from lightning.pytorch import seed_everything

import src.constants as constants
from src.attributions.global_scores.diversity_score import (
    calculate_diversity_score,
    plot_cluster_images,
    plot_cluster_proportions,
)
from src.datasets import ImageDataset, TensorDataset
from src.diffusion_utils import build_pipeline, generate_images, load_ckpt_model
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Calculate model behavior scores")
    parser.add_argument(
        "--sample_dir",
        type=str,
        help="directory path of samples generated by a model",
        default=None,
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="directory path of reference samples, from a dataset or a diffusion model",
        default=None,
    )
    parser.add_argument(
        "--outdir", type=str, help="results parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=constants.DATASET,
        default=None,
    )
    parser.add_argument(
        "--db",
        type=str,
        help="filepath of database for recording scores",
        required=True,
    )
    parser.add_argument(
        "--excluded_class",
        help='Classes to be excluded, e.g. "1, 2, 3, etc" ',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        default=None,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="proportion of full dataset to keep in the datamodel distribution",
        default=0.5,
    )
    parser.add_argument(
        "--removal_seed",
        type=int,
        help="random seed for sampling from the removal distribution",
        default=0,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=constants.METHOD,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="experiment name to record in the database file",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for computation",
        default=512,
    )
    parser.add_argument(
        "--device", type=str, help="device used for computation", default="cuda:0"
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for image sample generation",
        default=42,
    )
    # params for sample generation
    parser.add_argument(
        "--generate_samples",
        help="whether to generate samples",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_samples", type=int, default=100000, help="number of generated samples"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of diffusion steps for generating images",
    )
    parser.add_argument(
        "--use_ema",
        help="whether to use the EMA model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--trained_steps",
        type=int,
        help="steps for specific ckeck points",
        default=None,
    )
    # params for loading the pruned model
    parser.add_argument(
        "--pruning_ratio",
        type=float,
        help="ratio for remaining parameters.",
        default=0.3,
    )
    parser.add_argument(
        "--pruner",
        type=str,
        default="magnitude",
        choices=["taylor", "random", "magnitude", "reinit", "diff-pruning"],
    )
    parser.add_argument(
        "--thr", type=float, default=0.05, help="threshold for diff-pruning"
    )
    parser.add_argument(
        "--precompute_stage",
        type=str,
        default=None,
        choices=[None, "save", "reuse"],
        help=(
            "Whether to precompute the VQVAE output."
            "Choose between None, save, and reuse."
        ),
    )

    args = parser.parse_args()
    return args


def main(args):
    """Main function for calculating global model behaviors."""
    seed_everything(args.seed)
    info_dict = vars(args)

    # Check if there's need to generate samples.

    removal_dir = "full"
    if args.excluded_class is not None:
        removal_dir = f"excluded_{args.excluded_class}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"

    model_loaddir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        "models",
        removal_dir,
    )

    model, ema_model, remaining_idx, removed_idx = load_ckpt_model(args, model_loaddir)
    if args.use_ema:
        ema_model.copy_to(model.parameters())

    pipeline, vqvae, vqvae_latent_dict = build_pipeline(args, model)

    generated_samples = generate_images(args, pipeline)
    (
        entropy,
        cluster_proportions,
        ref_cluster_images,
        new_cluster_images,
    ) = calculate_diversity_score(
        ref_image_dir_or_tensor=os.path.join(
            constants.OUTDIR, args.dataset, "generated_samples"
        ),
        generated_images_dir_or_tensor=generated_samples,
        num_cluster=20,
    )

    sample_fig = plot_cluster_images(
        ref_cluster_images=ref_cluster_images,
        new_cluster_images=new_cluster_images,
        num_cluster=20,
    )
    # fig.savefig("test.jpg")

    hist_fig = plot_cluster_proportions(
        cluster_proportions=cluster_proportions, num_cluster=20
    )
    # fig.savefig("test2.jpg")

    print(f"entropy {entropy}")
    info_dict["entropy"] = entropy

    info_dict["sample_dir"] = args.sample_dir
    info_dict["remaining_idx"] = remaining_idx
    info_dict["removed_idx"] = removed_idx

    with open(args.db, "a+") as f:
        f.write(json.dumps(info_dict) + "\n")
    print(f"Results saved to the database at {args.db}")

    sample_fig.savefig(
        args.db.replace(
            ".jsonl",
            "."
            + os.path.join(
                args.dataset,
                args.method,
                "models",
                removal_dir,
            ).replace("/", "_"),
        )
        + "_sample.jpg"
    )
    hist_fig.savefig(
        args.db.replace(
            ".jsonl",
            "."
            + os.path.join(
                args.dataset,
                args.method,
                "models",
                removal_dir,
            ).replace("/", "_"),
        )
        + "_hist.jpg"
    )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
