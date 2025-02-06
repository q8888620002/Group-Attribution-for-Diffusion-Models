"""Calculate model behavior scores for diffusion models."""

import argparse
import json
import os
import time

import numpy as np
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Subset
from skimage.metrics import  structural_similarity
from PIL import Image

import src.constants as constants
from src.attributions.global_scores.diversity_score import (
    calculate_diversity_score,
    plot_cluster_images,
    plot_cluster_proportions,
)
from src.datasets import (
    ImageDataset,
    TensorDataset,
    create_dataset,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)
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
        default=64,
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
        excluded_class = [int(k) for k in args.excluded_class.split(",")]
        excluded_class.sort()
        excluded_class_str = ",".join(map(str, excluded_class))

        removal_dir = f"excluded_{excluded_class_str}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"
        # Loading training images

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)
    if args.excluded_class is not None:
        excluded_class = [int(k) for k in args.excluded_class.split(",")]
        remaining_idx, removed_idx = remove_data_by_class(
            train_dataset, excluded_class=excluded_class
        )
    elif args.removal_dist is not None:
        if args.removal_dist == "uniform":
            remaining_idx, removed_idx = remove_data_by_uniform(
                train_dataset, seed=args.removal_seed
            )
        elif args.removal_dist == "datamodel":
            if args.dataset in ["cifar100", "cifar100_f", "celeba"]:
                remaining_idx, removed_idx = remove_data_by_datamodel(
                    train_dataset,
                    alpha=args.datamodel_alpha,
                    seed=args.removal_seed,
                    by_class=True,
                )
            else:
                remaining_idx, removed_idx = remove_data_by_datamodel(
                    train_dataset,
                    alpha=args.datamodel_alpha,
                    seed=args.removal_seed,
                )
        elif args.removal_dist == "shapley":
            if args.dataset in ["cifar100", "cifar100_f", "celeba"]:
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed, by_class=True
                )
            else:
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed
                )
        else:
            raise NotImplementedError
    else:
        remaining_idx = np.arange(len(train_dataset))
        removed_idx = np.array([], dtype=int)

    if len(remaining_idx) < args.batch_size:
        remaining_dataloader = DataLoader(
            Subset(train_dataset, remaining_idx),
            batch_size=len(remaining_idx),
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
    else:
        remaining_dataloader = DataLoader(
            Subset(train_dataset, remaining_idx),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
        )
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
    behavior_start_time = time.time()
    generated_samples = generate_images(args, pipeline)
    (
        entropy,
        cluster_count,
        cluster_proportions,
        ref_cluster_images,
        new_cluster_images,
    ) = calculate_diversity_score(
        ref_image_dir_or_tensor=os.path.join(
            constants.OUTDIR, args.dataset, "cluster_imgs"
        ),
        generated_images_dir_or_tensor=generated_samples,
        num_cluster=20,
    )

    # best_match_indices = [-1] * len(
    #     generated_samples
    # )  # Store indices of training images with highest SSIM
    # memorization_results = {i: 0 for i in range(args.n_samples)}

    starttime = time.time()
    # sample_idx = 0

    # for batch_idx, batch_imgs in enumerate(remaining_dataloader):
    #     batch_imgs = batch_imgs[0].numpy().astype(np.float32)
    #     batch_imgs_flipped = np.flip(batch_imgs, axis=3)

    #     for idx, gen_image in enumerate(generated_samples):
    #         gen_image = gen_image.numpy().astype(np.float32)

    #         ssim_vals_original = [
    #             structural_similarity(
    #                 gen_image, train_img, channel_axis=0, data_range=1.0
    #             )
    #             for train_img in batch_imgs
    #         ]
    #         ssim_vals_flipped = [
    #             structural_similarity(
    #                 gen_image, train_img_flipped, channel_axis=0, data_range=1.0
    #             )
    #             for train_img_flipped in batch_imgs_flipped
    #         ]

    #         max_ssim_vals = np.maximum(ssim_vals_original, ssim_vals_flipped)
    #         max_idx = np.argmax(max_ssim_vals)
    #         best_ssim = max_ssim_vals[max_idx]

    #         if best_ssim > memorization_results[idx]:
    #             memorization_results[idx] = best_ssim
    #             best_match_indices[idx] = (
    #                 max_idx + sample_idx
    #             )  # Calculate global index of the training image

    #     sample_idx += len(batch_imgs)

    # print(f"calculation time: {time.time() - starttime}.")
    # output_folder = "results/celeba"  

    # for idx, gen_image in enumerate(generated_samples):
    #     gen_image = gen_image.numpy().astype(np.float32)

    #     best_match_idx = best_match_indices[idx]
    #     if best_match_idx != -1:
    #         best_training_img = remaining_dataloader.dataset[best_match_idx][
    #             0
    #         ].numpy()  # Access the corresponding best match image

    #         # Convert arrays to images
    #         gen_img = Image.fromarray(
    #             np.rollaxis((gen_image * 255).astype(np.uint8), 0, 3), "RGB"
    #         )
    #         train_img = Image.fromarray(
    #             np.rollaxis((best_training_img * 255).astype(np.uint8), 0, 3), "RGB"
    #         )

    #         total_width = gen_img.width + train_img.width
    #         max_height = max(gen_img.height, train_img.height)
    #         combined_img = Image.new("RGB", (total_width, max_height))
    #         combined_img.paste(gen_img, (0, 0))
    #         combined_img.paste(train_img, (gen_img.width, 0))

    #         # Save the combined image
    #         combined_img.save(f"{output_folder}/{idx}_image_ssim={memorization_results[idx]}.png")

    # info_dict["mem_results"] = sum(
    #     1 for value in memorization_results.values() if value > 0.6
    # ) / len(memorization_results)
    # info_dict["num_classes"] = len(remaining_idx)

    # sample_fig = plot_cluster_images(
    #     ref_cluster_images=ref_cluster_images,
    #     new_cluster_images=new_cluster_images,
    #     num_cluster=20,
    # )

    # hist_fig = plot_cluster_proportions(
    #     cluster_proportions=cluster_proportions, num_cluster=20
    # )

    print(f"entropy {entropy}")
    info_dict["entropy"] = entropy

    info_dict["sample_dir"] = args.sample_dir
    info_dict["cluster_count"] = cluster_count
    info_dict["cluster_proportions"] = cluster_proportions
    info_dict["remaining_idx"] = remaining_idx
    info_dict["removed_idx"] = removed_idx
    info_dict["total_sampling_time"] = time.time() - behavior_start_time

    with open(args.db, "a+") as f:
        f.write(json.dumps(info_dict) + "\n")
    print(f"Results saved to the database at {args.db}")

    os.makedirs(args.db.replace(".jsonl", ""), exist_ok=True)
    # print(args.db.replace(".jsonl","."))

    # sample_fig.savefig(
    #     os.path.join(
    #         args.db.replace(".jsonl", ""),
    #         os.path.join(
    #             args.dataset,
    #             args.method,
    #             "models",
    #             removal_dir,
    #         ).replace("/", "_")
    #         + "_sample.jpg",
    #     )
    # )

    # hist_fig.savefig(
    #     os.path.join(
    #         args.db.replace(".jsonl", ""),
    #         os.path.join(
    #             args.dataset,
    #             args.method,
    #             "models",
    #             removal_dir,
    #         ).replace("/", "_")
    #         + "_hist.jpg",
    #     )
    # )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
