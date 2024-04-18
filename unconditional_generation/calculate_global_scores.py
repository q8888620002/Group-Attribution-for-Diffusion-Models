"""Calculate model behavior scores for diffusion models."""
import argparse
import json
import os

from lightning.pytorch import seed_everything

import src.constants as constants
from src.attributions.global_scores import fid_score, inception_score, precision_recall
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
    args = parser.parse_args()
    return args


def main(args):
    """Main function for calculating global model behaviors."""
    seed_everything(args.seed)
    info_dict = vars(args)

    # Check if there's need to generate samples.
    dims = 2048

    if not args.sample_dir:
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

        model, ema_model, remaining_idx, removed_idx = load_ckpt_model(
            args, model_loaddir
        )
        if args.use_ema:
            ema_model.copy_to(model.parameters())

        pipeline, vqvae, vqvae_latent_dict = build_pipeline(args, model)

        generated_samples = generate_images(args, pipeline)

        images_dataset = TensorDataset(generated_samples)

        is_value = inception_score.eval_is(
            images_dataset, args.batch_size, resize=True, normalize=True
        )

        precision, recall = precision_recall.eval_pr(
            args.dataset,
            images_dataset,
            args.batch_size,
            row_batch_size=10000,
            col_batch_size=10000,
            nhood_size=3,
            device=args.device,
            reference_dir=args.reference_dir,
        )

        fid_value_str = fid_score.calculate_fid(
            args.dataset,
            images_dataset,
            args.batch_size,
            args.device,
            args.reference_dir,
        )

        print(
            f"FID score: {fid_value_str}; Precision:{precision};"
            f"Recall:{recall}; inception score: {is_value}"
        )
        info_dict["fid_value"] = fid_value_str
        info_dict["precision"] = precision
        info_dict["recall"] = recall
        info_dict["is"] = is_value

    else:
        # Check if subdirectories exist for conditional image generation.
        subdir_list = [
            entry
            for entry in os.listdir(args.sample_dir)
            if os.path.isdir(os.path.join(args.sample_dir, entry))
        ]
        if len(subdir_list) == 0:
            # Aggregate FID score. This is the standard practice even for
            # conditional image generation. For example, see
            # https://huggingface.co/docs/diffusers/main/en/conceptual/evaluation#class-conditioned-image-generation
            print("Calculating the FID score...")
            sample_images = ImageDataset(args.sample_dir)

            is_value = inception_score.eval_is(
                sample_images, args.batch_size, resize=True, normalize=True
            )

            fid_value = fid_score.calculate_fid_given_paths(
                paths=[args.sample_dir, args.reference_dir],
                batch_size=args.batch_size,
                device=args.device,
                dims=dims,
            )
            precision, recall = precision_recall.eval_pr(
                args.dataset,
                sample_images,
                args.batch_size,
                row_batch_size=10000,
                col_batch_size=10000,
                nhood_size=3,
                device=args.device,
                reference_dir=args.reference_dir,
            )

            fid_value_str = f"{fid_value:.4f}"

            print(
                f"FID score: {fid_value_str}; Precision:{precision}; "
                f"Recall:{recall}; inception score: {is_value}"
            )
            info_dict["fid_value"] = fid_value_str
            info_dict["precision"] = precision
            info_dict["recall"] = recall
            info_dict["is"] = is_value

        else:
            # Class-wise FID scores. If each class has too few reference samples, the
            # scores can be unstable.

            avg_is_value = 0
            avg_fid_value = 0
            avg_precision_value = 0
            avg_recall_value = 0

            for subdir in subdir_list:
                print(f"Calculating the FID score for class {subdir}...")

                sample_images = ImageDataset(os.path.join(args.sample_dir, subdir))

                is_value = inception_score.eval_is(
                    sample_images, args.batch_size, resize=True, normalize=True
                )

                fid_value = fid_score.calculate_fid_given_paths(
                    paths=[
                        os.path.join(args.sample_dir, subdir),
                        os.path.join(args.reference_dir, subdir),
                    ],
                    batch_size=args.batch_size,
                    device=args.device,
                    dims=dims,
                )
                precision, recall = precision_recall.eval_pr(
                    args.dataset,
                    sample_images,
                    args.batch_size,
                    row_batch_size=10000,
                    col_batch_size=10000,
                    nhood_size=3,
                    device=args.device,
                    reference_dir=os.path.join(args.reference_dir, subdir),
                )

                fid_value_str = f"{fid_value:.4f}"
                avg_is_value += is_value
                avg_fid_value += fid_value
                avg_precision_value += precision
                avg_recall_value += recall

                print(
                    f"FID/Precision/Recall score for {subdir}:"
                    f"{fid_value_str}/{precision}/{recall}/{is_value}"
                )
                info_dict[f"is_value/{subdir}"] = is_value
                info_dict[f"fid_value/{subdir}"] = fid_value_str
                info_dict[f"precision/{subdir}"] = precision
                info_dict[f"recall/{subdir}"] = recall

            avg_is_value /= len(subdir_list)
            avg_fid_value /= len(subdir_list)
            avg_precision_value /= len(subdir_list)
            avg_recall_value /= len(subdir_list)
            avg_is_value /= len(subdir_list)

            avg_fid_value_str = f"{avg_fid_value:.4f}"
            print(
                f"Average FID score:{avg_fid_value_str};Precision:{avg_precision_value}"
                f";Recall:{avg_recall_value}; inception score: {avg_is_value}"
            )

            info_dict["avg_is"] = avg_is_value
            info_dict["avg_fid_value"] = avg_fid_value_str
            info_dict["avg_precision"] = avg_precision_value
            info_dict["avg_recall"] = avg_recall_value

    info_dict["sample_dir"] = args.sample_dir
    info_dict["remaining_idx"] = remaining_idx
    info_dict["removed_idx"] = removed_idx

    with open(args.db, "a+") as f:
        f.write(json.dumps(info_dict) + "\n")
    print(f"Results saved to the database at {args.db}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
