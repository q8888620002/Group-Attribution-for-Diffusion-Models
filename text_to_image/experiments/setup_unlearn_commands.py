"""Set up the commands for each experiment that runs train_text_to_image_lora.py"""

import argparse
import os

import pandas as pd

from src.constants import DATASET_DIR, LOGDIR, TMP_OUTDIR
from src.ddpm_config import LoraTrainingConfig, TextToImageModelBehaviorConfig
from src.experiment_utils import format_config_arg, update_job_file
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset to run the experiment on",
        choices=["artbench_post_impressionism"],
        default="artbench_post_impressionism",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for model optimization (e.g., weight initialization)",
        default=42,
    )
    parser.add_argument(
        "--model_behavior_seed",
        type=int,
        help="random seed for computing model behaviors",
        default=42,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="number of images to generate for computing model behaviors",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["sparse_gd", "gd"],
        default="sparse_gd",
        help="unlearning method",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["shapley", "loo", "aoi", "uniform"],
        default="shapley",
    )
    parser.add_argument(
        "--removal_unit",
        type=str,
        help="unit of data for removal",
        choices=["artist", "filename"],
        default="artist",
    )
    parser.add_argument(
        "--num_removal_subsets",
        type=int,
        help="number of removal subsets to run",
        default=1000,
    )
    parser.add_argument(
        "--num_subsets_per_job",
        type=int,
        help="number of removal subsets to run for each SLURM job",
        default=20,
    )
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    if args.dataset == "artbench_post_impressionism":
        training_config = LoraTrainingConfig.artbench_post_impressionism_config
        train_data_dir = os.path.join(
            DATASET_DIR, "artbench-10-imagefolder-split", "train"
        )
        training_config["train_data_dir"] = train_data_dir
        training_config["output_dir"] = os.path.join(TMP_OUTDIR, f"seed{args.seed}")
        training_config["seed"] = args.seed
        training_config["method"] = args.method
        training_config["removal_dist"] = args.removal_dist
        training_config["removal_unit"] = args.removal_unit

        model_behavior_config = (
            TextToImageModelBehaviorConfig.artbench_post_impressionism_config
        )
        model_behavior_config["seed"] = args.model_behavior_seed
        model_behavior_config["num_images"] = args.num_images

        if args.removal_dist in ["loo", "aoi"]:
            if args.removal_unit == "artist":
                args.num_removal_subsets = 258
            else:
                args.num_removal_subsets = 5000
    else:
        raise ValueError("--dataset should be one of ['artbench_post_impressionism']")

    # Set up coutput directories and files.
    removal_dist_name = "full"
    if args.removal_dist is not None:
        removal_dist_name = f"{args.removal_unit}_{args.removal_dist}"

    exp_name = os.path.join(
        args.dataset,
        args.method,
        removal_dist_name,
        f"seed{args.seed}",
    )
    command_outdir = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "commands", "unlearn", exp_name
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "unlearn", exp_name)
    os.makedirs(logdir, exist_ok=True)

    db_dir = os.path.join(TMP_OUTDIR, f"seed{args.seed}", args.dataset)
    os.makedirs(db_dir, exist_ok=True)
    db = os.path.join(
        db_dir, f"{args.method}_{args.removal_unit}_{args.removal_dist}.jsonl"
    )

    idx_list = [i for i in range(args.num_removal_subsets)]
    if os.path.exists(db):
        df = pd.read_json(db, lines=True)
        if args.removal_dist in ["shapley", "uniform"]:
            df["removal_seed"] = (
                df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
            )
            existing_idx_list = df["removal_seed"].tolist()
        else:
            df["group_idx"] = (
                df["exp_name"].str.split("idx_", expand=True)[1].astype(int)
            )
            existing_idx_list = df["group_idx"].tolist()

        idx_list = set(idx_list) - set(existing_idx_list)
        idx_list = sorted(list(idx_list))
        if len(idx_list) == 0:
            print("Model behaviors have already been computed for all subsets!")
        elif 0 < len(idx_list) < args.num_removal_subsets:
            print(f"Only {len(idx_list)} subsets are missing model behaviors")
    assert len(idx_list) % args.num_subsets_per_job == 0

    ckpt_dir = os.path.join(
        LOGDIR, f"seed{args.seed}", args.dataset, "model_behaviors", "unlearn", exp_name
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    num_jobs = 0
    with open(command_file, "w") as handle:
        command = ""
        for i, idx in enumerate(idx_list):
            command += "accelerate launch"
            command += " --gpu_ids=0"
            command += " --mixed_precision={}".format("fp16")
            command += " text_to_image/train_text_to_image_lora.py"
            for key, val in training_config.items():
                command += " " + format_config_arg(key, val)
            if args.removal_dist in ["shapley", "uniform"]:
                command += f" --removal_seed={idx}"
            elif args.removal_dist == "loo":
                command += f" --loo_idx={idx}"
            else:
                command += f" --aoi_idx={idx}"

            command += " ; "

            command += "python text_to_image/compute_model_behaviors.py"
            for key, val in model_behavior_config.items():
                command += " " + format_config_arg(key, val)

            if args.removal_dist in ["shapley", "uniform"]:
                ckpt_path = os.path.join(ckpt_dir, f"{args.removal_dist}_seed_{idx}.pt")
                lora_dir = os.path.join(
                    TMP_OUTDIR,
                    f"seed{args.seed}",
                    args.dataset,
                    args.method,
                    "models",
                    f"{args.removal_unit}_{args.removal_dist}",
                    f"{args.removal_dist}_seed={idx}",
                )
                idx_exp_name = os.path.join(exp_name, f"{args.removal_dist}_seed_{idx}")
            else:
                ckpt_path = os.path.join(ckpt_dir, f"{args.removal_dist}_idx_{idx}.pt")
                lora_dir = os.path.join(
                    TMP_OUTDIR,
                    f"seed{args.seed}",
                    args.dataset,
                    args.method,
                    "models",
                    f"{args.removal_unit}_{args.removal_dist}",
                    f"{args.removal_dist}_idx={idx}",
                )
                idx_exp_name = os.path.join(exp_name, f"{args.removal_dist}_idx_{idx}")

            command += " --ckpt_path={}".format(ckpt_path)
            command += " --db={}".format(db)
            command += " --exp_name={}".format(idx_exp_name)
            command += " --lora_dir={}".format(lora_dir)

            if (i + 1) % args.num_subsets_per_job == 0:
                handle.write(command + "\n")
                command = ""
                num_jobs += 1
            else:
                command += " ; "
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(os.getcwd(), "text_to_image", "experiments", "unlearn.job")
    array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    job_name = "unlearn-" + exp_name.replace("/", "-")
    output = os.path.join(logdir, "run-%A-%a.out")
    update_job_file(
        job_file=job_file,
        job_name=job_name,
        output=output,
        array=array,
        command_file=command_file,
    )


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
