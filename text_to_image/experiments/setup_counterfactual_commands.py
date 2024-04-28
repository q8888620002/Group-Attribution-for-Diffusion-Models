"""Set up the commands for counterfactual model retrainings."""

import argparse
import os

from src.constants import DATASET_DIR, LOGDIR, OUTDIR
from src.ddpm_config import LoraTrainingConfig
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
        "--removal_unit",
        type=str,
        help="unit of data for removal",
        choices=["artist", "filename"],
        default="artist",
    )
    parser.add_argument(
        "--removal_rank_dir",
        type=str,
        help="directory containing removal rank files",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--removal_rank_proportion",
        type=float,
        help="proportion of top ranked units to remove",
        default=0.1,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        help="number of generated images for local model behaviors",
        default=5,
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
        training_config["output_dir"] = os.path.join(OUTDIR, f"seed{args.seed}")
        training_config["seed"] = args.seed
        training_config["method"] = "retrain"
        training_config["removal_unit"] = args.removal_unit
        training_config["removal_rank_proportion"] = args.removal_rank_proportion
    else:
        raise ValueError("--dataset should be one of ['artbench_post_impressionism']")

    # Set up coutput directories and files.
    exp_name = os.path.join(
        args.dataset,
        f"counterfactual_top_{args.removal_rank_proportion}",
        f"seed{args.seed}",
    )
    command_outdir = os.path.join(
        os.getcwd(),
        "text_to_image",
        "experiments",
        "commands",
        "counterfactual",
        exp_name,
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "counterfactual", exp_name)
    os.makedirs(logdir, exist_ok=True)

    rank_method_list = ["max_pixel_similarity", "max_clip_similarity"]
    prefix_list = [f"all_generated_images_{args.removal_unit}_rank"]
    for i in range(args.num_images):
        prefix_list.append(f"generated_image_{i}_{args.removal_unit}_rank")

    num_jobs = 0
    with open(command_file, "w") as handle:
        command = ""
        for rank_method in rank_method_list:
            for prefix in prefix_list:
                command += "accelerate launch"
                command += " --gpu_ids=0"
                command += " --mixed_precision={}".format("fp16")
                command += " text_to_image/train_text_to_image_lora.py"
                for key, val in training_config.items():
                    command += " " + format_config_arg(key, val)
                removal_rank_file = os.path.join(
                    args.removal_rank_dir, f"{prefix}_{rank_method}.npy"
                )
                command += " --removal_rank_file={}".format(removal_rank_file)

                handle.write(command + "\n")
                command = ""
                num_jobs += 1
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "counterfactual.job"
    )
    array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    job_name = "train-" + exp_name.replace("/", "-")
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
