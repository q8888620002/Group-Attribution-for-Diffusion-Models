"""Set up the commands for each experiment that runs train_text_to_image_lora.py"""

import argparse
import os

from src.constants import DATASET_DIR, LOGDIR, OUTDIR
from src.ddpm_config import LoraTrainingConfig
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
        "--method",
        type=str,
        choices=["retrain"],
        default="retrain",
        help="training or unlearning method",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["shapley"],
        default=None,
    )
    parser.add_argument(
        "--num_removal_subsets",
        type=int,
        help="number of removal subsets to run",
        default=500,
    )
    parser.add_argument(
        "--num_subsets_per_job",
        type=int,
        help="number of removal subsets to run for each SLURM job",
        default=1,
    )
    parser.add_argument(
        "--removal_unit",
        type=str,
        help="unit of data for removal",
        choices=["artist", "filename"],
        default=None,
    )
    args = parser.parse_args()
    return args


def format_config_arg(key, val):
    """Format a training configuration key-value pair as command line argument."""
    if val is None:
        command_arg = ""
    elif type(val) is bool:
        if val:
            command_arg = "--{}".format(key)
        else:
            command_arg = ""
    else:
        command_arg = "--{}={}".format(key, val)
    return command_arg


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
        training_config["method"] = args.method
        training_config["removal_dist"] = args.removal_dist
        training_config["removal_unit"] = args.removal_unit
    else:
        raise ValueError("--dataset should be one of ['artbench_post_impressionism']")

    # Set up coutput directories and files.
    exp_name = os.path.join(
        args.dataset,
        args.method,
        "full"
        if args.removal_dist is None
        else f"{args.removal_unit}_{args.removal_dist}",
    )
    command_outdir = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "commands", exp_name
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, exp_name)
    os.makedirs(logdir, exist_ok=True)

    num_jobs = 0
    if args.removal_dist is None:
        # Set up the full training command.
        command = "accelerate launch"
        command += " --gpu_ids=0"
        command += " --mixed_precision={}".format("fp16")
        command += " text_to_image/train_text_to_image_lora.py"
        for key, val in training_config.items():
            command += " " + format_config_arg(key, val)

        with open(command_file, "w") as handle:
            handle.write(command + "\n")
            num_jobs += 1
    else:
        assert args.num_removal_subsets % args.num_subsets_per_job == 0
        with open(command_file, "w") as handle:
            command = ""
            for seed in range(args.num_removal_subsets):
                command += "accelerate launch"
                command += " --gpu_ids=0"
                command += " --mixed_precision={}".format("fp16")
                command += " text_to_image/train_text_to_image_lora.py"
                for key, val in training_config.items():
                    command += " " + format_config_arg(key, val)
                command += f" --removal_seed={seed}"
                if (seed + 1) % args.num_subsets_per_job == 0:
                    handle.write(command + "\n")
                    command = ""
                    num_jobs += 1
                else:
                    command += " ; "
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(os.getcwd(), "text_to_image", "experiments", "train.job")
    job_array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    updated_job_lines = []
    with open(job_file, "r") as handle:
        job_lines = handle.readlines()
        for line in job_lines:
            if line.startswith("#SBATCH --job-name"):
                line = "#SBATCH --job-name=" + exp_name.replace("/", "-") + "\n"
            if line.startswith("#SBATCH --output"):
                line = "#SBATCH --output=" + os.path.join(logdir, "run-%A-%a.out\n")
            if line.startswith("#SBATCH --array"):
                line = f"#SBATCH --array={job_array}\n"
            if line.startswith("PARAMS_FILE"):
                line = 'PARAMS_FILE="{}"\n'.format(command_file)
            updated_job_lines.append(line)

    with open(job_file, "w") as handle:
        for line in updated_job_lines:
            handle.write(line)
    print(f"SLURM job file updated at {job_file}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
