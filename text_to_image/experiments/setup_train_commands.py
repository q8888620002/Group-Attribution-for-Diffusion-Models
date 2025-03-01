"""Set up the commands for each experiment that runs train_text_to_image_lora.py"""

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
        "--method",
        type=str,
        choices=["retrain", "sparse_gd", "gd"],
        default="retrain",
        help="training or unlearning method",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["shapley", "datamodel", "loo", "aoi", "uniform"],
        default=None,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="alpha value for the datamodel removal distribution",
        default=None,
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
        default=1,
    )
    parser.add_argument(
        "--removal_unit",
        type=str,
        help="unit of data for removal",
        choices=["artist", "filename"],
        default="artist",
    )
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    if args.removal_dist == "datamodel":
        assert args.datamodel_alpha is not None

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
        if args.removal_dist == "datamodel":
            training_config["datamodel_alpha"] = args.datamodel_alpha

        if args.removal_unit == "artist":
            num_groups = 258
        else:
            num_groups = 5000
    else:
        raise ValueError("--dataset should be one of ['artbench_post_impressionism']")

    if args.method in ["sparse_gd", "gd"]:
        training_config["checkpointing_steps"] = 100
        training_config["checkpoint_attn_procs"] = True

    # Set up coutput directories and files.
    removal_dist_name = "full"
    if args.removal_dist is not None:
        removal_dist_name = f"{args.removal_unit}_{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dist_name += f"_alpha={args.datamodel_alpha}"
    exp_name = os.path.join(
        args.dataset,
        args.method,
        removal_dist_name,
        f"seed{args.seed}",
    )
    command_outdir = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "commands", "train", exp_name
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "train", exp_name)
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
    elif args.removal_dist in ["loo", "aoi"]:
        assert num_groups % args.num_subsets_per_job == 0
        with open(command_file, "w") as handle:
            command = ""
            for i in range(num_groups):
                command += "accelerate launch"
                command += " --gpu_ids=0"
                command += " --mixed_precision={}".format("fp16")
                command += " text_to_image/train_text_to_image_lora.py"
                for key, val in training_config.items():
                    command += " " + format_config_arg(key, val)

                if args.removal_dist == "loo":
                    command += f" --loo_idx={i}"
                else:
                    command += f" --aoi_idx={i}"

                if (i + 1) % args.num_subsets_per_job == 0:
                    handle.write(command + "\n")
                    command = ""
                    num_jobs += 1
                else:
                    command += " ; "
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
