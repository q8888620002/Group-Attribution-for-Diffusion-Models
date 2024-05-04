"""Set up the commands for each pruning experiment."""

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
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    if args.dataset == "artbench_post_impressionism":
        num_update_steps_per_epoch = 79
        training_config = LoraTrainingConfig.artbench_post_impressionism_config
        train_data_dir = os.path.join(
            DATASET_DIR, "artbench-10-imagefolder-split", "train"
        )
        training_config["train_data_dir"] = train_data_dir
        training_config["output_dir"] = os.path.join(OUTDIR, f"seed{args.seed}")
        training_config["seed"] = args.seed
        training_config["method"] = "pruned_ft"
        training_config["checkpointing_steps"] = num_update_steps_per_epoch * 10

        trained_lora_dir = os.path.join(
            training_config["output_dir"],
            args.dataset,
            "retrain",
            "models",
            "full",
        )
    else:
        raise ValueError("--dataset should be one of ['artbench_post_impressionism']")

    # Set up output directories and files.
    exp_name = os.path.join(args.dataset, "pruned_ft", "full", f"seed{args.seed}")
    command_outdir = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "commands", "prune", exp_name
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "prune", exp_name)
    os.makedirs(logdir, exist_ok=True)

    num_jobs = 0
    with open(command_file, "w") as handle:
        for i in range(1, 10):
            pruning_ratio = i / 10
            for learning_rate in [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]:
                model_outdir = os.path.join(
                    training_config["output_dir"],
                    args.dataset,
                    f"pruned_ft_ratio={pruning_ratio}_lr={learning_rate}",
                    "models",
                    "full",
                )
                weight_file = os.path.join(
                    model_outdir, "pytorch_lora_weights.safetensors"
                )
                if not os.path.exists(weight_file):
                    command = "python text_to_image/prune_lora.py"
                    command += " --lora_dir={}".format(trained_lora_dir)
                    command += " --pruning_ratio={}".format(pruning_ratio)
                    command += " ; "
                    command += "accelerate launch"
                    command += " --gpu_ids=0"
                    command += " --mixed_precision={}".format("fp16")
                    command += " text_to_image/train_text_to_image_lora.py"
                    for key, val in training_config.items():
                        if key not in ["learning_rate"]:
                            command += " " + format_config_arg(key, val)
                    command += " --learning_rate={}".format(learning_rate)
                    command += " --checkpoint_attn_procs"
                    command += " --pruning_ratio={}".format(pruning_ratio)

                    handle.write(command + "\n")
                    num_jobs += 1
    if num_jobs > 0:
        print(f"Commands saved to {command_file}")

        # Update the SLURM job submission file.
        job_file = os.path.join(
            os.getcwd(), "text_to_image", "experiments", "prune.job"
        )
        array = f"1-{num_jobs}" if num_jobs > 1 else "1"
        job_name = "prune-" + exp_name.replace("/", "-")
        output = os.path.join(logdir, "run-%A-%a.out")
        update_job_file(
            job_file=job_file,
            job_name=job_name,
            output=output,
            array=array,
            command_file=command_file,
        )
    else:
        print("All pruning jobs are completed!")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
