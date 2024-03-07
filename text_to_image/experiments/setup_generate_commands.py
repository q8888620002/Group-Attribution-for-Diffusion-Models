"""Set up the commands that run generate_samples.py"""

import argparse
import os

from src.constants import LOGDIR, OUTDIR
from src.ddpm_config import TextToImageGenerationConfig
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
        "--lora_dir",
        type=str,
        default=None,
        help="directory containing LoRA weights to load",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50000,
        help="total number of images to generate",
    )
    parser.add_argument(
        "--num_images_per_job",
        type=int,
        default=500,
        help="number of images to generate for each SLURM job",
    )
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    assert args.num_images % args.num_images_per_job == 0
    num_jobs = int(args.num_images / args.num_images_per_job)

    if args.lora_dir is not None:
        outdir = args.lora_dir.replace("/models/", "/generated_images/")
        exp_name = os.path.join(
            args.dataset, args.lora_dir.split(args.dataset + "/")[-1]
        )
    else:
        outdir = os.path.join(OUTDIR, args.dataset, "pretrained", "generated_images")
        exp_name = os.path.join(args.dataset, "pretrained")

    if args.dataset == "artbench_post_impressionism":
        config = TextToImageGenerationConfig.artbench_post_impressionism_config
        config["outdir"] = outdir
        config["num_images"] = args.num_images_per_job
        if args.lora_dir is not None:
            config["lora_dir"] = args.lora_dir
    else:
        raise ValueError("--dataset should be one of ['artbench_post_impressionism']")

    # Set up coutput directories and files.
    command_outdir = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "commands", "generate", exp_name
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "generate", exp_name)
    os.makedirs(logdir, exist_ok=True)

    with open(command_file, "w") as handle:
        for seed in range(num_jobs):
            command = "python text_to_image/generate_samples.py"
            for key, val in config.items():
                command += " " + format_config_arg(key, val)
            command += f" --seed={seed}"
            handle.write(command + "\n")
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(os.getcwd(), "text_to_image", "experiments", "generate.job")
    array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    job_name = "generate-" + exp_name.replace("/", "-")
    if job_name.endswith("-"):
        job_name = job_name[:-1]
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
