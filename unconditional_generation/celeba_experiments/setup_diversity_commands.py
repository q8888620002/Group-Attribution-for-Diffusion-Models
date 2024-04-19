"""Set up the commands that run generate_samples.py"""

import argparse
import os

from src import constants
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
        choices=["celeba"],
        default="celeba",
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        default=None,
    )
    parser.add_argument(
        "--num_seeds",
        type=int,
        default=8,
        help="Number of seeds to generate scripts for.",
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="proportion of full dataset to keep in the datamodel distribution",
        default=0.5,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=constants.METHOD,
    )
    parser.add_argument("--exp_name", type=str, help="experiment name")
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""

    # Set up coutput directories and files.
    command_outdir = os.path.join(
        os.getcwd(),
        "unconditional_generation",
        "celeba_experiments",
        "commands",
        "diversity",
        args.exp_name,
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "diversity", args.exp_name)
    os.makedirs(logdir, exist_ok=True)

    with open(command_file, "w") as handle:
        for seed in range(args.num_seeds):
            command = (
                "python unconditional_generation/calculate_global_scores_diversity.py"
            )

            command += f" --dataset {args.dataset}"
            command += f" --removal_dist {args.removal_dist}"
            command += f" --removal_seed {seed}"
            command += " --trained_steps 20001"
            command += " --use_ema"
            command += f" --method {args.method}"
            command += " --num_inference_steps 100"
            command += f" --exp_name {args.exp_name}"
            command += " --seed 42"
            command += f" --db {os.path.join(constants.OUTDIR, args.dataset, args.exp_name)+'.jsonl'}"
            command += " --n_samples 1000"
            handle.write(command + "\n")
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(
        os.getcwd(), "unconditional_generation", "celeba_experiments", "diversity.job"
    )
    num_jobs = args.num_seeds
    array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    job_name = "diversity-" + args.exp_name.replace("/", "-")
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
