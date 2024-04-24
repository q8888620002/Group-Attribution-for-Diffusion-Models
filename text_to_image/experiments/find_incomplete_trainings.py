"""Find incomplete training jobs set up by setup_train_commands.py"""

import argparse
import math
import os

from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_dir",
        type=str,
        help="parent directory containing all the subdirectories with model weights",
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
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    subdir_list = [
        entry
        for entry in os.listdir(args.model_dir)
        if os.path.isdir(os.path.join(args.model_dir, entry))
    ]
    incomplete_jobs = []
    for seed in range(args.num_removal_subsets):
        job_id = math.ceil((seed + 1) / args.num_subsets_per_job)
        weight_dir = [entry for entry in subdir_list if entry.endswith(f"seed={seed}")]
        assert len(weight_dir) <= 1
        if len(weight_dir) == 0:
            incomplete_jobs.append(job_id)
        else:
            weight_dir = weight_dir[0]
            weight_path = os.path.join(
                args.model_dir, weight_dir, "pytorch_lora_weights.safetensors"
            )
            if not os.path.isfile(weight_path):
                incomplete_jobs.append(job_id)
    incomplete_jobs_str = ",".join([f"{job}" for job in incomplete_jobs])
    num_incomplete_jobs = len(incomplete_jobs)
    print(f"Num of incomplete jobs: {num_incomplete_jobs}")
    print(f"Incomplete jobs: {incomplete_jobs_str}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
