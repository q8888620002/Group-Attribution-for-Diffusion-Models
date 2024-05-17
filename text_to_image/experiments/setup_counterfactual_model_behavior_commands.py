"""Set up the commands for counterfactual model retrainings."""

import argparse
import os

import pandas as pd

from src.constants import LOGDIR, OUTDIR
from src.ddpm_config import TextToImageModelBehaviorConfig
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
        "--model_behavior_seed",
        type=int,
        help="random seed for computing model behaviors",
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
        "--rank_method",
        type=str,
        choices=["max_pixel_similarity", "avg_aesthetic_score", "dtrak"],
        default=None,
        required=True,
    )
    parser.add_argument(
        "--num_executions_per_job",
        type=int,
        help="number of script executions to run for each SLURM job",
        default=20,
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="number of images to generate for computing model behaviors",
    )
    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    if args.dataset == "artbench_post_impressionism":
        config = TextToImageModelBehaviorConfig.artbench_post_impressionism_config
        config["seed"] = args.model_behavior_seed
        config["num_images"] = args.num_images
    else:
        raise ValueError
    rank_method = f"{args.removal_unit}_rank_{args.rank_method}"

    db_dir = os.path.join(OUTDIR, args.dataset, "counterfactual")
    os.makedirs(db_dir, exist_ok=True)
    db = os.path.join(db_dir, f"{rank_method}.jsonl")
    config["db"] = db
    df = pd.read_json(db, lines=True) if os.path.exists(db) else None

    ckpt_dir = os.path.join(
        LOGDIR, args.dataset, "counterfactual_model_behaviors", rank_method
    )
    os.makedirs(ckpt_dir, exist_ok=True)
    image_parent_dir = os.path.join(OUTDIR, args.dataset, "counterfactual", "images")
    os.makedirs(image_parent_dir, exist_ok=True)

    command_outdir = os.path.join(
        os.getcwd(),
        "text_to_image",
        "experiments",
        "commands",
        "counterfactual_model_behavior",
        rank_method,
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    num_executions, num_jobs = 0, 0
    with open(command_file, "w") as handle:
        command = ""
        for removal_rank_proportion in [0.05, 0.1, 0.25, 0.4]:
            for opt_seed in [42, 43, 44, 45, 46]:
                exp_name = os.path.join(
                    rank_method,
                    f"top_{removal_rank_proportion}",
                    f"opt_seed={opt_seed}",
                )
                record_exists = False
                if df is not None:
                    record_exists = (df["exp_name"] == exp_name).any()

                if not record_exists:
                    command += "python text_to_image/compute_model_behaviors.py"
                    for key, val in config.items():
                        command += " " + format_config_arg(key, val)
                    ckpt_path = os.path.join(
                        ckpt_dir,
                        f"top_{removal_rank_proportion}_opt_seed={opt_seed}.pt",
                    )
                    command += " --ckpt_path={}".format(ckpt_path)
                    command += " --image_dir={}".format(
                        os.path.join(image_parent_dir, exp_name)
                    )
                    lora_dir = os.path.join(
                        OUTDIR,
                        f"seed{opt_seed}",
                        args.dataset,
                        "retrain",
                        "models",
                        f"counterfactual_top_{removal_rank_proportion}",
                        f"all_generated_images_{rank_method}",
                    )
                    command += " --lora_dir={}".format(lora_dir)
                    command += " --exp_name={}".format(exp_name)
                    num_executions += 1

                    if num_executions % args.num_executions_per_job == 0:
                        handle.write(command + "\n")
                        command = ""
                        num_jobs += 1
                    else:
                        command += " ; "

    if num_executions == 0:
        print("All model behaviors have been computed!")
    else:
        print(f"Commands saved to {command_file}")
        print(f"{num_executions} remaining executions to finish")
        assert num_executions % args.num_executions_per_job == 0

        # Update the SLURM job submission file.
        job_file = os.path.join(
            os.getcwd(),
            "text_to_image",
            "experiments",
            "counterfactual_model_behaviors.job",
        )
        array = f"1-{num_jobs}" if num_jobs > 1 else "1"
        job_name = f"counterfactual-behavior-{rank_method}"

        logdir = os.path.join(LOGDIR, "counterfactual_model_behaviors", rank_method)
        os.makedirs(logdir, exist_ok=True)
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
