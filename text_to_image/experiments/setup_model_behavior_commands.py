"""Set up the commands for computing model behaviors."""

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
        "--opt_seed",
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
        "--method",
        type=str,
        choices=["retrain", "pretrained", "pruned_ft"],
        default="retrain",
        help="training or unlearning method",
    )
    parser.add_argument(
        "--removal_method",
        type=str,
        choices=[
            "full",
            "artist_shapley",
            "artist_uniform",
            "artist_datamodel_alpha=0.1",
            "artist_datamodel_alpha=0.25",
            "artist_datamodel_alpha=0.5",
            "artist_datamodel_alpha=0.75",
            "artist_loo",
            "artist_aoi",
        ],
        default=None,
        help="removal method for [retrain]",
    )
    parser.add_argument(
        "--num_removal_subsets",
        type=int,
        help="number of removal subsets to run",
        default=500,
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
    # Set up coutput directories and files.
    exp_name = os.path.join(
        args.dataset,
        args.method,
        "" if args.method == "pretrained" else args.removal_method,
        "" if args.method == "pretrained" else f"seed{args.opt_seed}",
    )
    if exp_name.endswith("/"):
        exp_name = exp_name[:-1]

    command_outdir = os.path.join(
        os.getcwd(),
        "text_to_image",
        "experiments",
        "commands",
        "model_behaviors",
        exp_name,
    )
    os.makedirs(command_outdir, exist_ok=True)
    command_file = os.path.join(command_outdir, "command.txt")

    logdir = os.path.join(LOGDIR, "model_behaviors", exp_name)
    os.makedirs(logdir, exist_ok=True)

    # Set up common arguments.
    if args.dataset == "artbench_post_impressionism":
        config = TextToImageModelBehaviorConfig.artbench_post_impressionism_config
        config["seed"] = args.model_behavior_seed
        config["num_images"] = args.num_images
        num_update_steps_per_epoch = 79
        if "artist" in args.removal_method:
            num_groups = 258
        else:
            num_groups = 5000
    else:
        raise ValueError

    db_dir = os.path.join(OUTDIR, f"seed{args.opt_seed}", args.dataset)
    os.makedirs(db_dir, exist_ok=True)
    ckpt_dir = os.path.join(
        LOGDIR, f"seed{args.opt_seed}", args.dataset, "model_behaviors"
    )
    os.makedirs(ckpt_dir, exist_ok=True)

    # Set up job commands.
    num_jobs = 0
    if args.method == "pretrained":
        command = "python text_to_image/compute_model_behaviors.py"
        for key, val in config.items():
            command += " " + format_config_arg(key, val)
        ckpt_path = os.path.join(ckpt_dir, "pretrained.pt")
        db = os.path.join(db_dir, "pretrained.jsonl")
        command += " --ckpt_path={}".format(ckpt_path)
        command += " --db={}".format(db)
        command += " --exp_name={}".format(exp_name)

        with open(command_file, "w") as handle:
            handle.write(command + "\n")
            num_jobs += 1

    elif args.method == "retrain" and args.removal_method == "full":
        command = "python text_to_image/compute_model_behaviors.py"
        for key, val in config.items():
            command += " " + format_config_arg(key, val)
        ckpt_path = os.path.join(ckpt_dir, "full.pt")
        db = os.path.join(db_dir, "full.jsonl")
        lora_dir = os.path.join(
            OUTDIR, f"seed{args.opt_seed}", args.dataset, "retrain", "models", "full"
        )
        command += " --ckpt_path={}".format(ckpt_path)
        command += " --db={}".format(db)
        command += " --exp_name={}".format(exp_name)
        command += " --lora_dir={}".format(lora_dir)

        with open(command_file, "w") as handle:
            handle.write(command + "\n")
            num_jobs += 1

    elif args.method == "pruned_ft" and args.removal_method == "full":
        ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        lora_steps_list = [
            num_update_steps_per_epoch * num_epochs
            for num_epochs in [10, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
        ]
        learning_rate_list = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3]
        assert (
            len(ratio_list) * len(lora_steps_list) * len(learning_rate_list)
        ) % args.num_executions_per_job == 0
        ckpt_dir = os.path.join(ckpt_dir, "pruned_ft")
        os.makedirs(ckpt_dir, exist_ok=True)

        num_executions = 0
        with open(command_file, "w") as handle:
            command = ""
            for ratio in ratio_list:
                for learning_rate in learning_rate_list:
                    for lora_steps in lora_steps_list:
                        command += "python text_to_image/compute_model_behaviors.py"
                        for key, val in config.items():
                            command += " " + format_config_arg(key, val)
                        ckpt_filename = f"ratio_{ratio}"
                        ckpt_filename += f"_lr_{learning_rate}"
                        ckpt_filename += f"_lora_steps_{lora_steps}.pt"
                        ckpt_path = os.path.join(ckpt_dir, ckpt_filename)
                        db = os.path.join(db_dir, "pruned_ft_ratio_lr_lora_steps.jsonl")
                        lora_dir = os.path.join(
                            OUTDIR,
                            f"seed{args.opt_seed}",
                            args.dataset,
                            f"pruned_ft_ratio={ratio}_lr={learning_rate}",
                            "models",
                            "full",
                        )
                        command += " --ckpt_path={}".format(ckpt_path)
                        command += " --db={}".format(db)
                        command += " --exp_name={}".format(
                            os.path.join(
                                exp_name,
                                f"ratio={ratio}",
                                f"lr={learning_rate}",
                                f"lora_steps={lora_steps}",
                            )
                        )
                        command += " --lora_dir={}".format(lora_dir)
                        command += " --lora_steps={}".format(lora_steps)
                        num_executions += 1

                        if num_executions % args.num_executions_per_job == 0:
                            handle.write(command + "\n")
                            command = ""
                            num_jobs += 1
                        else:
                            command += " ; "

    elif "loo" in args.removal_method or "aoi" in args.removal_method:
        removal_dist = "_".join(args.removal_method.split("_")[1:])
        ckpt_dir = os.path.join(ckpt_dir, args.removal_method)
        os.makedirs(ckpt_dir, exist_ok=True)

        group_idx_list = [i for i in range(num_groups)]
        db = os.path.join(db_dir, f"{args.method}_{args.removal_method}.jsonl")
        if os.path.exists(db):
            df = pd.read_json(db, lines=True)
            df["group_idx"] = (
                df["exp_name"].str.split("idx_", expand=True)[1].astype(int)
            )
            existing_group_idx_list = df["group_idx"].tolist()
            group_idx_list = set(group_idx_list) - set(existing_group_idx_list)
            group_idx_list = sorted(list(group_idx_list))
            if len(group_idx_list) == 0:
                print("Model behaviors have already been computed for all subsets!")
            elif 0 < len(group_idx_list) < num_groups:
                print(f"Only {len(group_idx_list)} subsets are missing model behaviors")
        assert len(group_idx_list) % args.num_executions_per_job == 0

        with open(command_file, "w") as handle:
            command = ""
            for i, group_idx in enumerate(group_idx_list):
                command += "python text_to_image/compute_model_behaviors.py"
                for key, val in config.items():
                    command += " " + format_config_arg(key, val)
                ckpt_path = os.path.join(ckpt_dir, f"{removal_dist}_idx_{group_idx}.pt")
                lora_dir = os.path.join(
                    OUTDIR,
                    f"seed{args.opt_seed}",
                    args.dataset,
                    args.method,
                    "models",
                    args.removal_method,
                    f"{removal_dist}_idx={group_idx}",
                )
                command += " --ckpt_path={}".format(ckpt_path)
                command += " --db={}".format(db)
                command += " --exp_name={}".format(
                    os.path.join(exp_name, f"{removal_dist}_idx_{group_idx}")
                )
                command += " --lora_dir={}".format(lora_dir)

                if (i + 1) % args.num_executions_per_job == 0:
                    handle.write(command + "\n")
                    command = ""
                    num_jobs += 1
                else:
                    command += " ; "

    elif args.removal_method is not None:
        removal_dist = "_".join(args.removal_method.split("_")[1:])
        ckpt_dir = os.path.join(ckpt_dir, args.removal_method)
        os.makedirs(ckpt_dir, exist_ok=True)

        removal_seed_list = [i for i in range(args.num_removal_subsets)]
        db = os.path.join(db_dir, f"{args.method}_{args.removal_method}.jsonl")
        if os.path.exists(db):
            df = pd.read_json(db, lines=True)
            df["removal_seed"] = (
                df["exp_name"].str.split("seed_", expand=True)[1].astype(int)
            )
            existing_removal_seed_list = df["removal_seed"].tolist()
            removal_seed_list = set(removal_seed_list) - set(existing_removal_seed_list)
            removal_seed_list = sorted(list(removal_seed_list))
            if len(removal_seed_list) == 0:
                print("Model behaviors have already been computed for all subsets!")
            elif 0 < len(removal_seed_list) < args.num_removal_subsets:
                print(
                    f"Only {len(removal_seed_list)} subsets are missing model behaviors"
                )
        assert len(removal_seed_list) % args.num_executions_per_job == 0

        with open(command_file, "w") as handle:
            command = ""
            for i, removal_seed in enumerate(removal_seed_list):
                command += "python text_to_image/compute_model_behaviors.py"
                for key, val in config.items():
                    command += " " + format_config_arg(key, val)
                ckpt_path = os.path.join(
                    ckpt_dir, f"{removal_dist}_seed_{removal_seed}.pt"
                )
                lora_dir = os.path.join(
                    OUTDIR,
                    f"seed{args.opt_seed}",
                    args.dataset,
                    args.method,
                    "models",
                    args.removal_method,
                    f"{removal_dist}_seed={removal_seed}",
                )
                command += " --ckpt_path={}".format(ckpt_path)
                command += " --db={}".format(db)
                command += " --exp_name={}".format(
                    os.path.join(exp_name, f"{removal_dist}_seed_{removal_seed}")
                )
                command += " --lora_dir={}".format(lora_dir)

                if (i + 1) % args.num_executions_per_job == 0:
                    handle.write(command + "\n")
                    command = ""
                    num_jobs += 1
                else:
                    command += " ; "
    else:
        raise ValueError
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(
        os.getcwd(), "text_to_image", "experiments", "model_behaviors.job"
    )
    array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    job_name = "model-behavior-" + exp_name.replace("/", "-")
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
