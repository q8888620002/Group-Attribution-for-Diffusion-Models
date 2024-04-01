"""Set up the commands for each experiment that runs train_text_to_image_lora.py"""

import argparse
import os

from src.constants import DATASET_DIR, LOGDIR, OUTDIR
# from src.ddpm_config import LoraTrainingConfig
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

    args = parser.parse_args()
    return args


def main(args):
    """Main function."""
    if args.dataset == "celeba":
        pass
    else:
        raise ValueError("--dataset should be one of ['celeba']")

    # Set up coutput directories and files.
    exp_name = os.path.join(
        args.dataset,
        args.method,
        "full"
        if args.removal_dist is None
        else f"{args.removal_dist}",
    )
    command_outdir = os.path.join(
        os.getcwd(), "unconditional_generation", "celeba_experiments", "commands", "train", exp_name
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
    else:
        assert args.num_removal_subsets % args.num_subsets_per_job == 0
        with open(command_file, "w") as handle:
            command = ""
            for seed in range(args.num_removal_subsets):
                command += "accelerate launch"
                command += " --gpu_ids 0"
                command += " --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml"
                command += " unconditional_generation/main.py"
                command += f" --dataset {args.dataset}"
                command += f" --method {args.method}"
                command += f" --removal_dist {args.removal_dist}"
                command += f" --removal_seed {seed}"
                command += f" --mixed_precision fp16"
                command += f" --use_8bit_optimizer"
                command += f" --gradient_accumulation_steps 1"
                command += f" --precompute_stage reuse"
                
                if (seed + 1) % args.num_subsets_per_job == 0:
                    handle.write(command + "\n")
                    command = ""
                    num_jobs += 1
                else:
                    command += " ; "
    print(f"Commands saved to {command_file}")

    # Update the SLURM job submission file.
    job_file = os.path.join(os.getcwd(), "unconditional_generation", "celeba_experiments", "train.job")
    array = f"1-{num_jobs}" if num_jobs > 1 else "1"
    job_name = "train-" + exp_name.replace("/", "-")
    output = os.path.join(logdir, "run-%x-%A-%a.out")
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
