"""Make a file containing command line arguments for main.py to run experiments."""

import argparse
import os

import constants

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar"],
        default="mnist",
    )
    args = parser.parse_args()

    if args.dataset == "mnist":
        load = os.path.join(
            constants.OUTDIR,
            args.dataset,
            "retrain",
            "models",
            "full",
            "steps_00065660.pt",
        )
        excluded_class_list = [i for i in range(10)]
    else:
        raise NotImplementedError

    print(f"Writing the experiment parameter file for {args.dataset}")
    outfile = os.path.join("slurms", f"{args.dataset}_experiments.txt")
    with open(outfile, "w") as f:
        for method in ["ga", "gd"]:
            db = os.path.join(constants.OUTDIR, args.dataset, f"{method}_db.jsonl")
            for excluded_class in excluded_class_list:
                param = ""
                param += f"--load {load} "
                param += f"--dataset {args.dataset} "
                param += f"--excluded_class {excluded_class} "
                param += f"--method {method} "
                param += f"--db {db} "
                param += "--resume"
                f.write(param + "\n")
    print(f"File saved at {outfile}")
