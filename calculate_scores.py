import argparse
import os

from utils import *
from pytorch_fid import fid_score
import constants

def parse_args():
    parser = argparse.ArgumentParser(description="Calculating model behavior")

    parser.add_argument(
        "--sample1_dir",
        type=str,
        help="directory path for samples generated by pre-trained model 1.",
        default=None,
    )
    parser.add_argument(
        "--sample2_dir",
        type=str,
        help="directory path for samples generated by pre-trained model 2.",
        default=None,
    )
    parser.add_argument(
        "--outdir", 
        type=str, 
        help="output parent directory", 
        default=constants.OUTDIR
    )
    parser.add_argument(
        "--model_behavior",
        type=str,
        choices=["fid", "diversity", "clip", "pixel_dist"],
        help="model behavior",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba", "imagenette"],
        default="mnist",
    )
    parser.add_argument(
        "--no_clip",
        action="store_true",
        help="set to normal sampling method without clip x_0 which could yield unstable samples",
    )
    parser.add_argument(
        "--device", 
        type=str, 
        help="device to train"
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        default=None,
    )
    parser.add_argument(
        "--datamodel_alpha",
        type=float,
        help="proportion of full dataset to keep in the datamodel distribution",
        default=0.5,
    )
    parser.add_argument(
        "--removal_seed",
        type=int,
        help="random seed for sampling from the removal distribution",
        default=0,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "gd", "ga", "esd"],
        required=True,
    )

    args = parser.parse_args()

    return args

def main(args):
    """Main script for calculating model behavior"""

    results = {
        "fid": [], 
        "clip": [], 
        "diversity": [],
        "pixel_dist": []
    }

    paths = []

    paths.append(args.sample1_dir)
    paths.append(args.sample2_dir)

    removal_dir = "full"
    if args.excluded_class is not None:
        removal_dir = f"excluded_{args.excluded_class}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"
        
    output_dir = os.path.join(
        args.outdir, 
        args.dataset, 
        args.method, 
        removal_dir,
        "model_behavior"
    )

    if args.model_behavior == "fid":
        fid_value = fid_score.calculate_fid_given_paths(
            paths,
            batch_size=2048,
            device=args.device,
            dims=2048,
            num_workers=0
        )
        results["fid"] = fid_value

    elif args.model_behavior == "clip":
        raise NotImplementedError
    
    elif args.model_behavior == "diversity":
        raise NotImplementedError
    
    elif args.model_behavior == "pixel_dist":
        raise NotImplementedError

    else:
        raise NotImplementedError

    np.save(
        results, 
        os.path.join(output_dir,"results.npy")
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
