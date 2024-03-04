"""Aggregate subdirectories of files into a single output directory."""
import argparse
import os
import shutil

from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aggregate subdirectory image files")
    parser.add_argument(
        "--parent_dir",
        help="parent directory containing subdirectories",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help="output directory to store the aggregated image files",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    subdir_list = [
        entry
        for entry in os.listdir(args.parent_dir)
        if os.path.isdir(os.path.join(args.parent_dir, entry))
    ]

    for subdir in subdir_list:
        src_dir = os.path.join(args.parent_dir, subdir)
        print(f"Copying image files from {src_dir} to {args.outdir}...")

        for src_img_path in tqdm(os.listdir(src_dir)):
            dst_img_path = (
                subdir + "_" + src_img_path
            )  # Append subdir to avoid duplicates.
            shutil.copy(
                os.path.join(src_dir, src_img_path),
                os.path.join(args.outdir, dst_img_path),
            )
    print("Done!")
