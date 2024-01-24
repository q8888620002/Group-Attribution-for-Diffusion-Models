"""Resize Imagenette files and change extensions to be compatible with torch_fid."""
import argparse
import os

from PIL import Image
from torchvision import transforms

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="aggregate subdirectory image files")
    parser.add_argument(
        "--parent_dir",
        help="parent directory containing subdirectories of Imagenette files",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--outdir",
        help="output parent directory to store subdirectories of resized image files",
        type=str,
        required=True,
    )
    args = parser.parse_args()

    subdir_list = [entry for entry in os.listdir(args.parent_dir)]
    resize = transforms.Resize((256, 256))

    for subdir in subdir_list:
        src_dir = os.path.join(args.parent_dir, subdir)
        dst_dir = os.path.join(args.outdir, subdir)
        os.makedirs(dst_dir, exist_ok=True)
        print(f"Copying image files from {src_dir} to {dst_dir}...")

        for src_img_path in os.listdir(src_dir):
            dst_img_path = src_img_path.replace("JPEG", "jpeg")
            with Image.open(os.path.join(src_dir, src_img_path)) as img:
                img = resize(img)
                img.save(os.path.join(dst_dir, dst_img_path))

    print("Done!")
