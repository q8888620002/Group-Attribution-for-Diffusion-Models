"""Compute aesthetic score for each training image as a simple global baseline."""
import argparse
import os
import pickle

import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

from datasets import load_dataset
from src.aesthetics import get_aesthetic_model
from src.constants import DATASET_DIR
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute aesthetic score for each training image."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["artbench"],
        default="artbench",
        help="Dataset to determine which prompts to use for image generation",
    )
    parser.add_argument(
        "--cls",
        type=str,
        default="post_impressionism",
        help="generate images for this class",
    )
    parser.add_argument(
        "--group",
        type=str,
        default="artist",
        choices=["artist", "filename"],
        help="unit for how to group images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="output parent directory where the attribution scores will be saved",
        required=True,
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=2,
        help="number of subprocesses for the training data loader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="batch size for the training data loader",
    )
    return parser.parse_args()


def main(args):
    """Main function."""

    # Load the aesthetic model and the corresponding CLIP for aesthetic scoring.
    aesthetic_model = get_aesthetic_model(clip_model="vit_l_14")
    aesthetic_model = aesthetic_model.to("cuda")
    (
        open_clip_model,
        _,
        open_clip_preprocess,
    ) = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    open_clip_model = open_clip_model.to("cuda")

    # Load the training data.
    train_data_dir_dict = {
        "artbench": os.path.join(DATASET_DIR, "artbench-10-imagefolder-split/train")
    }
    train_data_dir = train_data_dir_dict[args.dataset]
    cls_key_dict = {"artbench": "style"}
    cls_key = cls_key_dict[args.dataset]

    data_files = {}
    data_files["train"] = os.path.join(train_data_dir, "**")
    dataset = load_dataset("imagefolder", data_files=data_files)
    cls_idx = np.where(np.array(dataset["train"][cls_key]) == args.cls)[0]
    dataset["train"] = dataset["train"].select(cls_idx)  # Subset to the class data.
    if args.dataset == "artbench":
        assert dataset["train"].num_rows == 5000

    # Find the image indices for each group.
    group_file = os.path.join(train_data_dir, f"{args.cls}_{args.group}s.csv")
    group_df = pd.read_csv(group_file)
    group_indices_dict = {}
    for i in group_df.index:
        group_name = group_df.iloc[i].item()
        group_indices = np.where(np.array(dataset["train"][args.group]) == group_name)[
            0
        ]
        group_indices_dict[i] = group_indices

    # Set up the dataloader.
    image_column = dataset["train"].column_names[0]

    def preprocess_clip_function(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [open_clip_preprocess(image) for image in images]
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    dataloader = torch.utils.data.DataLoader(
        dataset["train"].with_transform(preprocess_clip_function),
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    all_aesthetic_scores = []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            imgs = batch["pixel_values"].to("cuda")
            clip_embeddings = open_clip_model.encode_image(imgs)
            clip_embeddings /= clip_embeddings.norm(dim=-1, keepdim=True)
            aesthetic_scores = aesthetic_model(clip_embeddings).flatten()
            all_aesthetic_scores.append(aesthetic_scores.cpu().numpy())
    all_aesthetic_scores = np.concatenate(all_aesthetic_scores, dtype=np.float32)

    num_groups = len(group_indices_dict.keys())
    group_max_aesthetic_score = np.zeros(shape=(num_groups, 1))
    group_avg_aesthetic_score = np.zeros(shape=(num_groups, 1))

    for i, group_indices in group_indices_dict.items():
        group_aesthetic_scores = all_aesthetic_scores[group_indices]
        group_max_aesthetic_score[i, 0] = group_aesthetic_scores.max()
        group_avg_aesthetic_score[i, 0] = group_aesthetic_scores.mean()

    output_dict = {
        "max_aesthetic_score": group_max_aesthetic_score,
        "avg_aesthetic_score": group_avg_aesthetic_score,
    }

    # Save results.
    output_dir = os.path.join(args.output_dir, "baselines")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "image_aesthetic_score.npy"), "wb") as handle:
        np.save(handle, all_aesthetic_scores)

    with open(
        os.path.join(output_dir, f"aesthetic_score_{args.group}_indices_dict.pkl"), "wb"
    ) as handle:
        pickle.dump(group_indices_dict, handle)

    for name, output in output_dict.items():
        with open(os.path.join(output_dir, f"{args.group}_{name}.npy"), "wb") as handle:
            np.save(handle, output)

    # Rank groups and save ranked group indices.
    for name, output in output_dict.items():
        global_rank = np.argsort(-output.flatten(), kind="stable")
        with open(
            os.path.join(
                output_dir, f"all_generated_images_{args.group}_rank_{name}.npy"
            ),
            "wb",
        ) as handle:
            np.save(handle, global_rank)


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
