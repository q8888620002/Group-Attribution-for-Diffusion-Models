"""Run the pixel similarity baseline for data attribution."""
import argparse
import os

import clip
import numpy as np

# import open_clip
import pandas as pd
import torch
from torchvision import transforms
from tqdm import tqdm

from datasets import load_dataset
from diffusers import DiffusionPipeline

# from src.aesthetics import get_aesthetic_model
from src.constants import DATASET_DIR
from src.ddpm_config import PromptConfig
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run the pixel similarity baseline for data attribution."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/miniSD-diffusers",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Variant of the model files of the pretrained model identifier from "
            "huggingface.co/models, 'e.g.' fp16"
        ),
    )
    parser.add_argument(
        "--reference_lora_dir",
        type=str,
        default=None,
        help="directory for reference LoRA weights",
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["artbench"],
        default="artbench",
        help="Dataset to determine which prompts to use for image generation",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="number of images to generate for computing model behaviors",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="the resolution of generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducible image generation",
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
        default=4,
        help="number of subprocesses for the training data loader",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="batch size for the training data loader",
    )
    return parser.parse_args()


def load_pipeline(args):
    """Load diffusion model pipeline."""
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("cuda")
    return pipeline


def main(args):
    """Main function."""
    # Load models for computing scores.
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

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

    # Set up the training data loader.
    train_transforms = transforms.Compose(
        [
            transforms.Resize(
                args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
            ),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    image_column = dataset["train"].column_names[0]

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples

    def collate_fn(examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        return {"pixel_values": pixel_values}

    train_dataloader = torch.utils.data.DataLoader(
        dataset["train"].with_transform(preprocess_train),
        shuffle=False,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        num_workers=args.dataloader_num_workers,
        pin_memory=True,
    )

    # Set up the training data loader for CLIP embedding computation.
    def preprocess_clip_function(examples):
        images = [image.convert("RGB") for image in examples[image_column]]
        examples["pixel_values"] = [clip_preprocess(image) for image in images]
        return examples

    # Set up the prompt.
    prompt_dict = {"artbench": PromptConfig.artbench_config}
    prompt_dict = prompt_dict[args.dataset]
    prompt = prompt_dict[args.cls]

    # Load diffusion pipeline and the generator for reproducibility.
    reference_pipeline = load_pipeline(args)
    reference_pipeline.unet.load_attn_procs(
        args.reference_lora_dir, weight_name="pytorch_lora_weights.safetensors"
    )
    reference_generator = torch.Generator(device="cuda")
    reference_generator.manual_seed(args.seed)

    pixel_similarity_mat = []
    with torch.no_grad():
        for i in tqdm(range(args.num_images)):
            reference_img = reference_pipeline(
                prompt,
                num_inference_steps=100,
                generator=reference_generator,
                height=args.resolution,
                width=args.resolution,
            ).images[0]
            flat_reference_img = train_transforms(reference_img).to("cuda").flatten()
            flat_reference_img /= flat_reference_img.norm(dim=-1, keepdim=True)

            # Pixel cosine similarity.
            pixel_similarity_array = []
            for batch in train_dataloader:
                imgs = batch["pixel_values"].to("cuda")
                flat_imgs = imgs.flatten(start_dim=1)
                flat_imgs /= flat_imgs.norm(dim=-1, keepdim=True)
                pixel_similarity = torch.matmul(flat_imgs, flat_reference_img)
                pixel_similarity_array.append(pixel_similarity.cpu().numpy())
            pixel_similarity_mat.append(np.concatenate(pixel_similarity_array))

    pixel_similarity_mat = np.stack(pixel_similarity_mat, axis=1, dtype=np.float32)

    num_groups = len(group_indices_dict.keys())
    group_max_pixel_similarity = np.zeros(shape=(num_groups, args.num_images))
    group_avg_pixel_similarity = np.zeros(shape=(num_groups, args.num_images))

    for i, group_indices in group_indices_dict.items():
        group_pixel_similarity = pixel_similarity_mat[group_indices, :]
        group_max_pixel_similarity[i, :] = group_pixel_similarity.max(axis=0)
        group_avg_pixel_similarity[i, :] = group_pixel_similarity.mean(axis=0)

    output_dict = {
        "max_pixel_similarity": group_max_pixel_similarity,
        "avg_pixel_similarity": group_avg_pixel_similarity,
    }

    # Save results.
    output_dir = os.path.join(args.output_dir, "baselines")
    os.makedirs(output_dir, exist_ok=True)
    for name, output in output_dict.items():
        with open(os.path.join(output_dir, f"{args.group}_{name}.npy"), "wb") as handle:
            np.save(handle, output)

    # Rank groups and save ranked group indices.
    for name, output in output_dict.items():
        for i in range(args.num_images):
            file_prefix = f"generated_image_{i}_{args.group}_rank"
            rank = np.argsort(
                -output[:, i],  # Flip sign for descending rankings.
                kind="stable",
            )
            with open(
                os.path.join(output_dir, f"{file_prefix}_{name}.npy"), "wb"
            ) as handle:
                np.save(handle, rank)
        global_rank = np.argsort(-output.mean(axis=-1), kind="stable")
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
