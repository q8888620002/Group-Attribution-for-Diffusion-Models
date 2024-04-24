"""Generate images based on text prompts for pre-trained models."""
import argparse
import os

import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image
from tqdm import tqdm

from diffusers import DiffusionPipeline
from src.ddpm_config import PromptConfig
from src.utils import print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Text to image generation.")
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
        "--lora_dir",
        type=str,
        default=None,
        help="directory containing LoRA weights to load",
    )
    parser.add_argument(
        "--lora_steps",
        type=int,
        default=None,
        help="number of LoRA fine-tuning steps",
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
        help="number of images to generate per prompt",
    )
    parser.add_argument(
        "--ckpt_freq",
        type=int,
        default=25,
        help="number of saved images before saving a checkpoint",
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
        default=None,
        help="generate images for this class",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=None,
        help="output directory to save all the generated images in individual files",
        required=True,
    )
    parser.add_argument(
        "--sep_outdir",
        action="store_true",
        help="whether to store images geneated by each prompt in a separate directory",
    )
    return parser.parse_args()


def main(args):
    """Main function."""
    prompt_dict = {"artbench": PromptConfig.artbench_config}
    prompt_dict = prompt_dict[args.dataset]
    if args.cls is not None:
        prompt_dict = {args.cls: prompt_dict[args.cls]}
    label_list = sorted(list(prompt_dict.keys()))

    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("cuda")

    if args.lora_dir is not None:
        weight_name = "pytorch_lora_weights"
        if args.lora_steps is not None:
            weight_name += f"_{args.lora_steps}"
        weight_name += ".safetensors"
        pipeline.unet.load_attn_procs(args.lora_dir, weight_name=weight_name)
        weight_path = os.path.join(args.lora_dir, weight_name)
        print(f"LoRA weights loaded from {weight_path}")

    ckpt_file = f"ckpt_seed={args.seed}"
    if args.cls is not None:
        ckpt_file = f"{args.cls}_" + ckpt_file
    ckpt_file = os.path.join(args.outdir, ckpt_file)

    generator = torch.Generator(device="cuda")
    if os.path.exists(ckpt_file):
        ckpt = torch.load(ckpt_file)
        generator.set_state(ckpt["rng_state"])
        starting_idx = ckpt["starting_idx"]
        completed_label_set = ckpt["completed_label_set"]
        print(f"Checkpoint loaded from {ckpt_file}")
    else:
        generator = generator.manual_seed(args.seed)
        starting_idx = 0
        completed_label_set = set()
    label_list = [label for label in label_list if label not in completed_label_set]

    for label in label_list:
        prompt = prompt_dict[label]
        if args.sep_outdir:
            label_outdir = os.path.join(args.outdir, label)
        else:
            label_outdir = args.outdir
        os.makedirs(label_outdir, exist_ok=True)

        progress_bar = tqdm(initial=starting_idx, total=args.num_images)
        print(f"Generating images for {label}: {prompt}")
        for i in range(starting_idx, args.num_images):
            img = pipeline(
                prompt,
                num_inference_steps=100,
                generator=generator,
                height=args.resolution,
                width=args.resolution,
            ).images[0]
            outfile = os.path.join(
                label_outdir, f"{label}_seed={args.seed}_sample_{i}.jpg"
            )
            if os.path.exists(outfile):
                os.remove(outfile)
            save_image(to_tensor(img), outfile)
            progress_bar.update(1)
            if (i + 1) % args.ckpt_freq == 0:
                ckpt = {
                    "rng_state": generator.get_state(),
                    "completed_label_set": completed_label_set,
                    "starting_idx": i + 1,
                }
                torch.save(ckpt, ckpt_file)
                print(f"Checkpoint saved to {ckpt_file}")

        print(f"Images saved to {label_outdir}")

        # Also save the checkpoint after generating all images for a label.
        completed_label_set.add(label)
        starting_idx = 0
        ckpt = {
            "rng_state": generator.get_state(),
            "completed_label_set": completed_label_set,
            "starting_idx": starting_idx,
        }
        torch.save(ckpt, ckpt_file)
        print(f"Checkpoint saved to {ckpt_file}")
    os.remove(ckpt_file)  # Remove checkpoint file once completion is reached.
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
