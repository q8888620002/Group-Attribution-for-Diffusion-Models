"""Save a pre-trained model and generate images from it."""
import argparse
import math
import os

import numpy as np
import torch
from diffusers import DiffusionPipeline
from lightning.pytorch import seed_everything
from torchvision.utils import save_image

import constants
from ddpm_config import DDPMConfig
from utils import ImagenetteCaptioner, create_dataset, print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and save a large-scale diffusion model"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset that a pre-trained model is loaded for",
        choices=["imagenette"],
        default="imagenette",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for generating images from the pre-trained model",
        default=42,
    )
    parser.add_argument(
        "--device", type=str, help="device for image generation", default="cuda:0"
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    return parser.parse_args()


def main(args):
    """Main function for saving a pre-trained model and generating images from it."""
    seed_everything(args.seed)
    model_outdir = os.path.join(
        args.outdir,
        args.dataset,
        "pretrained",
        "models",
    )
    os.makedirs(model_outdir, exist_ok=True)
    sample_outdir = os.path.join(
        args.outdir,
        args.dataset,
        "pretrained",
        "samples",
    )
    os.makedirs(sample_outdir, exist_ok=True)

    if args.dataset == "imagenette":
        model_id = "CompVis/ldm-text2im-large-256"
        pipeline = DiffusionPipeline.from_pretrained(model_id)
        unet = pipeline.unet
        steps = 0
        unet_path = os.path.join(model_outdir, f"unet_steps_{steps:0>8}.pt")
        torch.save(unet, unet_path)
        print(f"Pre-trained U-Net from {model_id} saved at {unet_path}")

        pipeline = pipeline.to(args.device)
        config = {**DDPMConfig.imagenette_config}
        train_dataset = create_dataset(dataset_name=args.dataset, train=True)
        captioner = ImagenetteCaptioner(train_dataset)

        with torch.no_grad():
            samples = []
            n_samples_per_cls = math.ceil(config["n_samples"] / captioner.num_classes)
            classes = [idx for idx in range(captioner.num_classes)]
            for _ in range(n_samples_per_cls):
                samples.append(
                    pipeline(
                        prompt=captioner(classes),
                        num_inference_steps=50,
                        eta=0.3,
                        guidance_scale=6,
                        output_type="numpy",
                    ).images
                )
        samples = np.concatenate(samples)
        samples = torch.from_numpy(samples).permute([0, 3, 1, 2])
        if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
            samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]

        sample_path = os.path.join(sample_outdir, f"steps_{steps:0>8}.png")
        save_image(samples, sample_path, nrow=captioner.num_classes)
        print(f"Images generated by {model_id} saved at {sample_path}")
    else:
        raise ValueError(f"dataset={args.dataset} is not one of ['imagenette']")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
    print("Done!")
