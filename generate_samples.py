"""Generate samples for a given diffusion model."""
import argparse
import math
import os

import diffusers
import numpy as np
import torch
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    LDMPipeline,
    VQModel,
)
from diffusers.training_utils import EMAModel
from lightning.pytorch import seed_everything
from torchvision.utils import save_image

import constants
from ddpm_config import DDPMConfig
from diffusion.models import CNN
from utils import ImagenetteCaptioner, LabelTokenizer, create_dataset, get_max_steps


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar", "celeba", "imagenette"],
        default="mnist",
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="batch size for sample generation."
    )
    parser.add_argument(
        "--excluded_class",
        type=int,
        help="dataset class to exclude for class-wise data removal",
        default=None,
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
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--db", type=str, help="database file for storing results", default=None
    )
    parser.add_argument(
        "--exp_name", type=str, help="experiment name in the database", default=None
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of diffusion steps for generating images",
    )
    parser.add_argument(
        "--num_train_steps",
        type=int,
        default=1000,
        help="number of diffusion steps during training",
    )
    parser.add_argument(
        "--device", type=str, help="device of training", default="cuda:0"
    )

    return parser.parse_args()


def main(args):
    """Function to calculate model behavior"""
    seed_everything(args.seed)
    device = args.device

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
        classifier = CNN().to(device)
        classifier.load_state_dict(
            torch.load("eval/models/epochs=10_cnn_weights.pt", map_location=device)
        )
        classifier.eval()
    elif args.dataset == "imagenette":
        config = {**DDPMConfig.imagenette_config}
    else:
        raise ValueError(
            (
                f"dataset={args.dataset} is not one of "
                "['cifar', 'mnist', 'celeba', 'imagenette']"
            )
        )
    model_cls = getattr(diffusers, config["unet_config"]["_class_name"])

    removal_dir = "full"
    if args.excluded_class is not None:
        removal_dir = f"excluded_{args.excluded_class}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"

    model_loaddir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        "models",
        removal_dir,
    )
    sample_outdir = os.path.join(
        args.outdir, args.dataset, args.method, "samples", removal_dir
    )

    # Load pre-trained model.

    pretrained_steps = get_max_steps(model_loaddir)

    if pretrained_steps is not None:
        ckpt_path = os.path.join(model_loaddir, f"ckpt_steps_{pretrained_steps:0>8}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = model_cls(**config["unet_config"])
        model.load_state_dict(ckpt["unet"])

        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=model_cls,
            model_config=model.config,
        )

        print(f"Pre-trained model loaded from {args.load}")
        print(f"\tU-Net loaded from {ckpt_path}")
        print("\tEMA started from the loaded U-Net")
    else:
        raise ValueError(f"No pre-trained checkpoints found at {model_loaddir}")

    ema_model.to(device)

    if args.dataset == "imagenette":
        # The pipeline is of class LDMTextToImagePipeline.
        train_dataset = create_dataset(dataset_name=args.dataset, train=True)
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        pipeline.unet = model

        vqvae = pipeline.vqvae
        text_encoder = pipeline.bert
        tokenizer = pipeline.tokenizer
        captioner = ImagenetteCaptioner(train_dataset)
        label_tokenizer = LabelTokenizer(captioner=captioner, tokenizer=tokenizer)

        vqvae.requires_grad_(False)
        text_encoder.requires_grad_(False)

        vqvae = vqvae.to(device)
        text_encoder = text_encoder.to(device)
    elif args.dataset == "celeba":
        model_id = "CompVis/ldm-celebahq-256"
        vqvae = VQModel.from_pretrained(model_id, subfolder="vqvae")

        for param in vqvae.parameters():
            param.requires_grad = False

        pipeline = LDMPipeline(
            unet=model,
            vqvae=vqvae,
            scheduler=DDIMScheduler(**config["scheduler_config"]),
        ).to(device)
    else:
        pipeline = DDPMPipeline(
            unet=model, scheduler=DDPMScheduler(**config["scheduler_config"])
        ).to(device)

    pipeline_scheduler = pipeline.scheduler

    with torch.no_grad():
        if args.dataset == "imagenette":
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
        elif args.dataset == "celeba":
            pipeline = LDMPipeline(
                unet=model,
                vqvae=vqvae,
                scheduler=pipeline_scheduler,
            ).to(device)
        else:
            pipeline = DDIMPipeline(
                unet=model,
                scheduler=DDIMScheduler(num_train_timesteps=args.num_train_steps),
            )

        generated_imgs = []
        n_batches = args.n_samples // args.batch_size

        # TODO add text-to-image generation

        for index in range(n_batches):
            images = pipeline(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                output_type="numpy",
            ).images

            generated_imgs.append(images)
            print(f"generating samples {index}/{n_batches}")

        generated_imgs = np.concatenate(generated_imgs, axis=0)

    for i, images in enumerate(generated_imgs):
        save_image(
            torch.from_numpy(images).permute([2, 0, 1]),
            os.path.join(sample_outdir, f"sample_{i:04d}.png"),
        )

    print(f"Images saved to {sample_outdir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
