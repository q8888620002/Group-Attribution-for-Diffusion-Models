"""Utilities for duffusion pipeline"""
import os

import torch
from diffusers import DDIMPipeline, DDIMScheduler, DiffusionPipeline
from diffusers.training_utils import EMAModel
from torchvision import transforms
from tqdm import tqdm

from utils import ImagenetteCaptioner, create_dataset, get_max_steps


def load_ckpt_model(args, model_cls, model_strc, model_loaddir):
    """Load model parameters from the latest checkpoint in a directory."""

    trained_steps = (
        args.trained_steps
        if args.trained_steps is not None
        else get_max_steps(model_loaddir)
    )

    if trained_steps is not None:
        ckpt_path = os.path.join(model_loaddir, f"ckpt_steps_{trained_steps:0>8}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if args.method != "retrain":
            # Load pruned model
            pruned_model_path = os.path.join(
                args.outdir,
                args.dataset,
                "pruned",
                "models",
                (
                    f"pruner={args.pruner}"
                    + f"_pruning_ratio={args.pruning_ratio}"
                    + f"_threshold={args.thr}"
                ),
                f"ckpt_steps_{0:0>8}.pt",
            )
            pruned_model_ckpt = torch.load(pruned_model_path, map_location="cpu")
            model = pruned_model_ckpt["unet"]
        else:
            model = model_strc

        remaining_idx = ckpt["remaining_idx"].numpy().tolist()
        removed_idx = ckpt["removed_idx"].numpy().tolist()

        model.load_state_dict(ckpt["unet"])

        model_str = "U-Net"

        if args.use_ema:
            ema_model = EMAModel(
                model.parameters(),
                model_cls=model_cls,
                model_config=model.config,
            )
            ema_model.load_state_dict(ckpt["unet_ema"])
            ema_model.copy_to(model.parameters())
            model_str = "EMA"

        print(f"Trained model loaded from {model_loaddir}")
        print(f"\t{model_str} loaded from {ckpt_path}")
    else:
        raise ValueError(f"No trained checkpoints found at {model_loaddir}")

    return model, remaining_idx, removed_idx


def build_pipeline(args, model):
    """Build the diffusion pipeline for the sepcific dataset and U-Net model."""
    # Get the diffusion model pipeline for inference.
    if args.dataset == "imagenette":
        # The pipeline is of class LDMTextToImagePipeline.
        train_dataset = create_dataset(dataset_name=args.dataset, train=True)
        captioner = ImagenetteCaptioner(train_dataset)

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/ldm-text2im-large-256"
        ).to(args.device)
        pipeline.unet = model.to(args.device)
    elif args.dataset == "celeba":
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").to(
            args.device
        )
        pipeline.unet = model.to(args.device)
    else:
        pipeline = DDIMPipeline(unet=model, scheduler=DDIMScheduler()).to(args.device)

    return pipeline


def generate_images(args, pipeline):

    results = []

    batch_size_list = [args.batch_size] * (args.n_samples // args.batch_size)
    remaining_sample_size = args.n_samples % args.batch_size

    if remaining_sample_size > 0:
        batch_size_list.append(remaining_sample_size)

    if args.dataset != "imagenette":
        # For unconditional diffusion models.
        with torch.no_grad():
            counter = 0
            for batch_size in tqdm(batch_size_list):
                noise_generator = torch.Generator(device=args.device).manual_seed(
                    counter
                )
                images = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                    generator=noise_generator,
                ).images

                counter += 1
                for image in images:
                    image = torch.from_numpy(image).permute([2, 0, 1])
                    # Align with image saving process in generate_samples.py
                    permuted_image = (
                        image.mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    results.append(transforms.ToTensor()(permuted_image))

    return torch.stack(results).float()
