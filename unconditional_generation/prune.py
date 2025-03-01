"""Pruning diffusion models"""

import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch_pruning as tp
from accelerate import Accelerator
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

import diffusers
import src.constants as constants
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    LDMPipeline,
)
from diffusers.models.attention import Attention
from diffusers.models.resnet import Downsample2D, Upsample2D
from src.datasets import create_dataset
from src.ddpm_config import DDPMConfig
from src.diffusion_utils import ImagenetteCaptioner, LabelTokenizer
from src.utils import get_max_steps


def parse_args():
    """Parsing arguments"""
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--load", type=str, help="path for loading pre-trained model", default=None
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=constants.DATASET,
        default=None,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )

    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )

    parser.add_argument(
        "--pruning_ratio",
        type=float,
        help="ratio for remaining parameters.",
        default=0.3,
    )

    parser.add_argument(
        "--pruner",
        type=str,
        default="magnitude",
        choices=["taylor", "random", "magnitude", "reinit", "diff-pruning"],
    )
    parser.add_argument(
        "--thr", type=float, default=0.05, help="threshold for diff-pruning"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="The dropout rate for fine-tuning."
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
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before a backward/update pass.",
    )
    parser.add_argument(
        "--ema_inv_gamma",
        type=float,
        default=1.0,
        help="inverse gamma value for EMA decay",
    )
    parser.add_argument(
        "--ema_power",
        type=float,
        default=3 / 4,
        help="power value for EMA decay",
    )
    parser.add_argument(
        "--ema_max_decay",
        type=float,
        default=0.9999,
        help="maximum decay magnitude EMA",
    )
    return parser.parse_args()


def print_args(args):
    """Print script name and args."""
    print(f"Running {sys.argv[0]} with arguments")
    for arg in vars(args):
        print(f"\t{arg}={getattr(args, arg)}")


def run_inference(
    accelerator,
    model,
    config,
    args,
    vqvae,
    captioner,
    pipeline,
    pipeline_scheduler,
):
    """Wrapper function for inference. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model).eval()

    with torch.no_grad():
        if args.dataset == "imagenette":
            samples = []
            n_samples_per_cls = math.ceil(config["n_samples"] / captioner.num_classes)
            classes = [idx for idx in range(captioner.num_classes)]
            for _ in range(n_samples_per_cls):
                samples.append(
                    pipeline(
                        prompt=captioner(classes),
                        num_inference_steps=args.num_inference_steps,
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
            ).to(accelerator.device)
            samples = pipeline(
                batch_size=4,  # config["n_samples"],
                num_inference_steps=args.num_inference_steps,
                output_type="numpy",
            ).images
        else:
            pipeline = DDIMPipeline(
                unet=model,
                scheduler=DDIMScheduler(num_train_timesteps=args.num_train_steps),
            )
            samples = pipeline(
                batch_size=config["n_samples"],
                num_inference_steps=args.num_inference_steps,
                output_type="numpy",
            ).images

        samples = torch.from_numpy(samples).permute([0, 3, 1, 2])
    return samples


def main(args):
    """Main function for pruning and fine-tuning."""
    # loading images for gradient-based pruning

    seed_everything(args.opt_seed, workers=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print_args(args)

    if args.dataset in ["cifar", "cifar2"]:
        config = {**DDPMConfig.cifar_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset in ["cifar100", "cifar100_f"]:
        config = {**DDPMConfig.cifar100_f_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 256, 256).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 256, 256).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "imagenette":
        config = {**DDPMConfig.imagenette_config}
    else:
        raise ValueError(
            (f"dataset={args.dataset} is not one of " f"{constants.DATASET}")
        )
    model_cls = getattr(diffusers, config["unet_config"]["_class_name"])

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config["batch_size"],
        num_workers=4,
    )

    clean_images = next(iter(train_dataloader))
    if isinstance(clean_images, (list, tuple)):
        clean_images = clean_images[0]
    clean_images = clean_images.to(device)
    noise = torch.randn(clean_images.shape).to(clean_images.device)

    pre_trained_path = args.load

    # Loading pretrained(locked) model
    accelerator.print("Loading pretrained model from {}".format(pre_trained_path))

    # load model and scheduler

    existing_steps = get_max_steps(pre_trained_path)
    if existing_steps is not None:
        # Check if there is an existing checkpoint to resume from. This occurs when
        # model runs are interrupted (e.g., exceeding job time limit).
        ckpt_path = os.path.join(
            pre_trained_path, f"ckpt_steps_{existing_steps:0>8}.pt"
        )
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = model_cls(**config["unet_config"])
        model.load_state_dict(ckpt["unet"])
        param_update_steps = 0

        accelerator.print(f"U-Net resumed from {ckpt_path}")

    else:
        raise ValueError(f"No pre-trained checkpoints found at {args.load}")

    if args.dataset == "imagenette":
        # The pipeline is of class LDMTextToImagePipeline.
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
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").to(
            device
        )
        pipeline.unet = model.to(device)
        vqvae = pipeline.vqvae
        pipeline.vqvae.config.scaling_factor = 1
        vqvae.requires_grad_(False)

        vqvae = vqvae.to(device)

        captioner = None
    else:
        pipeline = DDPMPipeline(
            unet=model, scheduler=DDPMScheduler(**config["scheduler_config"])
        ).to(device)
        vqvae = None
        captioner = None

    pipeline_scheduler = pipeline.scheduler

    pruning_params = (
        f"pruner={args.pruner}_pruning_ratio={args.pruning_ratio}_threshold={args.thr}"
    )

    if args.pruning_ratio > 0:
        if args.pruner == "taylor":
            imp = tp.importance.TaylorImportance(
                multivariable=True
            )  # standard first-order taylor expansion
        elif args.pruner == "random" or args.pruner == "reinit":
            imp = tp.importance.RandomImportance()
        elif args.pruner == "magnitude":
            imp = tp.importance.MagnitudeImportance()
        elif args.pruner == "diff-pruning":
            imp = tp.importance.TaylorImportance(
                multivariable=False
            )  # a modified version, estimating the accumulated error of weight removal
        else:
            raise NotImplementedError

        ignored_layers = [model.conv_out]
        channel_groups = {}

        if args.dataset == "celeba":  # Prunig attention for LDM
            for m in model.modules():
                if isinstance(m, Attention):
                    channel_groups[m.to_q] = m.heads
                    channel_groups[m.to_k] = m.heads
                    channel_groups[m.to_v] = m.heads

        pruner = tp.pruner.MagnitudePruner(
            model,
            example_inputs,
            importance=imp,
            iterative_steps=1,
            channel_groups=channel_groups,
            ch_sparsity=args.pruning_ratio,
            ignored_layers=ignored_layers,
        )

        base_macs, base_params = tp.utils.count_ops_and_params(model, example_inputs)
        model.zero_grad()
        model.eval()

        if args.pruner in ["taylor", "diff-pruning"]:
            loss_max = 0
            accelerator.print("Accumulating gradients for pruning...")
            for step_k in tqdm(range(pipeline_scheduler.num_train_timesteps)):
                timesteps = (
                    step_k
                    * torch.ones((config["batch_size"],), device=clean_images.device)
                ).long()
                noisy_images = pipeline_scheduler.add_noise(
                    clean_images, noise, timesteps
                )
                model_output = model(noisy_images, timesteps).sample
                loss = nn.functional.mse_loss(model_output, noise)
                loss.backward()

                if args.pruner == "diff-pruning":
                    if loss > loss_max:
                        loss_max = loss
                    if loss < loss_max * args.thr:
                        # taylor expansion over pruned timesteps ( L_t / L_max > thr )
                        break

        for g in pruner.step(interactive=True):
            g.prune()

        # Update static attributes
        for m in model.modules():
            if isinstance(m, (Upsample2D, Downsample2D)):
                m.channels = m.conv.in_channels
                m.out_channels == m.conv.out_channels

        macs, params = tp.utils.count_ops_and_params(model, example_inputs)
        accelerator.print(model)
        accelerator.print(
            "#Params: {:.4f} M => {:.4f} M".format(base_params / 1e6, params / 1e6)
        )
        accelerator.print(
            "#MACS: {:.4f} G => {:.4f} G".format(base_macs / 1e9, macs / 1e9)
        )
        model.zero_grad()
        del pruner

        if args.pruner == "reinit":

            def reset_parameters(model):
                for m in model.modules():
                    if hasattr(m, "reset_parameters"):
                        m.reset_parameters()

            reset_parameters(model)

    if args.pruning_ratio > 0:
        model_outdir = os.path.join(
            args.outdir, args.dataset, "pruned", "models", pruning_params
        )
        os.makedirs(model_outdir, exist_ok=True)

        # Here the entire pruned model has to be saved.
        torch.save(
            {
                "unet": accelerator.unwrap_model(model),
            },
            os.path.join(model_outdir, f"ckpt_steps_{param_update_steps:0>8}.pt"),
        )
        accelerator.print(f"Checkpoint saved at step {param_update_steps}")

    with torch.no_grad():
        # Testing smaple generation after pruning.

        sample_outdir = os.path.join(
            args.outdir, args.dataset, "pruned", "samples", pruning_params
        )
        os.makedirs(sample_outdir, exist_ok=True)

        samples = run_inference(
            accelerator=accelerator,
            model=model,
            config=config,
            args=args,
            vqvae=vqvae,
            captioner=captioner,
            pipeline=pipeline,
            pipeline_scheduler=pipeline_scheduler,
        )
        if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
            samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]
        img_nrows = int(math.sqrt(config["n_samples"]))
        if args.dataset == "imagenette":
            img_nrows = captioner.num_classes
        save_image(
            samples,
            os.path.join(sample_outdir, f"steps_{param_update_steps:0>8}.png"),
            nrow=img_nrows,
        )
    accelerator.print("Done pruning!")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
