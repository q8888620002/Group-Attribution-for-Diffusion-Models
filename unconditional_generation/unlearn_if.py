"""
Influence unlearning (IU)[1,2] and calculate correpsonding global scores.

[1]: https://github.com/OPTML-Group/Unlearn-Sparse/tree/public
[2]: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/Wfisher.py
"""

import argparse
import json
import math
import os

import diffusers
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline
from diffusers.optimization import get_scheduler
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

import src.constants as constants
from src.attributions.global_scores import fid_score, inception_score, precision_recall
from src.datasets import (
    TensorDataset,
    create_dataset,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)
from src.ddpm_config import DDPMConfig
from src.diffusion_utils import (
    ImagenetteCaptioner,
    LabelTokenizer,
    build_pipeline,
    generate_images,
    load_ckpt_model,
)
from src.unlearn.Wfisher import apply_perturb, get_grad, woodfisher_diff
from src.utils import get_max_steps, print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--load",
        type=str,
        help="directory path for loading pre-trained model",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=constants.DATASET,
        default="mnist",
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
        choices=["uniform", "datamodel", "shapley"],
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
        choices="if",
    )
    parser.add_argument(
        "--if_alpha", type=float, help="ratio for purturbing model weights", default=0.3
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--keep_all_ckpts",
        help="whether to keep all the checkpoints",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before a backward/update pass.",
    )
    # Global behavior calculation related.

    parser.add_argument(
        "--db",
        type=str,
        help="filepath of database for recording scores",
        required=True,
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        help="directory path of reference samples, from a dataset or a diffusion model",
        default=None,
    )
    parser.add_argument(
        "--use_ema",
        help="whether to use the EMA model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size for computation",
        default=128,
    )
    parser.add_argument(
        "--n_samples", type=int, default=10240, help="number of generated samples"
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
        "--precompute_stage",
        type=str,
        default=None,
        choices=[None, "save", "reuse"],
        help=(
            "Whether to precompute the VQVAE output."
            "Choose between None, save, and reuse."
        ),
    )
    parser.add_argument(
        "--use_8bit_optimizer",
        default=False,
        action="store_true",
        help="Whether to use 8bit optimizer",
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
    return parser.parse_args()


def main(args):
    """Main function for training or unlearning."""

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device
    args.device = device

    info_dict = vars(args)

    if accelerator.is_main_process:
        print_args(args)

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "cifar2":
        config = {**DDPMConfig.cifar2_config}
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
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

    sample_outdir = os.path.join(
        args.outdir, args.dataset, args.method, "samples", removal_dir
    )

    if accelerator.is_main_process:
        # Make the output directories once in the main process.
        os.makedirs(sample_outdir, exist_ok=True)

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)
    if args.excluded_class is not None:
        remaining_idx, removed_idx = remove_data_by_class(
            train_dataset, excluded_class=args.excluded_class
        )
    elif args.removal_dist is not None:
        if args.removal_dist == "uniform":
            remaining_idx, removed_idx = remove_data_by_uniform(
                train_dataset, seed=args.removal_seed
            )
        elif args.removal_dist == "datamodel":
            remaining_idx, removed_idx = remove_data_by_datamodel(
                train_dataset, alpha=args.datamodel_alpha, seed=args.removal_seed
            )
        elif args.removal_dist == "shapley":
            remaining_idx, removed_idx = remove_data_by_shapley(
                train_dataset, seed=args.removal_seed
            )
        else:
            raise NotImplementedError
    else:
        remaining_idx = np.arange(len(train_dataset))
        removed_idx = np.array([], dtype=int)

    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    model_strc = model_cls(**config["unet_config"])

    args.trained_steps = get_max_steps(args.load)

    model, remaining_idx, removed_idx = load_ckpt_model(
        args, model_cls, model_strc, args.load
    )
    pipeline = build_pipeline(args, model)

    remaining_dataloader = DataLoader(
        Subset(train_dataset, remaining_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        generator=torch.Generator().manual_seed(args.opt_seed),
    )

    training_steps = len(remaining_dataloader)
    removed_dataloader = remaining_dataloader

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
        # The pipeline is of class LDMPipeline.
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        pipeline.unet = model

        vqvae = pipeline.vqvae
        pipeline.vqvae.config.scaling_factor = 1
        vqvae.requires_grad_(False)

        if args.precompute_stage is None:
            # Move the VQ-VAE model to the device without any operations.
            vqvae = vqvae.to(device)

        elif args.precompute_stage == "save":
            assert removal_dir == "full", "Precomputation should be done for full data"
            # Precompute and save the VQ-VAE latents
            vqvae = vqvae.to(device)
            vqvae.train()  # The vqvae output is STATIC even in train mode.
            # if accelerator.is_main_process:
            vqvae_latent_dict = {}
            with torch.no_grad():
                for image_temp, label_temp, imageid_temp in tqdm(
                    DataLoader(
                        dataset=train_dataset,
                        batch_size=32,
                        num_workers=4,
                        shuffle=False,
                    )
                ):
                    vqvae_latent = vqvae.encode(image_temp.to(device), False)[0]
                    assert len(vqvae_latent) == len(
                        image_temp
                    ), "Output and input batch sizes should match"

                    # Store the encoded outputs in the dictionary
                    for i in range(len(vqvae_latent)):
                        vqvae_latent_dict[imageid_temp[i]] = vqvae_latent[i]

            # Save the dictionary of latents to a file
            vqvae_latent_dir = os.path.join(
                args.outdir,
                args.dataset,
                "precomputed_emb",
            )
            os.makedirs(vqvae_latent_dir, exist_ok=True)
            torch.save(
                vqvae_latent_dict,
                os.path.join(vqvae_latent_dir, "vqvae_output.pt"),
            )

            accelerator.print(
                "VQVAE output saved. Set precompute_state=reuse to unload VQVAE model."
            )
            raise SystemExit(0)
        elif args.precompute_stage == "reuse":
            # Load the precomputed output, avoiding GPU memory usage by the VQ-VAE model
            pipeline.vqvae = None
            vqvae_latent_dir = os.path.join(
                args.outdir,
                args.dataset,
                "precomputed_emb",
            )
            vqvae_latent_dict = torch.load(
                os.path.join(
                    vqvae_latent_dir,
                    "vqvae_output.pt",
                ),
                map_location="cpu",
            )

        captioner = None
    else:
        pipeline = DDPMPipeline(
            unet=model, scheduler=DDPMScheduler(**config["scheduler_config"])
        ).to(device)
        vqvae = None
        captioner = None
    pipeline_scheduler = pipeline.scheduler

    if not args.use_8bit_optimizer:
        optimizer_kwargs = config["optimizer_config"]["kwargs"]
        optimizer = getattr(torch.optim, config["optimizer_config"]["class_name"])(
            model.parameters(), **optimizer_kwargs
        )
    else:
        # https://huggingface.co/docs/transformers/v4.20.1/en/perf_train_gpu_one#8bit-adam
        import bitsandbytes as bnb
        from transformers.trainer_pt_utils import get_parameter_names

        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": config["optimizer_config"]["kwargs"]["weight_decay"],
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer_kwargs = config["optimizer_config"]["kwargs"]
        del optimizer_kwargs["weight_decay"]
        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            **optimizer_kwargs,
        )

    lr_scheduler_kwargs = config["lr_scheduler_config"]["kwargs"]
    lr_scheduler = get_scheduler(
        config["lr_scheduler_config"]["name"],
        optimizer=optimizer,
        num_training_steps=training_steps,
        **lr_scheduler_kwargs,
    )

    (
        remaining_dataloader,
        removed_dataloader,
        model,
        optimizer,
        pipeline_scheduler,
        lr_scheduler,
    ) = accelerator.prepare(
        remaining_dataloader,
        removed_dataloader,
        model,
        optimizer,
        pipeline_scheduler,
        lr_scheduler,
    )

    # Influence Unlearning (IU)
    # This is mainly from Wfisher() in
    # https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/Wfisher.py#L113.

    model.eval()

    vqvae_latent_dict = (
        None
        if not (args.dataset == "celeba" and args.precompute_stage == "reuse")
        else vqvae_latent_dict
    )

    print("Calculating forget gradients....")
    total, forget_grad = get_grad(args, removed_dataloader, pipeline, vqvae_latent_dict)

    print("Calculating remaining gradients...")
    total_2, retain_grad = get_grad(
        args, remaining_dataloader, pipeline, vqvae_latent_dict
    )

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    perturb = woodfisher_diff(
        args,
        remaining_dataloader,
        pipeline,
        retain_grad - forget_grad,
        vqvae_latent_dict,
    )

    # Apply parameter purturbation to Unet.

    model = apply_perturb(model, args.if_alpha * perturb)
    pipeline.unet = model

    # Calculate global model score.
    # This is done only once for the main process.

    if accelerator.is_main_process:
        samples = pipeline(
            batch_size=config["n_samples"],
            num_inference_steps=args.num_inference_steps,
            output_type="numpy",
        ).images

        samples = torch.from_numpy(samples).permute([0, 3, 1, 2])

        save_image(
            samples,
            os.path.join(
                sample_outdir, f"prutirb_ratio_{args.if_alpha}_steps_{0:0>8}.png"
            ),
            nrow=int(math.sqrt(config["n_samples"])),
        )
        print(f"Save test images, steps_{0:0>8}.png, in {sample_outdir}.")
        print(f"Generating {args.n_samples}...")

        generated_samples = generate_images(args, pipeline)

        images_dataset = TensorDataset(generated_samples)

        is_value = inception_score.eval_is(
            images_dataset, args.batch_size, resize=True, normalize=True
        )

        precision, recall = precision_recall.eval_pr(
            args.dataset,
            images_dataset,
            args.batch_size,
            row_batch_size=10000,
            col_batch_size=10000,
            nhood_size=3,
            device=device,
            reference_dir=args.reference_dir,
        )

        fid_value_str = fid_score.calculate_fid(
            args.dataset,
            images_dataset,
            args.batch_size,
            device,
            args.reference_dir,
        )

        print(
            f"FID score: {fid_value_str}; Precision:{precision};"
            f"Recall:{recall}; inception score: {is_value}"
        )
        info_dict["fid_value"] = fid_value_str
        info_dict["precision"] = precision
        info_dict["recall"] = recall
        info_dict["is"] = is_value

        with open(args.db, "a+") as f:
            f.write(json.dumps(info_dict) + "\n")
        print(f"Results saved to the database at {args.db}")

        return accelerator.is_main_process


if __name__ == "__main__":
    args = parse_args()
    is_main_process = main(args)
    if is_main_process:
        print("Influence unlearning done!")
