"""Train or perform unlearning on a diffusion model."""

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

import diffusers
import src.constants as constants
import wandb  # wandb for monitoring loss https://wandb.ai/
from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from src.datasets import (
    create_dataset,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)
from src.ddpm_config import DDPMConfig
from src.diffusion_utils import ImagenetteCaptioner, LabelTokenizer, run_inference
from src.utils import compute_grad_norm, compute_param_norm, get_max_steps, print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--load",
        type=str,
        help="directory path for loading pre-trained model",
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=constants.DATASET,
        required=True,
    )
    parser.add_argument(
        "--log_freq",
        type=int,
        help="training log message printing frequence",
        default=20,
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
        "--wandb",
        help="whether to monitor model training with wandb",
        action="store_true",
        default=False,
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
        choices=constants.METHOD,
        required=True,
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
        "--db", type=str, help="database file for storing results", default=None
    )
    parser.add_argument(
        "--exp_name", type=str, help="experiment name in the database", default=None
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before a backward/update pass.",
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
    parser.add_argument(
        "--save_null_model",
        action="store_true",
        default=False,
        help="Whether to save the null model",
    )
    return parser.parse_args()


def main(args):
    """Main function for training or unlearning."""

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
    )
    device = accelerator.device

    if accelerator.is_main_process:
        print_args(args)

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "cifar2":
        config = {**DDPMConfig.cifar2_config}
    elif args.dataset == "cifar100":
        config = {**DDPMConfig.cifar100_config}
    elif args.dataset == "cifar100_f":
        config = {**DDPMConfig.cifar100_f_config}
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
    elif args.dataset == "imagenette":
        config = {**DDPMConfig.imagenette_config}
    else:
        raise ValueError((f"dataset={args.dataset} is not one of {constants.DATASET}"))
    model_cls = getattr(diffusers, config["unet_config"]["_class_name"])

    removal_dir = "full"
    if args.excluded_class is not None:
        removal_dir = f"excluded_{args.excluded_class}"
    if args.removal_dist is not None:
        removal_dir = f"{args.removal_dist}/{args.removal_dist}"
        if args.removal_dist == "datamodel":
            removal_dir += f"_alpha={args.datamodel_alpha}"
        removal_dir += f"_seed={args.removal_seed}"

    model_outdir = os.path.join(
        args.outdir,
        args.dataset,
        args.method,
        "models",
        removal_dir,
    )
    sample_outdir = os.path.join(
        args.outdir, args.dataset, args.method, "samples", removal_dir
    )

    if accelerator.is_main_process:
        # Make the output directories once in the main process.
        os.makedirs(model_outdir, exist_ok=True)
        os.makedirs(sample_outdir, exist_ok=True)

    train_dataset = create_dataset(dataset_name=args.dataset, train=True)
    if args.excluded_class is not None:
        remaining_idx, removed_idx = remove_data_by_class(
            train_dataset, excluded_class=args.excluded_class
        )
    elif args.removal_dist is not None:
        if args.removal_dist == "uniform":
            remaining_idx, removed_idx = remove_data_by_uniform(
                train_dataset, seed=args.removal_seed, by_class=True
            )
        elif args.removal_dist == "datamodel":
            if args.dataset in ["cifar100", "cifar100_f", "celeba"]:
                remaining_idx, removed_idx = remove_data_by_datamodel(
                    train_dataset,
                    alpha=args.datamodel_alpha,
                    seed=args.removal_seed,
                    by_class=True,
                )
            else:
                remaining_idx, removed_idx = remove_data_by_datamodel(
                    train_dataset, alpha=args.datamodel_alpha, seed=args.removal_seed
                )
        elif args.removal_dist == "shapley":
            if args.dataset in ["cifar100", "cifar100_f", "celeba"]:
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed, by_class=True
                )
            else:
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed
                )
        else:
            raise NotImplementedError
    else:
        remaining_idx = np.arange(len(train_dataset))
        removed_idx = np.array([], dtype=int)

    if args.method == "ga":
        # Gradient ascent trains on the removed images.
        remaining_idx, removed_idx = removed_idx, remaining_idx

    # Save the removed and remaining indices for reproducibility.
    np.save(os.path.join(model_outdir, "remaining_idx.npy"), remaining_idx)
    np.save(os.path.join(model_outdir, "removed_idx.npy"), removed_idx)

    seed_everything(args.opt_seed, workers=True)  # Seed for model optimization.

    total_steps_time = 0
    training_steps = config["training_steps"][args.method]
    existing_steps = get_max_steps(model_outdir)

    # Load full model instead of state_dict for pruned model.
    # if method is retrain.
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
        accelerator.print(f"Pruned U-Net resumed from {pruned_model_path}")
    else:
        model = model_cls(**config["unet_config"])

    if existing_steps is not None:
        # Check if there is an existing checkpoint to resume from. This occurs when
        # model runs are interrupted (e.g., exceeding job time limit).
        ckpt_path = os.path.join(model_outdir, f"ckpt_steps_{existing_steps:0>8}.pt")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")

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
            ema_model.load_state_dict(ckpt["unet_ema"])
            param_update_steps = existing_steps

            remaining_idx = ckpt["remaining_idx"].numpy()
            removed_idx = ckpt["removed_idx"].numpy()
            total_steps_time = ckpt["total_steps_time"]

            accelerator.print(f"U-Net and U-Net EMA resumed from {ckpt_path}")

        except RuntimeError:
            existing_steps = None
            # If the ckpt file is corrupted, reinit the model.
            accelerator.print(
                f"Check point {ckpt_path} is corrupted, "
                " reintialize model and remove old check point.."
            )

            os.system(f"rm -rf {model_outdir}")
            # Randomly initialize the model.
            model = model_cls(**config["unet_config"])
            ema_model = EMAModel(
                model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=False,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=model_cls,
                model_config=model.config,
            )
            param_update_steps = 0
            accelerator.print("Model randomly initialized")

    elif args.load:
        # If there are no checkpoints to resume from, and a pre-trained model is
        # specified for fine-tuning or unlearning.
        pretrained_steps = get_max_steps(args.load)
        if pretrained_steps is not None:
            ckpt_path = os.path.join(args.load, f"ckpt_steps_{pretrained_steps:0>8}.pt")
            ckpt = torch.load(ckpt_path, map_location="cpu")

            model.load_state_dict(ckpt["unet"])

            # Consider the pre-trained model as model weight initialization, so the EMA
            # starts with the pre-trained model.
            ema_model = EMAModel(
                model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=False,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_cls=model_cls,
                model_config=model.config,
            )
            param_update_steps = 0

            accelerator.print(f"Pre-trained model loaded from {args.load}")
            accelerator.print(f"\tU-Net loaded from {ckpt_path}")
            accelerator.print("\tEMA started from the loaded U-Net")
        else:
            raise ValueError(f"No pre-trained checkpoints found at {args.load}")
    else:
        # Randomly initialize the model.
        ema_model = EMAModel(
            model.parameters(),
            decay=args.ema_max_decay,
            use_ema_warmup=False,
            inv_gamma=args.ema_inv_gamma,
            power=args.ema_power,
            model_cls=model_cls,
            model_config=model.config,
        )
        param_update_steps = 0
        accelerator.print("Model randomly initialized")
    ema_model.to(device)

    if (accelerator.state.num_processes) > 1:
        assert (
            config["batch_size"] % accelerator.state.num_processes == 0
        ), "Batch size should be divisible by number of processes"
        accelerator.print(
            f"Batch size is {config['batch_size']} "
            f"and number of processes is {accelerator.state.num_processes}",
            f"Batch size will be divided by number of processes."
            "Per process batch size is "
            f"{config['batch_size'] // accelerator.state.num_processes}",
        )

    num_workers = 4 if torch.get_num_threads() >= 4 else torch.get_num_threads()

    if len(remaining_idx) < (config["batch_size"] // accelerator.state.num_processes):
        shuffle = False

        remaining_dataloader = DataLoader(
            Subset(train_dataset, remaining_idx),
            batch_size=len(remaining_idx),
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
        )
    else:
        shuffle = True

        remaining_dataloader = DataLoader(
            Subset(train_dataset, remaining_idx),
            batch_size=config["batch_size"] // accelerator.state.num_processes,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

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
                weights_only=True,
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
    if existing_steps is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        accelerator.print(f"Optimizer and lr scheduler resumed from {ckpt_path}")

    loss_fn = nn.MSELoss(reduction="mean")

    if args.wandb:
        wandb.init(
            project="Data Shapley for Diffusion",
            notes=f"Experiment for {args.method};{args.removal_dist};{args.dataset}",
            dir="/gscratch/aims/diffusion-attr/results_ming/wandb",
            tags=[f"{args.method}"],
            config={
                "training_steps": training_steps,
                "batch_size": config["batch_size"],
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "model": model.config._class_name,
            },
        )

    (
        remaining_dataloader,
        model,
        optimizer,
        pipeline_scheduler,
        lr_scheduler,
    ) = accelerator.prepare(
        remaining_dataloader,
        model,
        optimizer,
        pipeline_scheduler,
        lr_scheduler,
    )

    if args.save_null_model and accelerator.is_main_process:
        torch.save(
            {
                "unet": accelerator.unwrap_model(model).state_dict(),
                "unet_ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict(),
                "remaining_idx": torch.from_numpy(remaining_idx),
                "removed_idx": torch.from_numpy(removed_idx),
                "total_steps_time": total_steps_time,
            },
            os.path.join(model_outdir, f"ckpt_steps_{param_update_steps:0>8}.pt"),
        )
        print(f"Checkpoint saved at step {param_update_steps}")

    progress_bar = tqdm(
        range(training_steps),
        initial=param_update_steps,
        desc="Step",
        disable=not accelerator.is_main_process,
    )
    steps_start_time = time.time()
    while param_update_steps < training_steps:
        for j, batch_r in enumerate(remaining_dataloader):

            model.train()

            image_r, label_r = batch_r[0], batch_r[1]

            if args.precompute_stage == "reuse":
                imageid_r = batch_r[2]

            image_r = image_r.to(device)

            if args.dataset == "imagenette":
                image_r = vqvae.encode(image_r).latent_dist.sample()
                image_r = image_r * vqvae.config.scaling_factor
                input_ids_r = label_tokenizer(label_r).to(device)
                encoder_hidden_states_r = text_encoder(input_ids_r)[0]
            elif args.dataset == "celeba":
                if args.precompute_stage is None:
                    # Directly encode the images if there's no precomputation
                    image_r = vqvae.encode(image_r, False)[0]
                elif args.precompute_stage == "reuse":
                    # Retrieve the latent representations.
                    image_r = torch.stack(
                        [vqvae_latent_dict[imageid_r[i]] for i in range(len(image_r))]
                    ).to(device)
                image_r = image_r * vqvae.config.scaling_factor
            noise = torch.randn_like(image_r).to(device)

            # Antithetic sampling of time steps.
            timesteps = torch.randint(
                0,
                pipeline_scheduler.config.num_train_timesteps,
                (len(image_r) // 2 + 1,),
                device=image_r.device,
            ).long()
            timesteps = torch.cat(
                [
                    timesteps,
                    pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
                ],
                dim=0,
            )[: len(image_r)]

            noisy_images_r = pipeline_scheduler.add_noise(image_r, noise, timesteps)

            with accelerator.accumulate(model):
                optimizer.zero_grad()
                if args.dataset == "imagenette":
                    eps_r = model(
                        noisy_images_r, timesteps, encoder_hidden_states_r
                    ).sample
                else:
                    eps_r = model(noisy_images_r, timesteps).sample
                loss = loss_fn(eps_r, noise)

                if args.method == "ga":
                    loss *= -1.0

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    # Clip the gradients when the gradients are synced. This has to
                    # happen before calling optimizer.step() to update the model
                    # parameters.
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()

                if accelerator.sync_gradients:
                    # Update the EMA model when the gradients are synced (that is, when
                    # model parameters are updated).
                    ema_model.step(model.parameters())
                    param_update_steps += 1
                    progress_bar.update(1)

                    # Print info when the parameters have enough number of updates.
                    # This is done only once with the main process.
                    if (
                        param_update_steps % args.log_freq == 0
                        and accelerator.is_main_process
                    ):
                        steps_time = time.time() - steps_start_time
                        total_steps_time += steps_time

                        info = f"Step[{param_update_steps}/{training_steps}]"
                        info += f", steps_time: {steps_time:.3f}"
                        info += f", loss: {loss.detach().cpu().item():.5f}"

                        # Check gradient norm and parameter norm.
                        grad_norm = compute_grad_norm(
                            accelerator=accelerator, model=model
                        )
                        param_norm = compute_param_norm(
                            accelerator=accelerator, model=model
                        )
                        info += f", gradient norms: {grad_norm:.5f}"
                        info += f", parameters norms: {param_norm:.5f}"
                        info += f", lr: {lr_scheduler.get_last_lr()[0]:.6f}"
                        print(info, flush=True)

                        if args.wandb:
                            wandb.log(
                                {
                                    "Step": param_update_steps,
                                    "loss": loss.detach().cpu().item(),
                                    "steps_time": steps_time,
                                    "gradient norm": grad_norm,
                                    "parameter norm": param_norm,
                                    "lr": lr_scheduler.get_last_lr()[0],
                                }
                            )
                        steps_start_time = time.time()

                    # Generate samples for evaluation. This is done only once for the
                    # main process.
                    if (
                        param_update_steps % config["sample_freq"][args.method] == 0
                        or param_update_steps == training_steps
                    ) and accelerator.is_main_process:
                        sampling_start_time = time.time()
                        samples = run_inference(
                            accelerator=accelerator,
                            model=model,
                            ema_model=ema_model,
                            config=config,
                            args=args,
                            vqvae=vqvae,
                            captioner=captioner,
                            pipeline=pipeline,
                            pipeline_scheduler=pipeline_scheduler,
                        )
                        sampling_time = time.time() - sampling_start_time
                        sampling_info = f"Step[{param_update_steps}/{training_steps}]"
                        sampling_info += f", sampling_time: {sampling_time:.3f}"
                        print(sampling_info, flush=True)

                        if args.db is not None:
                            info_dict = vars(args)
                            info_dict["param_update_steps"] = f"{param_update_steps}"
                            info_dict["loss"] = f"{loss.detach().cpu().item():.5f}"
                            info_dict["lr"] = f"{lr_scheduler.get_last_lr()[0]:.6f}"
                            info_dict["steps_time"] = f"{steps_time:.3f}"
                            info_dict["sampling_time"] = f"{sampling_time:.3f}"

                            with open(args.db, "a+") as f:
                                f.write(json.dumps(info_dict) + "\n")
                            print(f"Results saved to the database at {args.db}")

                        if len(samples) > constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE:
                            samples = samples[: constants.MAX_NUM_SAMPLE_IMAGES_TO_SAVE]
                        img_nrows = int(math.sqrt(config["n_samples"]))
                        if args.dataset == "imagenette":
                            img_nrows = captioner.num_classes
                        save_image(
                            samples,
                            os.path.join(
                                sample_outdir, f"steps_{param_update_steps:0>8}.png"
                            ),
                            nrow=img_nrows,
                        )
                        steps_start_time = time.time()

                    # Checkpoints for training. This is done only once for the main
                    # process.
                    if (
                        param_update_steps % config["ckpt_freq"][args.method] == 0
                        or param_update_steps == training_steps
                    ) and accelerator.is_main_process:
                        if not args.keep_all_ckpts:
                            pattern = os.path.join(model_outdir, "ckpt_steps_*.pt")
                            for filename in glob.glob(pattern):
                                os.remove(filename)

                        torch.save(
                            {
                                "unet": accelerator.unwrap_model(model).state_dict(),
                                "unet_ema": ema_model.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "lr_scheduler": lr_scheduler.state_dict(),
                                "remaining_idx": torch.from_numpy(remaining_idx),
                                "removed_idx": torch.from_numpy(removed_idx),
                                "total_steps_time": total_steps_time,
                            },
                            os.path.join(
                                model_outdir, f"ckpt_steps_{param_update_steps:0>8}.pt"
                            ),
                        )
                        print(f"Checkpoint saved at step {param_update_steps}")
                        steps_start_time = time.time()

            if param_update_steps == training_steps:
                break
    return accelerator.is_main_process


if __name__ == "__main__":
    args = parse_args()
    is_main_process = main(args)
    if is_main_process:
        print("Model optimization done!")
