"""
Influence unlearning (IU) with wood fisher approximation from [1,2,3]

and calculate correpsonding global scores.

[1]: https://github.com/OPTML-Group/Unlearn-Sparse/tree/public
[2]: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/Wfisher.py
[3]: https://arxiv.org/pdf/2304.04934.pdf
"""

import argparse
import json
import math
import os
import time
from copy import deepcopy

import bitsandbytes as bnb
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from lightning.pytorch import seed_everything
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    structural_similarity,
)
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm
from transformers.trainer_pt_utils import get_parameter_names

import src.constants as constants
from diffusers.optimization import get_scheduler
from src.attributions.global_scores import fid_score, inception_score, precision_recall
from src.attributions.global_scores.diversity_score import calculate_diversity_score
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
        help='Classes to be excluded, e.g. "1, 2, 3, etc" ',
        type=str,
        default=None,
    )
    parser.add_argument(
        "--removal_dist",
        type=str,
        help="distribution for removing data",
        choices=["uniform", "datamodel", "shapley", "loo", "add_one_in"],
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
        choices=constants.METHOD,
    )
    parser.add_argument(
        "--iu_ratio", type=float, help="ratio for purturbing model weights", default=0.5
    )
    parser.add_argument(
        "--ga_ratio", type=float, help="ratio for unlearning steps", default=1.0
    )
    parser.add_argument(
        "--gd_steps", type=int, help="ratio for unlearning steps", default=4000
    )
    parser.add_argument(
        "--lora_rank", type=int, help="rank of matrix for LORA", default=16
    )
    parser.add_argument(
        "--lora_dropout", type=float, help="rank of matrix for LORA", default=0.05
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
        "--batch_size",
        type=int,
        help="batch size for computation",
        default=32,
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
        "--model_behavior",
        type=str,
        help="experiment name to record in the database file",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--use_ema",
        help="whether to use the EMA model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="experiment name to record in the database file",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--n_noises",
        type=int,
        help="number of noises per sample and timestep for computing diffusion loss",
        default=50,
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
        "--pruned",
        help="whether to used pruned model",
        default=False,
        action="store_true",
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
    elif args.dataset == "cifar100":
        config = {**DDPMConfig.cifar100_config}
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
    elif args.dataset == "mnist":
        config = {**DDPMConfig.mnist_config}
    elif args.dataset == "imagenette":
        config = {**DDPMConfig.imagenette_config}
    else:
        raise ValueError(
            (f"dataset={args.dataset} is not one of " f"{constants.DATASETs}")
        )

    removal_dir = "full"
    if args.excluded_class is not None:
        excluded_class = [int(k) for k in args.excluded_class.split(",")]
        excluded_class.sort()
        excluded_class_str = ",".join(map(str, excluded_class))
        removal_dir = f"excluded_{excluded_class_str}"
    elif args.removal_dist is not None:
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

    if args.removal_dist is not None:
        if args.removal_dist == "uniform":
            remaining_idx, removed_idx = remove_data_by_uniform(
                train_dataset, seed=args.removal_seed, by_class=True
            )
        elif args.removal_dist == "datamodel":
            remaining_idx, removed_idx = remove_data_by_datamodel(
                train_dataset, alpha=args.datamodel_alpha, seed=args.removal_seed
            )
        elif args.removal_dist == "shapley":
            if args.dataset == "cifar100" or "celeba":
                args.by_class = True
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed, by_class=args.by_class
                )
            else:
                args.by_class = False
                remaining_idx, removed_idx = remove_data_by_shapley(
                    train_dataset, seed=args.removal_seed
                )
        elif args.removal_dist in ["loo", "add_one_in"]:
            excluded_class = [int(k) for k in args.excluded_class.split(",")]
            remaining_idx, removed_idx = remove_data_by_class(
                train_dataset, excluded_class=excluded_class
            )
        else:
            if args.excluded_class is not None:
                excluded_class = [int(k) for k in args.excluded_class.split(",")]
                remaining_idx, removed_idx = remove_data_by_class(
                    train_dataset, excluded_class=excluded_class
                )
            else:
                raise NotImplementedError
    else:
        remaining_idx = np.arange(len(train_dataset))
        removed_idx = np.array([], dtype=int)

    # Seed for model optimization.
    seed_everything(args.opt_seed, workers=True)

    # Load model structure depending on unlearning methods.

    args.trained_steps = get_max_steps(args.load)
    model, ema_model, _, _ = load_ckpt_model(args, args.load)

    model.to(device)
    ema_model.to(device)

    pipeline, vqvae, vqvae_latent_dict = build_pipeline(args, model)

    num_workers = 4 if torch.get_num_threads() >= 4 else torch.get_num_threads()

    remaining_dataloader = DataLoader(
        Subset(train_dataset, remaining_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    removed_dataloader = DataLoader(
        Subset(train_dataset, removed_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    if args.method == "ga":
        training_steps = len(removed_dataloader)
    else:
        training_steps = len(remaining_dataloader)

    pipeline_scheduler = pipeline.scheduler

    if not args.use_8bit_optimizer:
        optimizer_kwargs = config["optimizer_config"]["kwargs"]

        if args.method == "ga":
            optimizer_kwargs["lr"] = 1e-5
            optimizer = getattr(torch.optim, config["optimizer_config"]["class_name"])(
                model.parameters(), **optimizer_kwargs
            )
        elif args.method in ["lora", "lora_u"]:
            from peft import LoraConfig, get_peft_model
            from peft.optimizers import create_loraplus_optimizer

            # Initialize LORA adapter

            lora_config = LoraConfig(
                r=args.lora_rank,
                target_modules=["to_q", "to_v", "to_k"],
                lora_alpha=32,
                lora_dropout=args.lora_dropout,
            )
            model = get_peft_model(model, lora_config)
            optimizer = create_loraplus_optimizer(
                model=model,
                optimizer_cls=getattr(
                    torch.optim, config["optimizer_config"]["class_name"]
                ),
                lr=config["optimizer_config"]["kwargs"]["lr"],
                loraplus_lr_ratio=16,
            )
        else:
            optimizer = getattr(torch.optim, config["optimizer_config"]["class_name"])(
                model.parameters(), **optimizer_kwargs
            )
    else:

        if args.method in ["lora", "lora_u"]:
            from peft import LoraConfig, get_peft_model
            from peft.optimizers import create_loraplus_optimizer

            # Initialize LORA adapter

            lora_config = LoraConfig(
                r=args.lora_rank,
                target_modules=["to_q", "to_v", "to_k"],
                lora_alpha=32,
                lora_dropout=args.lora_dropout,
            )
            model = get_peft_model(model, lora_config)
            optimizer = create_loraplus_optimizer(
                model=model,
                optimizer_cls=bnb.optim.Adam8bit,
                lr=config["optimizer_config"]["kwargs"]["lr"],
                loraplus_lr_ratio=16,
            )

        else:
            decay_parameters = get_parameter_names(model, [nn.LayerNorm])
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in model.named_parameters() if n in decay_parameters
                    ],
                    "weight_decay": config["optimizer_config"]["kwargs"][
                        "weight_decay"
                    ],
                },
                {
                    "params": [
                        p
                        for n, p in model.named_parameters()
                        if n not in decay_parameters
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

    unlearn_start_time = time.time()

    if args.method in ["iu", "iu_u"]:
        model.eval()
        vqvae_latent_dict = (
            None
            if not (args.dataset == "celeba" and args.precompute_stage == "reuse")
            else vqvae_latent_dict
        )

        print("Calculating gradients with removed dataset....")
        forget_count, forget_grad = get_grad(
            args, removed_dataloader, pipeline, vqvae_latent_dict
        )

        print("Calculating gradients with remaining dataset...")
        retain_count, retain_grad = get_grad(
            args, remaining_dataloader, pipeline, vqvae_latent_dict
        )

        # weight normalization to ensure 1^Tw =1
        retain_grad *= forget_count / ((forget_count + retain_count) * retain_count)

        # 1/N in equation (1)
        forget_grad /= forget_count + retain_count

        # woodfisher approximation for hessian matrix
        delta_w = woodfisher_diff(
            args,
            retain_count,
            remaining_dataloader,
            pipeline,
            forget_grad - retain_grad,
            vqvae_latent_dict,
        )

        # Apply parameter purturbation to Unet.
        print("Applying perturbation...")
        model = apply_perturb(model, args.iu_ratio * delta_w)
        ema_model.step(model.parameters())

    elif args.method in ["gd", "gd_u", "lora", "lora_u"]:
        param_update_steps = 0

        progress_bar = tqdm(
            range(args.gd_steps),
            initial=param_update_steps,
            desc="Step",
            disable=not accelerator.is_main_process,
        )

        loss_fn = nn.MSELoss(reduction="mean")

        while param_update_steps < args.gd_steps:

            for j, batch in enumerate(remaining_dataloader):

                model.train()

                image, label = batch[0], batch[1]

                if args.precompute_stage == "reuse":
                    imageid_r = batch[2]

                image = image.to(device)

                if args.dataset == "imagenette":
                    image = vqvae.encode(image).latent_dist.sample()
                    image = image * vqvae.config.scaling_factor
                    input_ids_r = label_tokenizer(label).to(device)
                    encoder_hidden_states = text_encoder(input_ids_r)[0]
                elif args.dataset == "celeba":
                    if args.precompute_stage is None:
                        # Directly encode the images if there's no precomputation
                        image = vqvae.encode(image, False)[0]
                    elif args.precompute_stage == "reuse":
                        # Retrieve the latent representations.
                        image = torch.stack(
                            [vqvae_latent_dict[imageid_r[i]] for i in range(len(image))]
                        ).to(device)
                    image = image * vqvae.config.scaling_factor
                noise = torch.randn_like(image).to(device)

                # Antithetic sampling of time steps.
                timesteps = torch.randint(
                    0,
                    pipeline_scheduler.config.num_train_timesteps,
                    (len(image) // 2 + 1,),
                    device=image.device,
                ).long()
                timesteps = torch.cat(
                    [
                        timesteps,
                        pipeline_scheduler.config.num_train_timesteps - timesteps - 1,
                    ],
                    dim=0,
                )[: len(image)]

                noisy_images = pipeline_scheduler.add_noise(image, noise, timesteps)

                with accelerator.accumulate(model):
                    optimizer.zero_grad()
                    if args.dataset == "imagenette":
                        eps = model(
                            noisy_images, timesteps, encoder_hidden_states
                        ).sample
                    else:
                        eps = model(noisy_images, timesteps).sample

                    loss = loss_fn(eps, noise)
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        # Clip the gradients when the gradients are synced. This has to
                        # happen before calling optimizer.step() to update the model
                        # parameters.
                        accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    lr_scheduler.step()

                    if accelerator.sync_gradients:
                        # Update the EMA model when the gradients are synced
                        # (that is, when model parameters are updated).
                        if args.method in ["lora", "lora_u"]:
                            # Merge LORA to the base model
                            merged_model = deepcopy(model)
                            merged_model.merge_and_unload()
                            ema_model.step(merged_model.parameters())
                            del merged_model
                        else:
                            ema_model.step(model.parameters())

                        param_update_steps += 1
                        progress_bar.update(1)

                if param_update_steps == args.gd_steps:
                    break

        model = accelerator.unwrap_model(model).eval()

    elif args.method in ["ga", "ga_u"]:

        training_steps = int(training_steps // args.ga_ratio)
        param_update_steps = 0

        progress_bar = tqdm(
            range(training_steps),
            initial=param_update_steps,
            desc="Step",
            disable=not accelerator.is_main_process,
        )

        loss_fn = nn.MSELoss(reduction="mean")

        while param_update_steps < training_steps:

            for j, batch_r in enumerate(removed_dataloader):

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
                            [
                                vqvae_latent_dict[imageid_r[i]]
                                for i in range(len(image_r))
                            ]
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
                        # Update the EMA model when the gradients are synced
                        # (that is, when model parameters are updated).
                        ema_model.step(model.parameters())
                        param_update_steps += 1
                        progress_bar.update(1)

                if param_update_steps == training_steps:
                    break

        model = accelerator.unwrap_model(model).eval()
    else:
        raise ValueError((f"Unlearning method: {args.method} doesn't exist "))

    total_steps_time = time.time() - unlearn_start_time
    # The EMA is used for inference.

    if args.method in ["lora", "lora_u"]:
        model = model.merge_and_unload()

    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
    pipeline.unet = model

    if args.dataset == "celeba":
        pipeline.vqvae = vqvae
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
                sample_outdir,
                f"prutirb_ratio_{args.iu_ratio}_steps_{training_steps:0>8}.png",
            ),
            nrow=int(math.sqrt(config["n_samples"])),
        )
        print(f"Save test images, steps_{training_steps:0>8}.png, in {sample_outdir}.")

        behavior_start_time = time.time()

        if args.model_behavior == "global":

            print(f"Generating {args.n_samples}...")

            generated_samples = generate_images(args, pipeline)

            if args.dataset == "celeba":
                (
                    entropy,
                    cluster_count,
                    cluster_proportions,
                    ref_cluster_images,
                    new_cluster_images,
                ) = calculate_diversity_score(
                    ref_image_dir_or_tensor=os.path.join(
                        constants.OUTDIR, args.dataset, "cluster_imgs"
                    ),
                    generated_images_dir_or_tensor=generated_samples,
                    num_cluster=20,
                )
                info_dict["entropy"] = entropy
                info_dict["cluster_count"] = cluster_count
                info_dict["cluster_proportions"] = cluster_proportions
            else:
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

        elif args.model_behavior == "local":

            full_model_dir = os.path.join(
                constants.OUTDIR, args.dataset, "retrain", "models", "full"
            )
            print(f"Loading full model checkpoint from {full_model_dir}")

            temp_method = args.method
            args.method = "retrain"
            args.trained_steps = None

            full_model, full_ema_model, _, _ = load_ckpt_model(args, full_model_dir)
            args.method = temp_method

            if args.use_ema:
                full_ema_model.copy_to(full_model.parameters())

            full_pipeline, vqvae, vqvae_latent_dict = build_pipeline(args, full_model)

            # Generate images with the same random noises as inputs.
            avg_mse_val, avg_nrmse_val, avg_ssim_val, avg_total_loss = 0, 0, 0, 0

            def generate_local_images(pipeline, device, random_seed, **pipeline_kwargs):
                """Generate numpy images from a pipeline."""
                pipeline = pipeline.to(device)
                images = pipeline(
                    generator=torch.Generator().manual_seed(random_seed),
                    output_type="numpy",
                    **pipeline_kwargs,
                ).images
                return images

            for random_seed in tqdm(range(args.n_samples)):
                info_prefix = f"generated_image_{random_seed}"
                full_image = generate_local_images(
                    pipeline=full_pipeline,
                    device=args.device,
                    random_seed=random_seed,
                    num_inference_steps=args.num_inference_steps,
                    batch_size=1,
                )
                removal_image = generate_local_images(
                    pipeline=pipeline,
                    device=args.device,
                    random_seed=random_seed,
                    num_inference_steps=args.num_inference_steps,
                    batch_size=1,
                )

                # Image similarity metrics as local model behaviors.
                mse_val = mean_squared_error(
                    image0=full_image[0], image1=removal_image[0]
                )
                nrmse_val = normalized_root_mse(
                    image_true=full_image[0], image_test=removal_image[0]
                )
                ssim_val = structural_similarity(
                    im1=full_image[0],
                    im2=removal_image[0],
                    channel_axis=-1,
                    data_range=1,
                )

                avg_mse_val += mse_val
                avg_nrmse_val += nrmse_val
                avg_ssim_val += ssim_val

                info_dict[f"{info_prefix}_mse"] = f"{mse_val:.8e}"
                info_dict[f"{info_prefix}_nrmse"] = f"{nrmse_val:.8e}"
                info_dict[f"{info_prefix}_ssim"] = f"{ssim_val:.8e}"

                # Diffusion loss at the discrete steps during inference as the
                # local model behavior.
                loss_fn = torch.nn.MSELoss(reduction="mean")
                full_image = (
                    torch.from_numpy(full_image).permute([0, 3, 1, 2]).to(args.device)
                )
                save_image(
                    full_image,
                    os.path.join(sample_outdir, f"generated_image_{random_seed}.png"),
                )
                pipeline = pipeline.to(args.device)
                pipeline.scheduler.set_timesteps(args.num_inference_steps)
                timesteps = pipeline.scheduler.timesteps.to(args.device)

                noise_generator = torch.Generator(device=args.device).manual_seed(
                    random_seed
                )

                with torch.no_grad():
                    total_loss = 0
                    if args.dataset == "celeba":
                        full_image = vqvae.encode(full_image, False)[0]
                        full_image = full_image * vqvae.config.scaling_factor

                    for _ in range(args.n_noises):
                        noises = torch.randn(
                            (timesteps.shape[0], *full_image.shape[1:]),
                            generator=noise_generator,
                            device=args.device,
                        )
                        noisy_full_images = pipeline.scheduler.add_noise(
                            full_image, noises, timesteps
                        )
                        pred_noises = pipeline.unet(noisy_full_images, timesteps).sample
                        total_loss += loss_fn(pred_noises, noises)
                    total_loss /= args.n_noises

                avg_total_loss += total_loss
                info_dict[f"{info_prefix}_diffusion_loss"] = f"{total_loss:.8e}"

            avg_mse_val /= args.n_samples
            avg_nrmse_val /= args.n_samples
            avg_ssim_val /= args.n_samples
            avg_total_loss /= args.n_samples

            info_dict["avg_mse"] = f"{avg_mse_val:.8e}"
            info_dict["avg_nrmse"] = f"{avg_nrmse_val:.8e}"
            info_dict["avg_ssim"] = f"{avg_ssim_val:.8e}"
            info_dict["avg_total_loss"] = f"{avg_total_loss:.8e}"

        info_dict["total_steps_time"] = total_steps_time
        info_dict["trained_steps"] = training_steps
        info_dict["remaining_idx"] = remaining_idx.tolist()
        info_dict["removed_idx"] = removed_idx.tolist()
        info_dict["device"] = str(args.device)
        info_dict["total_sampling_time"] = time.time() - behavior_start_time

        with open(args.db, "a+") as f:
            f.write(json.dumps(info_dict) + "\n")
        print(f"Results saved to the database at {args.db}")

        return accelerator.is_main_process


if __name__ == "__main__":
    args = parse_args()
    is_main_process = main(args)
    if is_main_process:
        print(f"Unlearning with {args.method} is done!")
