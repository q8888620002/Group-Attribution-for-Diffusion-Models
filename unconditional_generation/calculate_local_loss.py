"""Calcaulate local model behaviors."""
import argparse
import json
import os

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
from tqdm import tqdm

import diffusers
import src.constants as constants
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
from src.diffusion_utils import build_pipeline, load_ckpt_model
from src.utils import get_max_steps, print_args


def load_model_ckpt(model, ckpt_dir, use_ema, steps=None, return_ckpt: bool = False):
    """Load model parameters from the latest checkpoint in a directory."""

    steps = get_max_steps(ckpt_dir) if steps is None else steps

    if steps is None:
        raise RuntimeError(f"No checkpoints found in {ckpt_dir}")
    ckpt_path = os.path.join(ckpt_dir, f"ckpt_steps_{steps:0>8}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["unet"])
    model_str = "U-Net"
    if use_ema:
        ema_model = EMAModel(model.parameters())
        ema_model.load_state_dict(ckpt["unet_ema"])
        ema_model.copy_to(model.parameters())
        model_str = "EMA"
    print(f"{model_str} parameters loaded from {ckpt_path}")
    if return_ckpt:
        return ckpt


def generate_images(pipeline, device, random_seed, **pipeline_kwargs):
    """Generate numpy images from a pipeline."""
    pipeline = pipeline.to(device)
    images = pipeline(
        generator=torch.Generator().manual_seed(random_seed),
        output_type="numpy",
        **pipeline_kwargs,
    ).images
    return images


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="calculate local model behaviors")
    parser.add_argument(
        "--full_model_dir",
        type=str,
        help="directory path of model checkpoints trained with the full training set",
        required=True,
    )
    parser.add_argument(
        "--full_model_steps",
        type=int,
        help="steps for the full model",
        default=None,
    )
    parser.add_argument(
        "--load",
        type=str,
        help="directory path for loading pre-trained model",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--outdir", type=str, help="results parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=constants.DATASET,
        default="cifar",
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
        choices=["iu", "ga", "gd", "gd_u"],
    )
    parser.add_argument(
        "--iu_ratio", type=float, help="ratio for purturbing model weights", default=1.0
    )
    parser.add_argument(
        "--ga_ratio", type=float, help="ratio for unlearning steps", default=2.0
    )
    parser.add_argument(
        "--gd_steps", type=int, help="ratio for unlearning steps", default=4000
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
        "--db",
        type=str,
        help="filepath of database for recording model behaviors",
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="experiment name to record in the database file",
        default=None,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated samples for computing local model behaviors",
        default=100,
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
        help="number of diffusion steps for generating images",
        default=100,
    )
    parser.add_argument(
        "--device", type=str, help="device used for computation", default="cuda:0"
    )
    parser.add_argument(
        "--use_ema",
        help="whether to use the EMA model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate before a backward/update pass.",
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
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
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
    args = parser.parse_args()
    return args


def main(args):
    """Main function to calculate local model behaviors."""
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
    model_cls = getattr(diffusers, config["unet_config"]["_class_name"])
    full_model = model_cls(**config["unet_config"])

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

    pipeline_scheduler = pipeline.scheduler

    if not args.use_8bit_optimizer:
        optimizer_kwargs = config["optimizer_config"]["kwargs"]

        if args.method == "ga":
            optimizer_kwargs["lr"] = 1e-5
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

    if args.method == "ga":
        training_steps = len(removed_dataloader)
    else:
        training_steps = len(remaining_dataloader)

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

    if args.method in ["gd", "gd_u"]:
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
                        ema_model.step(model.parameters())
                        param_update_steps += 1
                        progress_bar.update(1)

                if param_update_steps == args.gd_steps:
                    break

        model = accelerator.unwrap_model(model).eval()

    print("Loading full model checkpoint...")
    load_model_ckpt(
        full_model,
        args.full_model_dir,
        args.use_ema,
        args.full_model_steps,
    )
    # Construct the diffusion pipeline.
    full_pipeline, vqvae, vqvae_latent_dict= build_pipeline(args, full_model)

    print("Building removal model.. pipeline")
    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
    pipeline.unet = model

    if args.dataset == "celeba":
        vqvae = full_pipeline.vqvae

    # Generate images with the same random noises as inputs.
    avg_mse_val, avg_nrmse_val, avg_ssim_val, avg_total_loss = 0, 0, 0, 0

    example_image = generate_images(
        pipeline=full_pipeline,
        device=args.device,
        random_seed=args.opt_seed,
        num_inference_steps=args.num_inference_steps,
        batch_size=1,
    )

    loss_fn = torch.nn.MSELoss(reduction="mean")
    removal_pipeline = pipeline.to(args.device)
    removal_pipeline.scheduler.set_timesteps(args.num_inference_steps)
    timesteps = removal_pipeline.scheduler.timesteps.to(args.device)

    for random_seed in tqdm(range(args.n_samples)):
        info_prefix = f"generated_image_{random_seed}"
        noise_generator = torch.Generator(device=args.device).manual_seed(random_seed)

        with torch.no_grad():
            total_loss = 0

            if args.dataset == "celeba":
                example_image = vqvae.encode(example_image, False)[0]
                example_image = example_image * vqvae.config.scaling_factor

            noises = torch.randn(
                torch.from_numpy(example_image).permute(0, 3, 1, 2).shape,
                generator=noise_generator,
                device=args.device,
            )
            full_input = noises
            removal_input = noises

            for t in timesteps:
                full_noisy_residual = full_pipeline.unet(full_input, t).sample
                previous_full_sample = full_pipeline.scheduler.step(
                    full_noisy_residual, t, full_input
                ).prev_sample
                full_input = previous_full_sample

                removal_noisy_residual = removal_pipeline.unet(removal_input, t).sample
                previous_noisy_sample = full_pipeline.scheduler.step(
                    removal_noisy_residual, t, removal_input
                ).prev_sample
                removal_input = previous_noisy_sample

                loss_val = loss_fn(full_noisy_residual, removal_noisy_residual)

                removal_image = (removal_input / 2 + 0.5).clamp(0, 1).squeeze()
                removal_image = (
                    (removal_image.permute(1, 2, 0) * 255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
                full_image = (full_input / 2 + 0.5).clamp(0, 1).squeeze()
                full_image = (
                    (full_image.permute(1, 2, 0) * 255)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )

                mse_val = mean_squared_error(image0=full_image, image1=removal_image)
                nrmse_val = normalized_root_mse(
                    image_true=full_image, image_test=removal_image
                )
                ssim_val = structural_similarity(
                    im1=full_image, im2=removal_image, channel_axis=-1, data_range=1
                )

                avg_mse_val += mse_val
                avg_nrmse_val += nrmse_val
                avg_ssim_val += ssim_val
                avg_total_loss += loss_val

                info_dict[f"{info_prefix}_{t}_mse"] = f"{mse_val:.8e}"
                info_dict[f"{info_prefix}_{t}_nrmse"] = f"{nrmse_val:.8e}"
                info_dict[f"{info_prefix}_{t}_ssim"] = f"{ssim_val:.8e}"
                info_dict[f"{info_prefix}_{t}_diffusion_loss"] = f"{loss_val:.8e}"
    import ipdb;ipdb.set_trace()
    #     info_dict[f"{info_prefix}_diffusion_loss"] = f"{total_loss:.8e}"

    # avg_total_loss /= args.n_noises*len(timesteps)
    # avg_mse_val /= args.n_samples
    # avg_nrmse_val /= args.n_samples
    # avg_ssim_val /= args.n_samples
    # avg_total_loss /= args.n_samples

    # info_dict["avg_mse"] = f"{avg_mse_val:.8e}"
    # info_dict["avg_nrmse"] = f"{avg_nrmse_val:.8e}"
    # info_dict["avg_ssim"] = f"{avg_ssim_val:.8e}"
    # info_dict["avg_total_loss"] = f"{avg_total_loss:.8e}"

    info_dict["device"] = str(info_dict["device"])
    with open(args.db, "a+") as f:
        f.write(json.dumps(info_dict) + "\n")
    print(f"Results saved to the database at {args.db}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
