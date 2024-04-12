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

import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from lightning.pytorch import seed_everything
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

import src.constants as constants
from diffusers.optimization import get_scheduler
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
        choices=["iu", "ga"],
    )
    parser.add_argument(
        "--iu_ratio", type=float, help="ratio for purturbing model weights", default=1.0
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
                args.by_class =True
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

    model, ema_model = load_ckpt_model(args)

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

    training_steps = len(removed_dataloader)

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

    if args.method == "iu":
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

    elif args.method == "ga":

        training_steps = training_steps // 2
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

    # The EMA is used for inference.

    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())
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
                sample_outdir,
                f"prutirb_ratio_{args.iu_ratio}_steps_{training_steps:0>8}.png",
            ),
            nrow=int(math.sqrt(config["n_samples"])),
        )
        print(f"Save test images, steps_{training_steps:0>8}.png, in {sample_outdir}.")
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

        info_dict["trained_steps"] = training_steps
        info_dict["remaining_idx"] = remaining_idx.tolist()
        info_dict["removed_idx"] = removed_idx.tolist()
        info_dict["device"] = str(args.device)

        with open(args.db, "a+") as f:
            f.write(json.dumps(info_dict) + "\n")
        print(f"Results saved to the database at {args.db}")

        return accelerator.is_main_process


if __name__ == "__main__":
    args = parse_args()
    is_main_process = main(args)
    if is_main_process:
        print("Influence unlearning done!")
