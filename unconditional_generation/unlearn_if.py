"""
Perform influence unlearning based on [1]

[1]: https://github.com/OPTML-Group/Unlearn-Sparse/blob/public/unlearn/Wfisher.py
"""

import argparse
import glob
import math
import os
import time

import diffusers
import numpy as np
import torch
import torch.nn as nn
from accelerate import Accelerator
from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from lightning.pytorch import seed_everything
from torch.autograd import grad
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image
from tqdm import tqdm

import src.constants as constants
from src.datasets import (
    create_dataset,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)
from src.ddpm_config import DDPMConfig
from src.diffusion_utils import ImagenetteCaptioner, LabelTokenizer, run_inference
from src.unlearn.Wfisher import get_grad
from src.utils import get_max_steps, print_args


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
    return parser.parse_args()


def sam_grad(model, loss):
    """Function to change model weights"""
    params = []
    for param in model.parameters():
        params.append(param)
    sample_grad = grad(loss, params)
    sample_grad = [x.view(-1) for x in sample_grad]
    return torch.cat(sample_grad)


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

    if args.method == "prune_fine_tune":

        model_outdir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            "models",
            (f"pruner={args.pruner}" + f"_pruning_ratio={args.pruning_ratio}"),
            removal_dir,
        )
    else:
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

    total_steps_time = 0
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

    remaining_dataloader = DataLoader(
        Subset(train_dataset, remaining_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=4,
        generator=torch.Generator().manual_seed(args.opt_seed),
    )

    # Round up with math.ceil to ensure all batches are used in each epoch.
    num_update_steps_per_epoch = math.ceil(
        len(remaining_dataloader) / args.gradient_accumulation_steps
    )
    training_steps = num_update_steps_per_epoch * config["training_epochs"][args.method]

    current_epochs = math.ceil(param_update_steps / num_update_steps_per_epoch)

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
    if existing_steps is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        accelerator.print(f"Optimizer and lr scheduler resumed from {ckpt_path}")

    loss_fn = nn.MSELoss(reduction="mean")

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

    # Unlearning with influence unlearning

    params = []
    for param in model.parameters():
        params.append(param.view(-1))
    forget_grad = torch.zeros_like(torch.cat(params)).to(device)
    retain_grad = torch.zeros_like(torch.cat(params)).to(device)

    total = 0
    model.eval()

    if args.dataset == "celeba" and args.precompute_stage == "reuse":
        total = get_grad(args, removed_dataloader, pipeline, vqvae_latent)
        total_2 = get_grad(args, removed_dataloader, pipeline, vqvae_latent)
    else:
        total = get_grad(args, removed_dataloader, pipeline)
        total_2 = get_grad(args, removed_dataloader, pipeline)

    retain_grad *= total / ((total + total_2) * total_2)
    forget_grad /= total + total_2

    k_vec = torch.clone(retain_grad - forget_grad)

    for idx, batch_r in enumerate(remaining_dataloader):
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
        eps_r = model(noisy_images_r, timesteps).sample
        loss = loss_fn(eps_r, noise)

        sample_grad = sam_grad(model, loss)
        if idx == 0:
            o_vec = torch.clone(sample_grad)
        else:
            tmp = torch.dot(o_vec, sample_grad)
            k_vec -= (
                torch.dot(k_vec, sample_grad) / (len(remaining_dataloader) + tmp)
            ) * o_vec
            o_vec -= (tmp / (len(remaining_dataloader) + tmp)) * o_vec

    # Apply purturbation to model params.

    curr = 0
    for param in model.parameters():
        length = param.view(-1).shape[0]
        param.view(-1).data += k_vec[curr : curr + length].data
        curr += length

    # Save checkpoints.
    if (
        current_epochs % config["ckpt_freq_epochs"][args.method] == 0
        and param_update_steps % num_update_steps_per_epoch == 0
    ) and accelerator.is_main_process:
        if not args.keep_all_ckpts:
            pattern = os.path.join(model_outdir, "ckpt_epochs_*.pt")
            for filename in glob.glob(pattern):
                os.remove(filename)

        torch.save(
            {
                "unet": accelerator.unwrap_model(model).state_dict(),
                "unet_ema": ema_model.state_dict(),
                "remaining_idx": torch.from_numpy(remaining_idx),
                "removed_idx": torch.from_numpy(removed_idx),
                "total_steps_time": total_steps_time,
            },
            os.path.join(model_outdir, f"ckpt_epochs_{current_epochs:0>5}.pt"),
        )
        print(f"Checkpoint saved epoch {current_epochs}")

    # Generate samples for evaluation. This is done only once for the
    # main process.
    if accelerator.is_main_process:
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

    return accelerator.is_main_process


if __name__ == "__main__":
    args = parse_args()
    is_main_process = main(args)
    if is_main_process:
        print("Model optimization done!")
