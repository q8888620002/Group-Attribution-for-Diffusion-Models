"""Class for TRAK score calculation."""
import argparse
import os

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
from diffusers import (
    DDIMScheduler,
    DDPMPipeline,
    DDPMScheduler,
    DiffusionPipeline,
    LDMPipeline,
    VQModel,
)
from lightning.pytorch import seed_everything
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Subset
from trak.projectors import CudaProjector, ProjectionType
from trak.utils import is_not_buffer

import src.constants as constants
from src.datasets import (
    create_dataset,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)
from src.ddpm_config import DDPMConfig
from src.diffusion_utils import ImagenetteCaptioner, LabelTokenizer
from src.utils import get_max_steps


def parse_args():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(
        description="Calculating gradient for D-TRAK and TRAK."
    )
    parser.add_argument(
        "--opt_seed",
        type=int,
        help="random seed for model training or unlearning",
        default=42,
    )
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
        default=None,
    )
    parser.add_argument(
        "--device", type=str, help="device of training", default="cuda:0"
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
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
        "--model_behavior",
        type=str,
        choices=[
            "loss",
            "mean",
            "mean-squared-l2-norm",
            "l1-norm",
            "l2-norm",
            "linf-norm",
        ],
        default=None,
        required=True,
        help="Specification for D-TRAK model behavior.",
    )

    parser.add_argument(
        "--t_strategy",
        type=str,
        choices=["uniform", "cumulative"],
        help="strategy for sampling time steps",
    )
    parser.add_argument(
        "--k_partition",
        type=int,
        default=None,
        help="Partition for embeddings across time steps.",
    )
    parser.add_argument(
        "--projector_dim",
        type=int,
        default=1024,
        help="Dimension for TRAK projector",
    )

    return parser.parse_args()


def count_parameters(model):
    """Helper function that return the sum of parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vectorize_and_ignore_buffers(g, params_dict=None):
    """
    Flattens and concatenates gradients from multiple weight matrices into a single tensor.

    Args:
    -------
        g (tuple of torch.Tensor): Gradients for each weight matrix, each with shape [batch_size, ...].
        params_dict (dict, optional): Dictionary to identify non-buffer gradients in 'g'.

    Returns
    -------
    torch.Tensor:
        Tensor with shape [batch_size, num_params], where each row represents flattened and
        concatenated gradients for a single batch instance. 'num_params' is the total count of
        flattened parameters across all weight matrices.

    Note:
    - If 'params_dict' is provided, only non-buffer gradients are processed.
    - The output tensor is formed by flattening each gradient tensor and concatenating them.
    """
    batch_size = len(g[0])
    out = []
    if params_dict is not None:
        for b in range(batch_size):
            out.append(
                torch.cat(
                    [
                        x[b].flatten()
                        for i, x in enumerate(g)
                        if is_not_buffer(i, params_dict)
                    ]
                )
            )
    else:
        for b in range(batch_size):
            out.append(torch.cat([x[b].flatten() for x in g]))
    return torch.stack(out)


def main(args):
    """Main function for computing project@gradient for D-TRAK and TRAK."""

    device = args.device

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
            (f"dataset={args.dataset} is not one of " f"{constants.DATASET}")
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

    model_outdir = os.path.join(
        args.outdir, args.dataset, "retrain", "models", removal_dir
    )

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

    config["batch_size"] = 8

    remaining_dataloader = DataLoader(
        Subset(train_dataset, remaining_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
    )
    existing_steps = get_max_steps(model_outdir)

    ## load full model

    ckpt_path = os.path.join(model_outdir, f"ckpt_steps_{existing_steps:0>8}.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = model_cls(**config["unet_config"])
    model.load_state_dict(ckpt["unet"])

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

    save_dir = os.path.join(
        args.outdir,
        args.dataset,
        "d_track",
        removal_dir,
        f"f={args.model_behavior}_t={args.t_strategy}",
    )

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # Init a memory-mapped array stored on disk directly for D-TRAK results.

    dstore_keys = np.memmap(
        save_dir,
        dtype=np.float32,
        mode="w+",
        shape=(len(remaining_idx), args.projector_dim),
    )

    # Initialize random matrix projector from trak
    projector = CudaProjector(
        grad_dim=count_parameters(model),
        proj_dim=args.projector_dim,
        seed=args.opt_seed,
        proj_type=ProjectionType.normal,  # proj_type=ProjectionType.rademacher,
        device=device,
        max_batch_size=config["batch_size"],
    )

    params = {
        k: v.detach() for k, v in model.named_parameters() if v.requires_grad is True
    }
    buffers = {
        k: v.detach() for k, v in model.named_buffers() if v.requires_grad is True
    }

    if args.model_behavior == "mean-squared-l2-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=noisy_latents,
                kwargs={
                    "timestep": timesteps,
                },
            )
            predictions = predictions.sample
            ####
            # predictions = predictions.reshape(1, -1)
            # f = torch.norm(predictions.float(), p=2.0, dim=-1)**2 # squared
            # f = f/predictions.size(1) # mean
            # f = f.mean()
            ####
            f = F.mse_loss(
                predictions.float(), torch.zeros_like(targets).float(), reduction="none"
            )
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    elif args.model_behavior == "mean":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=noisy_latents,
                kwargs={
                    "timestep": timesteps,
                },
            )
            predictions = predictions.sample
            ####
            f = predictions.float()
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    elif args.model_behavior == "l1-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=noisy_latents,
                kwargs={
                    "timestep": timesteps,
                },
            )
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=1.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    elif args.model_behavior == "l2-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=noisy_latents,
                kwargs={
                    "timestep": timesteps,
                },
            )
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=2.0, dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    elif args.model_behavior == "linf-norm":
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=noisy_latents,
                kwargs={
                    "timestep": timesteps,
                },
            )
            predictions = predictions.sample
            ####
            predictions = predictions.reshape(1, -1)
            f = torch.norm(predictions.float(), p=float("inf"), dim=-1)
            f = f.mean()
            ####
            # print(f.size())
            # print(f)
            ####
            return f

    else:
        print(args.model_behavior)

        def compute_f(params, buffers, noisy_latents, timesteps, targets):
            noisy_latents = noisy_latents.unsqueeze(0)
            timesteps = timesteps.unsqueeze(0)
            targets = targets.unsqueeze(0)

            predictions = functional_call(
                model,
                (params, buffers),
                args=noisy_latents,
                kwargs={
                    "timestep": timesteps,
                },
            )
            predictions = predictions.sample
            ####
            f = F.mse_loss(predictions.float(), targets.float(), reduction="none")
            f = f.reshape(1, -1)
            f = f.mean()
            ####
            return f

    ft_compute_grad = grad(compute_f)
    ft_compute_sample_grad = vmap(
        ft_compute_grad,
        in_dims=(
            None,
            None,
            0,
            0,
            0,
        ),
    )

    for step, (image, _) in enumerate(remaining_dataloader):

        seed_everything(args.opt_seed, workers=True)
        image = image.to(device)
        bsz = image.shape[0]

        if args.t_strategy == "uniform":
            selected_timesteps = range(0, 1000, 1000 // args.k_partition)
        elif args.t_strategy == "cumulative":
            selected_timesteps = range(0, args.k_partition)

        for index_t, t in enumerate(selected_timesteps):
            # Sample a random timestep for each image
            timesteps = torch.tensor([t] * bsz, device=image.device)
            timesteps = timesteps.long()
            seed_everything(args.opt_seed * 1000 + t)  # !!!!

            noise = torch.randn_like(image)
            noisy_latents = pipeline_scheduler.add_noise(image, noise, timesteps)

            # Get the target for loss depending on the prediction type
            if pipeline_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif pipeline_scheduler.config.prediction_type == "v_prediction":
                target = pipeline_scheduler.get_velocity(image, noise, timesteps)
            else:
                raise ValueError(
                    f"Unknown prediction type {pipeline_scheduler.config.prediction_type}"
                )

            ft_per_sample_grads = ft_compute_sample_grad(
                params,
                buffers,
                noisy_latents,
                timesteps,
                target,
            )
            # if len(keys) == 0:
            #     keys = ft_per_sample_grads.keys()

            ft_per_sample_grads = vectorize_and_ignore_buffers(
                list(ft_per_sample_grads.values())
            )

            # print(ft_per_sample_grads.size())
            # print(ft_per_sample_grads.dtype)

            if index_t == 0:
                emb = ft_per_sample_grads
            else:
                emb += ft_per_sample_grads
            # break

        emb = emb / args.k_partition
        print(emb.size())

        # If is_grads_dict == True, then turn emb into a dict.
        # emb_dict = {k: v for k, v in zip(keys, emb)}

        emb = projector.project(emb, model_id=0)
        print(emb.size())
        print(emb.dtype)

        while (
            np.abs(
                dstore_keys[
                    step * config["batch_size"] : step * config["batch_size"] + bsz,
                    0:32,
                ]
            ).sum()
            == 0
        ):
            print("saving")
            dstore_keys[
                step * config["batch_size"] : step * config["batch_size"] + bsz
            ] = (emb.detach().cpu().numpy())
        print(f"{step} / {len(remaining_dataloader)}, {t}")
        print(step * config["batch_size"], step * config["batch_size"] + bsz)


if __name__ == "__main__":
    args = parse_args()
    main(args)
