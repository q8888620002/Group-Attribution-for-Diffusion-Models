"""Class for TRAK score calculation."""
import argparse
import os

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

import constants
from ddpm_config import DDPMConfig
from utils import (
    ImagenetteCaptioner,
    LabelTokenizer,
    create_dataset,
    get_max_steps,
    remove_data_by_class,
    remove_data_by_datamodel,
    remove_data_by_shapley,
    remove_data_by_uniform,
)


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
        choices=["mnist", "cifar", "celeba", "imagenette"],
        default="mnist",
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
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "gd", "ga", "esd"],
        required=True,
    )
    parser.add_argument(
        "--f",
        type=str,
        default=None,
        choices=[
            "mean",
            "mean-squared-l2-norm",
            "l1-norm",
            "l2-norm",
            "linf-norm",
        ],
        help="TBD",
    )

    parser.add_argument(
        "--t_strategy",
        type=str,
        default=None,
        help="strategy for sampling time steps",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=None,
        help="TBD",
    )

    return parser.parse_args()


def count_parameters(model):
    """Helper function that return the sum of parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def vectorize_and_ignore_buffers(g, params_dict=None):
    """
    Gradients are given as a tuple :code:`(grad_w0, grad_w1, ... grad_wp)` where
    :code:`p` is the number of weight matrices. each :code:`grad_wi` has shape
    :code:`[batch_size, ...]` this function flattens :code:`g` to have shape
    :code:`[batch_size, num_params]`.
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
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
    elif args.dataset == "imagenette":
        config = {**DDPMConfig.imagenette_config}
    else:
        raise ValueError(
            (
                f"dataset={args.dataset} is not one of "
                "['cifar', 'mnist', 'celeba', 'imagenette']"
            )
        )

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
    config["batch_size"] = 32

    remaining_dataloader = DataLoader(
        Subset(train_dataset, remaining_idx),
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=1,
    )
    existing_steps = get_max_steps(model_outdir)

    unet_path = os.path.join(model_outdir, f"unet_steps_{existing_steps:0>8}.pt")
    # unet_ema_path = os.path.join(
    #     model_outdir, f"unet_ema_steps_{existing_steps:0>8}.pt"
    # )
    model = torch.load(unet_path, map_location=device)

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

    save_dir = os.path.join(args.outdir, args.dataset, removal_dir, "d_track")
    # Initialize random matrix projector from trak

    projector = CudaProjector(
        grad_dim=count_parameters(model),
        proj_dim=2048,
        seed=42,
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

    if args.f == "mean-squared-l2-norm":
        print(args.f)

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

    elif args.f == "mean":
        print(args.f)

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

    elif args.f == "l1-norm":
        print(args.f)

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

    elif args.f == "l2-norm":
        print(args.f)

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

    elif args.f == "linf-norm":
        print(args.f)

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
        print(args.f)

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

    for step, (image, label) in enumerate(remaining_dataloader):

        seed_everything(42, workers=True)
        image = image.to(device)
        bsz = image.shape[0]

        if args.t_strategy == "uniform":
            selected_timesteps = range(0, 1000, 1000 // args.K)
        elif args.t_strategy == "cumulative":
            selected_timesteps = range(0, args.K)

        for index_t, t in enumerate(selected_timesteps):
            # Sample a random timestep for each image
            timesteps = torch.tensor([t] * bsz, device=image.device)
            timesteps = timesteps.long()
            seed_everything(42 * 1000 + t)  # !!!!

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

        emb = emb / args.K
        print(emb.size())

        # If is_grads_dict == True, then turn emb into a dict.
        # emb_dict = {k: v for k, v in zip(keys, emb)}

        emb = projector.project(emb, is_grads_dict=False, model_id=0)  # ddpm
        print(emb.size())
        print(emb.dtype)


if __name__ == "__main__":
    args = parse_args()
    main(args)
