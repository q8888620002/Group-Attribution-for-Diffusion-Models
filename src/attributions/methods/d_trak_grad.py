"""Class for TRAK score calculation."""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from lightning.pytorch import seed_everything
from torch.func import functional_call, grad, vmap
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from tqdm import tqdm
from trak.projectors import CudaProjector, ProjectionType
from trak.utils import is_not_buffer

import diffusers
import src.constants as constants
from diffusers import DDPMPipeline, DDPMScheduler, DiffusionPipeline
from src.datasets import (
    ImageDataset,
    TensorDataset,
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
        "--method",
        type=str,
        help="training or unlearning method",
        choices=constants.METHOD,
        required=True,
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
            "loss",  # TRAK
            "mean",
            "mean-squared-l2-norm",  # D-TRAK
            "l1-norm",
            "l2-norm",
            "linf-norm",
            "ssim",
            "fid",
            "nrmse",
            "is",
        ],
        default=None,
        required=True,
        help="Specification for D-TRAK model behavior.",
    )
    parser.add_argument(
        "--model_behavior_value",
        type=float,
        default=None,
        help="Model output for a pre-calculated model behavior e.g. FID, SSIM, IS.",
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
    parser.add_argument(
        "--sample_dir",
        type=str,
        default=None,
        help="filepath of sample (generated) images ",
    )
    parser.add_argument(
        "--calculate_gen_grad",
        help="whether to generate validation set and calculate phi",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        help="number of generated images to consider for local model behaviors",
        default=None,
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
        g (tuple of torch.Tensor):
            Gradients for each weight matrix, each with shape [batch_size, ...].
        params_dict (dict, optional):
            Dictionary to identify non-buffer gradients in 'g'.

    Returns
    -------
    torch.Tensor:
        Tensor with shape [batch_size, num_params], where each row represents
        flattened and concatenated gradients for a single batch instance.
        'num_params' is the total count of flattened parameters across all weight matrices.

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
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "cifar2":
        config = {**DDPMConfig.cifar2_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "cifar100":
        config = {**DDPMConfig.cifar100_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "cifar100_f":
        config = {**DDPMConfig.cifar100_f_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "cifar100_new":
        config = {**DDPMConfig.cifar100_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 32, 32).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
    elif args.dataset == "celeba":
        config = {**DDPMConfig.celeba_config}
        example_inputs = {
            "sample": torch.randn(1, 3, 64, 64).to(device),
            "timestep": torch.ones((1,)).long().to(device),
        }
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

    if args.sample_dir is None:
        if not args.calculate_gen_grad:
            n_samples = len(train_dataset)

            remaining_dataloader = DataLoader(
                Subset(train_dataset, remaining_idx),
                batch_size=config["batch_size"],
                shuffle=True,
                num_workers=1,
                pin_memory=True,
            )

            save_dir = os.path.join(
                args.outdir,
                args.dataset,
                "d_trak",
                removal_dir,
                f"train_f={args.model_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
            )
        else:
            save_dir = os.path.join(
                args.outdir,
                args.dataset,
                "d_trak",
                removal_dir,
                f"gen_f={args.model_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
            )

    else:

        preprocess = transforms.Compose(
            [
                transforms.ToTensor(),  # Normalize to [0,1].
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                ),  # Normalize to [-1,1].
            ]
        )
        sample_dataset = ImageDataset(args.sample_dir, preprocess)
        n_samples = len(sample_dataset)
        remaining_dataloader = DataLoader(
            sample_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=1,
            pin_memory=True,
        )

        save_dir = os.path.join(
            args.sample_dir,
            "d_trak",
            f"reference_f={args.model_behavior}_t={args.t_strategy}_k={args.k_partition}_d={args.projector_dim}",
        )

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)
    existing_steps = get_max_steps(model_outdir)

    # load full model

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
        # The pipeline is of class LDMPipeline.
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        pipeline.unet = model

        pipeline.scheduler.set_timesteps(
            config["scheduler_config"]["num_train_timesteps"]
        )
        vqvae = pipeline.vqvae
        pipeline.vqvae.config.scaling_factor = 1
        vqvae.requires_grad_(False)

        if args.sample_dir is None:
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
        else:
            vqvae = vqvae.to(device)

        pipeline.to(device)

    else:
        pipeline = DDPMPipeline(
            unet=model, scheduler=DDPMScheduler(**config["scheduler_config"])
        ).to(device)

    pipeline_scheduler = pipeline.scheduler

    # Init a memory-mapped array stored on disk directly for D-TRAK results.

    if args.calculate_gen_grad:
        # Generate samples for Journey TRAK
        generated_samples = []
        n_samples = args.n_samples

        pipeline.scheduler.num_train_steps = 1000
        pipeline.scheduler.num_inference_steps = 100

        for random_seed in tqdm(range(n_samples)):
            noise_latents = []
            noise_generator = torch.Generator(device=args.device).manual_seed(
                random_seed
            )

            with torch.no_grad():
                noises = torch.randn(
                    example_inputs["sample"].shape,
                    generator=noise_generator,
                    device=args.device,
                )
                input = noises

                for t in range(999, -1, -1000 // args.k_partition):
                    noise_latents.append(input.squeeze(0).detach().cpu())

                    noisy_residual = pipeline.unet(input, t).sample
                    previous_noisy_image = pipeline.scheduler.step(
                        noisy_residual, t, input
                    ).prev_sample
                    input = previous_noisy_image

                noise_latents = torch.stack(noise_latents[::-1]) #  flip the order so noise_latents[0] gives us the final image
            generated_samples.append(noise_latents)
        generated_samples = torch.stack(generated_samples)
        bogus_labels = torch.zeros(n_samples, dtype=torch.int)
        images_dataset = TensorDataset(
            generated_samples, transform=None, label=bogus_labels
        )

        remaining_dataloader = DataLoader(
            images_dataset,
            batch_size=config["batch_size"],
            num_workers=1,
            pin_memory=True,
        )

    dstore_keys = np.memmap(
        save_dir,
        dtype=np.float32,
        mode="w+",
        shape=(n_samples, args.projector_dim),
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
    starttime = time.time()
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
    for step, batch in enumerate(remaining_dataloader):

        seed_everything(args.opt_seed, workers=True)

        image, label = batch[0], batch[1]
        image = image.to(device)
        bsz = image.shape[0]

        if args.dataset == "celeba" and not args.calculate_gen_grad:
            if args.sample_dir is None:  # Compute TRAK with pre-computed embeddings.
                imageid = batch[2]
                image = torch.stack(
                    [vqvae_latent_dict[imageid[i]] for i in range(len(image))]
                ).to(device)
            else:  # Directly encode the images if there's no precomputation
                image = vqvae.encode(image, False)[0]
            image = image * vqvae.config.scaling_factor

        if args.t_strategy == "uniform":
            selected_timesteps = range(0, 1000, 1000 // args.k_partition)
        elif args.t_strategy == "cumulative":
            selected_timesteps = range(0, args.k_partition)

        for index_t, t in enumerate(selected_timesteps):
            # Sample a random timestep for each image
            timesteps = torch.tensor([t] * bsz, device=image.device)
            timesteps = timesteps.long()
            seed_everything(args.opt_seed * 1000 + t)  # !!!!

            if args.calculate_gen_grad:
                noisy_latents = image[:, index_t, :, :, :]
                noise = torch.randn_like(noisy_latents)
            else:
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

    print("total_time", time.time() - starttime)

if __name__ == "__main__":
    args = parse_args()
    main(args)
