"""Calcaulate local model behaviors."""

# TODO: Implement conditional generation for Imagenette Text LDM.
import argparse
import json
import os

import torch
from skimage.metrics import (
    mean_squared_error,
    normalized_root_mse,
    structural_similarity,
)
from torchvision.utils import save_image
from tqdm import tqdm

import diffusers
import src.constants as constants
from diffusers import DDIMPipeline, DDIMScheduler, DiffusionPipeline
from diffusers.training_utils import EMAModel
from src.ddpm_config import DDPMConfig
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


def build_pipeline(dataset, model):
    """Build the diffusion pipeline for the sepcific dataset and U-Net model."""
    if args.dataset == "imagenette":
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-text2im-large-256")
        pipeline.unet = model
    elif args.dataset == "celeba":
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256")
        pipeline.vqvae.config.scaling_factor = 1
        pipeline.unet = model
    else:
        pipeline = DDIMPipeline(unet=model, scheduler=DDIMScheduler())
    return pipeline


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
        "--removal_model_dir",
        type=str,
        help="directory path of model checkpoints with data removal",
        default=None,
    )
    parser.add_argument(
        "--removal_model_steps",
        type=int,
        help="steps for the removal model",
        default=None,
    )
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
        choices=constants.METHOD,
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
    args = parser.parse_args()
    return args


def main(args):
    """Main function to calculate local model behaviors."""
    info_dict = vars(args)

    # Load model architectures.
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
        raise ValueError(
            (f"dataset={args.dataset} is not one of " f"{constants.DATASET}")
        )
    model_cls = getattr(diffusers, config["unet_config"]["_class_name"])
    full_model = model_cls(**config["unet_config"])

    if args.method in ["retrain", "gd_u"]:
        # Use the model architecture from the config file.
        removal_model = model_cls(**config["unet_config"])
        print("Removal model architecture loaded from DDPMConfig")
    else:
        # Use the pruned model architecure. The pruned model's weights are also loaded
        # and should be replaced.
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
        removal_model = pruned_model_ckpt["unet"]
        print(f"Removal model architecture loaded from {pruned_model_path}")

    # Get removal model directory.
    removal_model_dir = args.removal_model_dir
    if removal_model_dir is None:
        removal_dir = "full"
        if args.excluded_class is not None:
            removal_dir = f"excluded_{args.excluded_class}"
        if args.removal_dist is not None:
            removal_dir = f"{args.removal_dist}/{args.removal_dist}"
            if args.removal_dist == "datamodel":
                removal_dir += f"_alpha={args.datamodel_alpha}"
            removal_dir += f"_seed={args.removal_seed}"
        removal_model_dir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            "models",
            removal_dir,
        )
        info_dict["removal_model_dir"] = removal_model_dir

    sample_outdir = os.path.join(
        args.outdir,
        args.dataset,
        "local_scores",
        "ema_generated_samples" if args.use_ema else "generated_samples",
    )
    os.makedirs(sample_outdir, exist_ok=True)

    # Load full and removal model checkpoints.
    print("Loading full model checkpoint...")
    load_model_ckpt(
        full_model,
        args.full_model_dir,
        args.use_ema,
        args.full_model_steps,
    )

    print("Loading removal model checkpoint...")
    removal_ckpt = load_model_ckpt(
        removal_model,
        removal_model_dir,
        args.use_ema,
        args.removal_model_steps,
        return_ckpt=True,
    )
    info_dict["remaining_idx"] = removal_ckpt["remaining_idx"].numpy().tolist()
    info_dict["removed_idx"] = removal_ckpt["removed_idx"].numpy().tolist()

    # Construct the diffusion pipeline.
    full_pipeline = build_pipeline(args.dataset, full_model)
    removal_pipeline = build_pipeline(args.dataset, removal_model)

    if args.dataset == "celeba":
        vqvae = full_pipeline.vqvae

    # Generate images with the same random noises as inputs.
    avg_mse_val, avg_nrmse_val, avg_ssim_val, avg_total_loss = 0, 0, 0, 0

    for random_seed in tqdm(range(args.n_samples)):
        info_prefix = f"generated_image_{random_seed}"

        full_image = generate_images(
            pipeline=full_pipeline,
            device=args.device,
            random_seed=random_seed,
            num_inference_steps=args.num_inference_steps,
            batch_size=1,
        )

        removal_image = generate_images(
            pipeline=removal_pipeline,
            device=args.device,
            random_seed=random_seed,
            num_inference_steps=args.num_inference_steps,
            batch_size=1,
        )

        # Image similarity metrics as local model behaviors.
        mse_val = mean_squared_error(image0=full_image[0], image1=removal_image[0])
        nrmse_val = normalized_root_mse(
            image_true=full_image[0], image_test=removal_image[0]
        )
        ssim_val = structural_similarity(
            im1=full_image[0], im2=removal_image[0], channel_axis=-1, data_range=1
        )

        avg_mse_val += mse_val
        avg_nrmse_val += nrmse_val
        avg_ssim_val += ssim_val

        info_dict[f"{info_prefix}_mse"] = f"{mse_val:.8e}"
        info_dict[f"{info_prefix}_nrmse"] = f"{nrmse_val:.8e}"
        info_dict[f"{info_prefix}_ssim"] = f"{ssim_val:.8e}"

        # Diffusion loss at the discrete steps during inference as the local model
        # behavior.
        loss_fn = torch.nn.MSELoss(reduction="mean")
        full_image = torch.from_numpy(full_image).permute([0, 3, 1, 2]).to(args.device)
        save_image(
            full_image,
            os.path.join(sample_outdir, f"generated_image_{random_seed}.png"),
        )
        removal_pipeline = removal_pipeline.to(args.device)
        removal_pipeline.scheduler.set_timesteps(args.num_inference_steps)
        timesteps = removal_pipeline.scheduler.timesteps.to(args.device)

        noise_generator = torch.Generator(device=args.device).manual_seed(random_seed)

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
                noisy_full_images = removal_pipeline.scheduler.add_noise(
                    full_image, noises, timesteps
                )
                pred_noises = removal_pipeline.unet(noisy_full_images, timesteps).sample
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

    with open(args.db, "a+") as f:
        f.write(json.dumps(info_dict) + "\n")
    print(f"Results saved to the database at {args.db}")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
