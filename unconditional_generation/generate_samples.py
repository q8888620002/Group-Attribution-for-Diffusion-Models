"""Generate samples for a given diffusion model."""

import argparse
import os

import diffusers
import torch
from diffusers import DDIMPipeline, DDIMScheduler, DiffusionPipeline
from diffusers.training_utils import EMAModel
from lightning.pytorch import seed_everything
from torchvision.utils import save_image
from tqdm import tqdm

import src.constants as constants
from src.datasets import create_dataset
from src.ddpm_config import DDPMConfig
from src.diffusion_utils import ImagenetteCaptioner
from src.utils import get_max_epochs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Training DDPM")

    parser.add_argument(
        "--dataset",
        type=str,
        help="dataset for training or unlearning",
        choices=["mnist", "cifar2", "cifar", "celeba", "imagenette"],
        default="cifar",
    )
    parser.add_argument(
        "--n_samples", type=int, default=100000, help="number of generated samples"
    )
    parser.add_argument(
        "--batch_size", type=int, default=512, help="batch size for sample generation"
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
        "--trained_epochs",
        type=int,
        help="epochs for specific ckeck points",
        default=None,
    )
    parser.add_argument(
        "--method",
        type=str,
        help="training or unlearning method",
        choices=["retrain", "prune_fine_tune", "gd", "ga", "esd"],
        required=True,
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
        "--seed",
        type=int,
        help="random seed for image sample generation",
        default=42,
    )
    parser.add_argument(
        "--outdir", type=str, help="output parent directory", default=constants.OUTDIR
    )
    parser.add_argument(
        "--pretrained",
        help="whether to generate samples for a pre-trained model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=100,
        help="number of diffusion steps for generating images",
    )
    parser.add_argument(
        "--use_ema",
        help="whether to use the EMA model",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--device", type=str, help="device of training", default="cuda:0"
    )

    return parser.parse_args()


def main(args):
    """Main function to generate samples from a diffusion model."""
    seed_everything(args.seed)
    device = args.device

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
                "['cifar','cifar2', 'mnist', 'celeba', 'imagenette']"
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

    if args.pretrained:
        model_loaddir = os.path.join(
            args.outdir,
            args.dataset,
            "pretrained",
            "models",
        )
        sample_outdir = os.path.join(
            args.outdir,
            args.dataset,
            "pretrained",
            "ema_generated_samples" if args.use_ema else "generated_samples",
        )
    else:
        model_loaddir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            "models",
            removal_dir,
        )
        sample_outdir = os.path.join(
            args.outdir,
            args.dataset,
            args.method,
            "ema_generated_samples" if args.use_ema else "generated_samples",
            removal_dir,
        )
    os.makedirs(sample_outdir, exist_ok=True)

    # Load the trained U-Net model or U-Net EMA.

    trained_epochs = (
        args.trained_epochs
        if args.trained_epochs is not None
        else get_max_epochs(model_loaddir)
    )
    # sample_outdir = os.path.join(sample_outdir, trained_steps)

    if trained_epochs is not None:
        ckpt_path = os.path.join(model_loaddir, f"ckpt_epochs_{trained_epochs:0>5}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")

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
                f"ckpt_epochs_{1:0>5}.pt",
            )
            pruned_model_ckpt = torch.load(pruned_model_path, map_location="cpu")
            model = pruned_model_ckpt["unet"]
        else:
            model = model_cls(**config["unet_config"])

        model.load_state_dict(ckpt["unet"])

        model_str = "U-Net"

        if args.use_ema:
            ema_model = EMAModel(
                model.parameters(),
                model_cls=model_cls,
                model_config=model.config,
            )
            ema_model.load_state_dict(ckpt["unet_ema"])
            ema_model.copy_to(model.parameters())
            model_str = "EMA"

        print(f"Trained model loaded from {model_loaddir}")
        print(f"\t{model_str} loaded from {ckpt_path}")
    else:
        raise ValueError(f"No trained checkpoints found at {model_loaddir}")

    # Get the diffusion model pipeline for inference.
    if args.dataset == "imagenette":
        # The pipeline is of class LDMTextToImagePipeline.
        train_dataset = create_dataset(dataset_name=args.dataset, train=True)
        captioner = ImagenetteCaptioner(train_dataset)

        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/ldm-text2im-large-256"
        ).to(device)
        pipeline.unet = model.to(device)
    elif args.dataset == "celeba":
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").to(
            device
        )
        pipeline.vqvae.config.scaling_factor = 1
        pipeline.unet = model.to(device)
    else:
        pipeline = DDIMPipeline(unet=model, scheduler=DDIMScheduler()).to(device)

    # Generate images.
    batch_size_list = [args.batch_size] * (args.n_samples // args.batch_size)
    remaining_sample_size = args.n_samples % args.batch_size
    if remaining_sample_size > 0:
        batch_size_list.append(remaining_sample_size)

    if args.dataset != "imagenette":
        # For unconditional diffusion models.
        counter = 0
        with torch.no_grad():
            for batch_size in tqdm(batch_size_list):
                images = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                ).images
                for image in images:
                    save_image(
                        torch.from_numpy(image).permute([2, 0, 1]),
                        os.path.join(
                            sample_outdir, f"seed={args.seed}_sample_{counter}.png"
                        ),
                    )
                    counter += 1
        print(f"Generated {counter} samples and saved to {sample_outdir}")
    else:
        # Conditoinal generation for each class in Imagenette.
        for class_idx in range(captioner.num_classes):
            synset = captioner.label_to_synset[class_idx]
            synset_sample_outdir = os.path.join(sample_outdir, synset)
            os.makedirs(synset_sample_outdir, exist_ok=True)
            counter = 0
            with torch.no_grad():
                for batch_size in tqdm(batch_size_list):
                    images = pipeline(
                        prompt=captioner([class_idx] * batch_size),
                        num_inference_steps=args.num_inference_steps,
                        eta=0.3,
                        guidance_scale=6,
                        output_type="numpy",
                    ).images
                    for image in images:
                        save_image(
                            torch.from_numpy(image).permute([2, 0, 1]),
                            os.path.join(
                                synset_sample_outdir,
                                f"seed={args.seed}_sample_{counter}.png",
                            ),
                        )
                        counter += 1
            print(f"Generated {counter} samples and saved to {synset_sample_outdir}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("Sample generation completed!")
