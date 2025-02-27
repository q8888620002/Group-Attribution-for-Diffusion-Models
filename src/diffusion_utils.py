"""Utilities for duffusion pipeline"""

import math
import os
from typing import List

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import diffusers
from diffusers import (
    DDIMPipeline,
    DDIMScheduler,
    DDPMPipeline,
    DiffusionPipeline,
    LDMPipeline,
)
from diffusers.training_utils import EMAModel
from src.datasets import create_dataset
from src.ddpm_config import DDPMConfig
from src.utils import get_max_steps


class ImagenetteCaptioner:
    """
    Class to caption Imagenette labels.

    Args:
    ----
        dataset: a torchvision ImageFolder dataset.
    """

    def __init__(self, dataset: ImageFolder):
        self.label_to_synset = dataset.classes  # List of synsets.
        self.num_classes = len(self.label_to_synset)
        self.synset_to_word = {
            "n01440764": "tench",
            "n02102040": "English springer",
            "n02979186": "cassette player",
            "n03000684": "chainsaw",
            "n03028079": "church",
            "n03394916": "French horn",
            "n03417042": "garbage truck",
            "n03425413": "gas pump",
            "n03445777": "golf ball",
            "n03888257": "parachute",
        }

    def __call__(self, labels: torch.LongTensor) -> List[str]:
        """
        Convert integer labels to string captions.

        Args:
        ----
            labels: Tensor of integer labels.

        Returns
        -------
            A list of string captions, with the format of "a photo of a {object}."
        """
        captions = []
        for label in labels:
            synset = self.label_to_synset[label]
            word = self.synset_to_word[synset]
            captions.append(f"a photo of a {word}.")
        return captions


class LabelTokenizer:
    """
    Class to convert integer labels to caption token ids.

    Args:
    ----
        captioner: A class that converts integer labels to string captions.
        tokenizer: A Hugging Face PreTrainedTokenizer.
    """

    def __init__(self, captioner: ImagenetteCaptioner, tokenizer: PreTrainedTokenizer):
        self.captioner = captioner
        self.tokenizer = tokenizer

    def __call__(self, labels: torch.LongTensor) -> torch.LongTensor:
        """
        Converts integer labels to caption token ids.

        Args:
        ----
            labels: Tensor of integer labels.

        Returns
        -------
            Integer tensor of token ids, with padding and truncation if necessary.
        """
        captions = self.captioner(labels)
        inputs = self.tokenizer(
            captions,
            max_length=self.tokenizer.model_max_length,
            padding="longest",
            truncation=True,
            return_tensors="pt",
        )
        return inputs.input_ids


def load_ckpt_model(args, model_loaddir):
    """
        Load model parameters from the latest checkpoint in a directory.
    Args:
    ----
        args: arguments from training pipeline
        model_loaddir: model path
    Return:
    ------
        pre-trained model, indices of remaining and removed subset.
    """

    if args.dataset == "cifar":
        config = {**DDPMConfig.cifar_config}
    elif args.dataset == "cifar2":
        config = {**DDPMConfig.cifar2_config}
    elif args.dataset == "cifar100":
        config = {**DDPMConfig.cifar100_config}
    elif args.dataset == "cifar100_new":
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
            (
                f"dataset={args.dataset} is not one of "
                "['cifar', 'mnist', 'celeba', 'imagenette']"
            )
        )
    model_cls = getattr(diffusers, config["unet_config"]["_class_name"])

    trained_steps = (
        args.trained_steps
        if args.trained_steps is not None
        else get_max_steps(model_loaddir)
    )

    if trained_steps is not None:
        ckpt_path = os.path.join(model_loaddir, f"ckpt_steps_{trained_steps:0>8}.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if args.method not in ["retrain", "gd_u"]:
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

        # There may not be saved indices for pretrained model.
        try:
            remaining_idx = ckpt["remaining_idx"].numpy().tolist()
            removed_idx = ckpt["removed_idx"].numpy().tolist()
        except KeyError:
            train_dataset = create_dataset(dataset_name=args.dataset, train=True)
            remaining_idx = np.arange(len(train_dataset))
            removed_idx = np.array([], dtype=int)

        model.load_state_dict(ckpt["unet"])
        model.eval()
        
        model_str = "U-Net"

        print(f"Trained {model_str} loaded from {ckpt_path}")

        ema_model = EMAModel(
            model.parameters(),
            model_cls=model_cls,
            model_config=model.config,
        )
        ema_model.load_state_dict(ckpt["unet_ema"])
        model_str = "EMA"

        print(f"\t{model_str} loaded from {ckpt_path}")
    else:
        raise ValueError(f"No trained checkpoints found at {model_loaddir}")

    return model, ema_model, remaining_idx, removed_idx


def build_pipeline(args, model):
    """Build the diffusion pipeline for the sepcific dataset and U-Net model."""
    # Get the diffusion model pipeline for inference.

    device = args.device

    if args.dataset == "imagenette":
        # The pipeline is of class LDMTextToImagePipeline.
        train_dataset = create_dataset(dataset_name=args.dataset, train=True)
        pipeline = DiffusionPipeline.from_pretrained(
            "CompVis/ldm-text2im-large-256"
        ).to(device)
        pipeline.unet = model.to(device)

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
        pipeline = DiffusionPipeline.from_pretrained("CompVis/ldm-celebahq-256").to(
            device
        )
        pipeline.unet = model.to(device)

        vqvae = pipeline.vqvae
        pipeline.vqvae.config.scaling_factor = 1
        vqvae.requires_grad_(False)

        if args.precompute_stage is None:
            # Move the VQ-VAE model to the device without any operations.
            vqvae = vqvae.to(device)
            vqvae_latent_dict = None

        elif args.precompute_stage == "save":
            assert (
                args.removal_dist is None
            ), "Precomputation should be done for full data"
            # Precompute and save the VQ-VAE latents
            vqvae = vqvae.to(device)
            vqvae.train()  # The vqvae output is STATIC even in train mode.

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

            print(
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
        pipeline = DDPMPipeline(unet=model, scheduler=DDIMScheduler()).to(args.device)
        vqvae = None
        captioner = None
        vqvae_latent_dict = None

    return pipeline, vqvae, vqvae_latent_dict


def generate_images(args, pipeline):
    """Generate numpy images from a pipeline."""

    results = []

    batch_size_list = [args.batch_size] * (args.n_samples // args.batch_size)
    remaining_sample_size = args.n_samples % args.batch_size

    if remaining_sample_size > 0:
        batch_size_list.append(remaining_sample_size)

    if args.dataset != "imagenette":
        # For unconditional diffusion models.
        with torch.no_grad():
            counter = 0
            for batch_size in tqdm(batch_size_list):
                noise_generator = torch.Generator().manual_seed(counter)
                images = pipeline(
                    batch_size=batch_size,
                    num_inference_steps=args.num_inference_steps,
                    output_type="numpy",
                    generator=noise_generator,
                ).images

                counter += 1
                for image in images:
                    image = torch.from_numpy(image).permute([2, 0, 1])
                    # Align with image saving process in generate_samples.py
                    permuted_image = (
                        image.mul(255)
                        .add_(0.5)
                        .clamp_(0, 255)
                        .permute(1, 2, 0)
                        .to("cpu", torch.uint8)
                        .numpy()
                    )
                    results.append(transforms.ToTensor()(permuted_image))

    return torch.stack(results).float()


def run_inference(
    accelerator,
    model,
    ema_model,
    config,
    args,
    vqvae,
    captioner,
    pipeline,
    pipeline_scheduler,
):
    """Wrapper function for inference. To be run under the accelerator main process."""
    model = accelerator.unwrap_model(model).eval()
    ema_model.store(model.parameters())
    ema_model.copy_to(model.parameters())  # The EMA is used for inference.

    with torch.no_grad():
        if args.dataset == "imagenette":
            samples = []
            n_samples_per_cls = math.ceil(config["n_samples"] / captioner.num_classes)
            classes = [idx for idx in range(captioner.num_classes)]
            for _ in range(n_samples_per_cls):
                samples.append(
                    pipeline(
                        prompt=captioner(classes),
                        num_inference_steps=args.num_inference_steps,
                        eta=0.3,
                        guidance_scale=6,
                        output_type="numpy",
                    ).images
                )
            samples = np.concatenate(samples)
        elif args.dataset == "celeba":
            pipeline = LDMPipeline(
                unet=model,
                vqvae=vqvae,
                scheduler=pipeline_scheduler,
            ).to(accelerator.device)
            samples = pipeline(
                batch_size=config["n_samples"],
                num_inference_steps=args.num_inference_steps,
                output_type="numpy",
            ).images
        else:
            pipeline = DDIMPipeline(
                unet=model,
                scheduler=DDIMScheduler(num_train_timesteps=args.num_train_steps),
            )
            samples = pipeline(
                batch_size=config["n_samples"],
                num_inference_steps=args.num_inference_steps,
                output_type="numpy",
            ).images

        samples = torch.from_numpy(samples).permute([0, 3, 1, 2])
        ema_model.restore(model.parameters())
    return samples
