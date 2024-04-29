"""Compare images generated by two LoRA models."""
import argparse
import json
import os
import sys
import time

import clip
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from skimage.metrics import normalized_root_mse, structural_similarity
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
from transformers import CLIPTokenizer

from diffusers import DDPMScheduler, DiffusionPipeline
from src.aesthetics import get_aesthetic_model
from src.ddpm_config import PromptConfig
from src.utils import fix_get_processor, print_args


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare images generated by two LoRA models."
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="lambdalabs/miniSD-diffusers",
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help=(
            "Variant of the model files of the pretrained model identifier from "
            "huggingface.co/models, 'e.g.' fp16"
        ),
    )
    parser.add_argument(
        "--reference_lora_dir",
        type=str,
        default=None,
        help="directory for reference LoRA weights",
        required=True,
    )
    parser.add_argument(
        "--lora_dir",
        type=str,
        default=None,
        help="directory containing LoRA weights to load",
    )
    parser.add_argument(
        "--lora_steps",
        type=int,
        default=None,
        help="number of steps the LoRA weights have been trained on",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["artbench"],
        default="artbench",
        help="Dataset to determine which prompts to use for image generation",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=50,
        help="number of images to generate for computing model behaviors",
    )
    parser.add_argument(
        "--n_noises",
        type=int,
        default=3,
        help="number of noise samples per time step when calculating diffusion losses",
    )
    parser.add_argument(
        "--ckpt_freq",
        type=int,
        default=10,
        help="number of images before saving a checkpoint",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default=None,
        help="filepath for saving the checkpoint",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        default=None,
        help="directory path for saving the generated images",
    )
    parser.add_argument(
        "--clean_up_ckpt",
        default=False,
        action="store_true",
        help="whether to clean up the checkpoint once the process is finished",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=256,
        help="the resolution of generated image",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="seed for reproducible image generation",
    )
    parser.add_argument(
        "--cls",
        type=str,
        default="post_impressionism",
        help="generate images for this class",
    )
    parser.add_argument(
        "--db",
        type=str,
        help="path to database for saving the results",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        help="experiment name to save in the database",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--no_duplicate",
        action="store_true",
        help="whether to avoid running a process that produces duplicate records",
    )
    return parser.parse_args()


def load_pipeline(args):
    """Load diffusion model pipeline."""
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        revision=args.revision,
        variant=args.variant,
    )
    pipeline.safety_checker = None
    pipeline.requires_safety_checker = False
    pipeline.set_progress_bar_config(disable=True)
    pipeline = pipeline.to("cuda")
    return pipeline


def main(args):
    """Main function."""
    # Check for duplicate record in the database.
    results_dict = vars(args)
    if args.db is not None and os.path.exists(args.db) and args.no_duplicate:
        df = pd.read_json(args.db, lines=True)
        query_dict = {
            key: val
            for key, val in results_dict.items()
            if key not in ["no_duplicate", "ckpt_freq"]  # Keys irrelevant to results.
        }
        for key, val in query_dict.items():
            if df.shape[0] == 0:
                has_record = False
                break
            if val is None:
                df = df[df[key].isna()]  # None is converted to NaN when saving.
            else:
                df = df[df[key] == val]
        has_record = df.shape[0] > 0
        if has_record:
            print(
                f"Found duplicate record in database at {args.db}. Process cancelled."
            )
            sys.exit(0)  # Exit without raising an error.

    # Set up the prompt.
    prompt_dict = {"artbench": PromptConfig.artbench_config}
    prompt_dict = prompt_dict[args.dataset]
    prompt = prompt_dict[args.cls]

    # Load diffusion pipelines and their generators for reproducibility.
    reference_pipeline = load_pipeline(args)
    reference_pipeline.unet.load_attn_procs(
        args.reference_lora_dir, weight_name="pytorch_lora_weights.safetensors"
    )
    reference_generator = torch.Generator(device="cuda")
    reference_generator.manual_seed(args.seed)

    pipeline = load_pipeline(args)
    remaining_idx, removal_idx = None, None
    if args.lora_dir is not None:
        weight_name = "pytorch_lora_weights"
        if args.lora_steps is not None:
            weight_name += f"_{args.lora_steps}"
        weight_name += ".safetensors"
        fix_get_processor(pipeline.unet)  # Runtime bugfix.
        pipeline.unet.load_attn_procs(args.lora_dir, weight_name=weight_name)

        removal_idx_file = os.path.join(args.lora_dir, "removal_idx.csv")
        if os.path.exists(removal_idx_file):
            removal_idx_df = pd.read_csv(removal_idx_file)
            print(f"Removal index file loaded from {removal_idx_file}")
            remaining_idx = removal_idx_df["idx"][removal_idx_df["remaining"]].to_list()
            removal_idx = removal_idx_df["idx"][~removal_idx_df["remaining"]].to_list()
    else:
        print("Pretrained model is loaded")
    generator = torch.Generator(device="cuda")
    generator.manual_seed(args.seed)
    noise_generator = torch.Generator(device="cuda")
    noise_generator.manual_seed(args.seed)

    noise_scheduler = DDPMScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        revision=args.revision,
        variant=args.variant,
    )
    tokenizer = CLIPTokenizer.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=args.revision,
        variant=args.variant,
    )

    # Load CLIP for computing CLIP similarity.
    clip_model, clip_preprocess = clip.load("ViT-B/32", device="cuda")

    ssim_list, ssim_time_list = [], []
    nrmse_list, nrmse_time_list = [], []
    clip_similarity_list, clip_similarity_time_list = [], []
    clip_prompt_score_list, clip_prompt_score_time_list = [], []
    aesthetic_score_list, aesthetic_score_time_list = [], []
    simple_loss_list, simple_loss_time_list = [], []
    num_completed_images = 0

    # Load the aesthetic model and the corresponding CLIP for aesthetic scoring.
    aesthetic_model = get_aesthetic_model(clip_model="vit_l_14")
    aesthetic_model = aesthetic_model.to("cuda")
    (
        open_clip_model,
        _,
        open_clip_preprocess,
    ) = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
    open_clip_model = open_clip_model.to("cuda")

    # Load existing checkpoint.
    if args.ckpt_path is not None and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path)
        reference_generator.set_state(ckpt["reference_generator_state"])
        generator.set_state(ckpt["generator_state"])
        noise_generator.set_state(ckpt["noise_generator_state"])

        ssim_list = ckpt["ssim_list"]
        nrmse_list = ckpt["nrmse_list"]
        clip_similarity_list = ckpt["clip_similarity_list"]
        clip_prompt_score_list = ckpt["clip_prompt_score_list"]
        aesthetic_score_list = ckpt["aesthetic_score_list"]
        simple_loss_list = ckpt["simple_loss_list"]

        ssim_time_list = ckpt["ssim_time_list"]
        nrmse_time_list = ckpt["nrmse_time_list"]
        clip_similarity_time_list = ckpt["clip_similarity_time_list"]
        clip_prompt_score_time_list = ckpt["clip_prompt_score_time_list"]
        aesthetic_score_time_list = ckpt["aesthetic_score_time_list"]
        simple_loss_time_list = ckpt["simple_loss_time_list"]

        num_completed_images = ckpt["num_completed_images"]
        print(f"Checkpoint loaded from {args.ckpt_path}")

    if args.img_dir is not None:
        os.makedirs(args.img_dir, exist_ok=True)

    # Pre-computation.
    with torch.no_grad():
        # For diffusion loss.
        prompt_input_ids = tokenizer(
            [prompt],
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to("cuda")

        # For CLIP prompt score.
        clip_prompt_embedding = clip_model.encode_text(
            clip.tokenize([prompt]).to("cuda")
        )
        clip_prompt_embedding /= clip_prompt_embedding.norm(dim=-1, keepdim=True)
        clip_prompt_embedding = clip_prompt_embedding.squeeze()

    # Compute model behaviors.
    progress_bar = tqdm(initial=num_completed_images, total=args.num_images)
    for i in range(num_completed_images, args.num_images):
        reference_img = reference_pipeline(
            prompt,
            num_inference_steps=100,
            generator=reference_generator,
            height=args.resolution,
            width=args.resolution,
        ).images[0]

        generation_time = time.time()
        img = pipeline(
            prompt,
            num_inference_steps=100,
            generator=generator,
            height=args.resolution,
            width=args.resolution,
        ).images[0]
        generation_time = time.time() - generation_time

        if args.img_dir is not None:
            reference_img.save(
                os.path.join(
                    args.img_dir, f"reference_img_seed={args.seed}_sample_{i}.jpg"
                )
            )
            img.save(os.path.join(args.img_dir, f"img_seed={args.seed}_sample_{i}.jpg"))

        # SSIM.
        ssim_time_list.append(time.time())
        ssim_list.append(
            structural_similarity(
                im1=np.array(reference_img),
                im2=np.array(img),
                channel_axis=-1,
                data_range=255,
            )
        )
        ssim_time_list[-1] = time.time() - ssim_time_list[-1] + generation_time

        # Normalized root mean squared error.
        nrmse_time_list.append(time.time())
        nrmse_list.append(
            normalized_root_mse(
                image_true=np.array(reference_img), image_test=np.array(img)
            )
        )
        nrmse_time_list[-1] = time.time() - nrmse_time_list[-1] + generation_time

        # CLIP similarity.
        clip_similarity_time_list.append(time.time())
        with torch.no_grad():
            reference_clip_embedding = clip_model.encode_image(
                clip_preprocess(reference_img).unsqueeze(0).to("cuda")
            )
            reference_clip_embedding /= reference_clip_embedding.norm(
                dim=-1, keepdim=True
            )
            reference_clip_embedding = reference_clip_embedding.squeeze()

            clip_embedding = clip_model.encode_image(
                clip_preprocess(img).unsqueeze(0).to("cuda")
            )
            clip_embedding /= clip_embedding.norm(dim=-1, keepdim=True)
            clip_embedding = clip_embedding.squeeze()

            clip_similarity = torch.dot(reference_clip_embedding, clip_embedding)
            clip_similarity_list.append(clip_similarity.item())
        clip_similarity_time_list[-1] = (
            time.time() - clip_similarity_time_list[-1] + generation_time
        )

        # CLIP score for the prompt.
        clip_prompt_score_time_list.append(time.time())
        with torch.no_grad():
            clip_prompt_score = torch.dot(clip_embedding, clip_prompt_embedding)
            clip_prompt_score_list.append(clip_prompt_score.item())
        clip_prompt_score_time_list[-1] = (
            time.time() - clip_prompt_score_time_list[-1] + generation_time
        )

        # Simple diffusion loss.
        simple_loss_time_list.append(time.time())
        with torch.no_grad():
            timesteps = pipeline.scheduler.timesteps.to("cuda")
            encoder_hidden_states = pipeline.text_encoder(prompt_input_ids)[0]
            encoder_hidden_states = encoder_hidden_states.expand(
                timesteps.size(0), -1, -1
            )
            reference_img = to_tensor(reference_img).unsqueeze(0).to("cuda")

            simple_loss = 0
            for _ in range(args.n_noises):
                latents = pipeline.vae.encode(reference_img).latent_dist.sample(
                    generator=noise_generator
                )
                latents = latents.expand(timesteps.size(0), -1, -1, -1)
                latents = latents * pipeline.vae.config.scaling_factor
                noise = torch.randn(
                    latents.size(), generator=noise_generator, device="cuda"
                )
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                preds = pipeline.unet(
                    noisy_latents, timesteps, encoder_hidden_states
                ).sample
                simple_loss += F.mse_loss(preds, noise, reduction="mean")
            simple_loss /= args.n_noises
        simple_loss_list.append(simple_loss.item())
        simple_loss_time_list[-1] = time.time() - simple_loss_time_list[-1]

        # Aesthetic score.
        aesthetic_score_time_list.append(time.time())
        with torch.no_grad():
            open_clip_embedding = open_clip_model.encode_image(
                open_clip_preprocess(img).to("cuda").unsqueeze(0)
            )
            open_clip_embedding /= open_clip_embedding.norm(dim=-1, keepdim=True)
            aesthetic_score = aesthetic_model(open_clip_embedding)
            aesthetic_score_list.append(aesthetic_score.item())
        aesthetic_score_time_list[-1] = (
            time.time() - aesthetic_score_time_list[-1] + generation_time
        )

        num_completed_images += 1
        progress_bar.update(1)

        if (args.ckpt_path is not None) and (
            num_completed_images % args.ckpt_freq == 0
        ):
            ckpt = {
                "reference_generator_state": reference_generator.get_state(),
                "generator_state": generator.get_state(),
                "noise_generator_state": noise_generator.get_state(),
                "ssim_list": ssim_list,
                "ssim_time_list": ssim_time_list,
                "nrmse_list": nrmse_list,
                "nrmse_time_list": nrmse_time_list,
                "simple_loss_list": simple_loss_list,
                "simple_loss_time_list": simple_loss_time_list,
                "clip_similarity_list": clip_similarity_list,
                "clip_similarity_time_list": clip_similarity_time_list,
                "clip_prompt_score_list": clip_prompt_score_list,
                "clip_prompt_score_time_list": clip_prompt_score_time_list,
                "aesthetic_score_list": aesthetic_score_list,
                "aesthetic_score_time_list": aesthetic_score_time_list,
                "num_completed_images": num_completed_images,
            }
            torch.save(ckpt, args.ckpt_path)
            print(f"Checkpoint saved tp {args.ckpt_path}")

    # Save results to the database.
    if args.db is not None:
        # Local model behaviors.
        for i in range(num_completed_images):
            prefix = f"generated_image_{i}"
            results_dict[prefix + "_ssim"] = ssim_list[i]
            results_dict[prefix + "_nrmse"] = nrmse_list[i]
            results_dict[prefix + "_simple_loss"] = simple_loss_list[i]
            results_dict[prefix + "_clip_similarity"] = clip_similarity_list[i]
            results_dict[prefix + "_clip_prompt_score"] = clip_prompt_score_list[i]
            results_dict[prefix + "_aesthetic_score"] = aesthetic_score_list[i]

            results_dict[prefix + "_ssim_time"] = ssim_time_list[i]
            results_dict[prefix + "_nrmse_time"] = nrmse_time_list[i]
            results_dict[prefix + "_simple_loss_time"] = simple_loss_time_list[i]
            results_dict[prefix + "_clip_similarity_time"] = clip_similarity_time_list[
                i
            ]
            results_dict[
                prefix + "_clip_prompt_score_time"
            ] = clip_prompt_score_time_list[i]
            results_dict[prefix + "_aesthetic_score_time"] = aesthetic_score_time_list[
                i
            ]

        # Global model behaviors.
        for q in [0.5, 0.75, 0.9]:
            results_dict[f"aesthetic_score_{q}"] = np.quantile(
                aesthetic_score_list, q=q
            )
            results_dict[f"clip_prompt_score_{q}"] = np.quantile(
                clip_prompt_score_list, q=q
            )
        results_dict["aesthetic_score_avg"] = np.mean(aesthetic_score_list)
        results_dict["clip_prompt_score_avg"] = np.mean(clip_prompt_score_list)
        results_dict["aesthetic_score_time"] = np.sum(aesthetic_score_time_list)
        results_dict["clip_prompt_score_time"] = np.sum(clip_prompt_score_time_list)

        results_dict["remaining_idx"] = remaining_idx
        results_dict["removal_idx"] = removal_idx
        with open(args.db, "a+") as f:
            f.write(json.dumps(results_dict) + "\n")
        print(f"Results saved to the database at {args.db}")

    if (
        args.ckpt_path is not None
        and os.path.exists(args.ckpt_path)
        and args.clean_up_ckpt
    ):
        os.remove(args.ckpt_path)
        print(f"Checkpoint at {args.ckpt_path} removed")
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    print_args(args)
    main(args)
