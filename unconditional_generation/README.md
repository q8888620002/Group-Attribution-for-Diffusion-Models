# Data Attribution via Sparsified Unlearning

This README provides instructions for training diffusion models, including from retraining (exact unlearning) and **sparsified unlearning**, as well as steps for converting pre-trained checkpoints checkpoints for use with DDPMpipeline on Huggingface.

### Training a Diffusion Model from Scratch
To train a diffusion model from scratch, use the following command:
```bash
python main.py --dataset [dataset] --method [unlearning/retrain]
```

#### Efficient Training
For those utilizing consumer-grade GPUs with limited memory capacity (e.g., RTX 2080ti), we offer several features aimed at efficient training:
* precomputed VQVAE latent
* 8bit Adam optimizer (we use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package)
* data parallelism (we use [deepspeed](https://huggingface.co/docs/accelerate/v0.27.2/en/usage_guides/deepspeed#deepspeed-config-file) packafge)
* gradient accumulation

The above techniques are also useful for higher-end GPUs, as they allow for faster training and larger batch sizes, which can contribute to improved model performance.

Below are example commands:

First, run the following command to precompute the VQVAE latents for the training dataset.
```bash
accelerate launch --config_file deepspeed_config_dp.yaml \
main.py --dataset celeba \
--method retrain \
--mixed_precision fp16 \
--use_8bit_optimizer \
--precompute_stage save
```

Then, run the following command to train the model using the precomputed latents.
```bash
accelerate launch --config_file deepspeed_config_dp.yaml \
main.py --dataset celeba \
--method retrain \
--mixed_precision fp16 \
--use_8bit_optimizer \
--precompute_stage reuse
```

### Training with Unlearning with a removal distribution
For training with the unlearning method, use this command:
```bash
python main.py --dataset [dataset] --method [unlearning] --removal_dist [removal_dist]
```
Replace `[dataset]`, `[method]`, and `[removal_dist]` with appropriate values.

## Development Guidlines
1. Git pre-commit hooks (https://pre-commit.com/) are used to automatically
check and fix formatting errors before a Git commit happens. This ensure format
consistency, which reduces the number of lines of code to review. Run
`pre-commit install` to install all the hooks.

## References and Additional Resources

- **Understanding Diffusion Models:** A comprehensive explanation can be found in this blog: [Lilian Weng's Blog on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
- **Original DDPM Paper:** The Denoising Diffusion Probabilistic Models paper is available at [arXiv](https://arxiv.org/pdf/2006.11239.pdf).
- **PyTorch Implementation of DDPM:** A PyTorch version of DDPM can be found on GitHub: [Denoising Diffusion PyTorch Repository](https://github.com/lucidrains/denoising-diffusion-pytorch).
