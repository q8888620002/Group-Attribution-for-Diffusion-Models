# Data Attribution via Sparsified Unlearning

This README provides instructions for training **unconditional** diffusion models, including from retraining (exact unlearning) and **sparsified unlearning**. 

### Training a Diffusion Model from Scratch  and unlearning with a removal distribution
To train a diffusion model from scratch, use the following command:
```bash
python main.py --dataset [dataset] --method [unlearning/retrain/prune_fine_tune/gd/ga] \
## Removal distribution args
--removal_dist [datashapley/datamodel/uniform/None] \
--datamodel_alpha [0.5] \
--removal_seed [0]

## Model sparsification args. This is needed when training unlearned models
--pruning_ratio [0.3]\
--pruner [magnitude]\
--thr [0.05]\

## Training args
--keep_all_ckpts \

## Accelerator args
--mixed_precision [no, bf16, fp16]
--gradient_accumulation_steps [1]
```

#### Efficient Training for CelebA-HQ (256x 256)
For those utilizing consumer-grade GPUs with limited memory capacity (e.g., RTX 2080ti), we offer several features aimed at efficient training:
* precomputed VQVAE latent
* 8bit Adam optimizer (we use [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) package)
* data parallelism (we use [deepspeed](https://huggingface.co/docs/accelerate/v0.27.2/en/usage_guides/deepspeed#deepspeed-config-file) package)
* gradient accumulation

The above techniques are also useful for higher-end GPUs, as they allow for faster training and larger batch sizes, which can contribute to improved model performance.

Below are example commands:
1. Set up the configuration file for [Accelerator](https://huggingface.co/docs/accelerate/en/package_reference/accelerator)
2. Run the following command to precompute the VQVAE latent embeddings.

```bash
accelerate launch --config_file deepspeed_config_dp.yaml \
main.py --dataset celeba \
--method retrain \
--mixed_precision fp16 \
--use_8bit_optimizer \
--precompute_stage save
```
3. Change `--precompute_stage reuse` to retrain with the precomputed latent embeddings.

## References and Additional Resources

- **Understanding Diffusion Models:** A comprehensive explanation can be found in this blog: [Lilian Weng's Blog on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/).
- **Original DDPM Paper:** The Denoising Diffusion Probabilistic Models paper is available at [arXiv](https://arxiv.org/pdf/2006.11239.pdf).
- **PyTorch Implementation of DDPM:** A PyTorch version of DDPM can be found on GitHub: [Denoising Diffusion PyTorch Repository](https://github.com/lucidrains/denoising-diffusion-pytorch).
