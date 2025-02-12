# Data Attribution via Sparsified Unlearning

This README provides instructions for training **unconditional** diffusion models, including methods such as retraining (exact unlearning) and **sparsified unlearning**.

## Quick Start Guide

### 1. Training a Diffusion Model
Train a diffusion model from scratch with optional removal distribution:

```bash
python main.py \
    --dataset [cifar|celeba] \
    --method retrain \
    --removal_dist [datashapley|datamodel|uniform|None] \
    --datamodel_alpha 0.5 \
    --removal_seed 0 \
    --mixed_precision [no|bf16|fp16] \
    --gradient_accumulation_steps 1
```

#### Optional Model Sparsification Parameters:
```bash
    --pruning_ratio 0.3 \
    --pruner magnitude \
    --thr 0.05
```

### 2. Training with CelebA-HQ (256x256)
For memory-efficient training on CelebA-HQ:

1. Set up `deepspeed_config_dp.yaml` for Accelerator
2. Precompute VQVAE latent embeddings:
```bash
accelerate launch --config_file deepspeed_config_dp.yaml \
    main.py \
    --dataset celeba \
    --method retrain \
    --mixed_precision fp16 \
    --use_8bit_optimizer \
    --precompute_stage save
```
3. Train using precomputed embeddings by changing `--precompute_stage save` to `reuse`

### 3. Model Pruning
Prune a trained model:
```bash
python prune.py \
    --load model_path \
    --pruning_ratio 0.3 \
    --pruner magnitude \
    --thr 0.05
```
> Note: Pruned models require fine-tuning (retraining)

### 4. Model Unlearning
Unlearn a full model and compute model behavior:
```bash
python unlearn.py \
    --load full_model_path \
    --dataset [cifar|celeba] \
    --db path_to_saved_results \
    --method [gd|ga|lora|iu] \
    --model_behavior global
```

#### Unlearning Parameters:
```bash
    --iu_ratio 0.5 \
    --ga_ratio 1.0 \
    --gd_steps 2000 \
    --lora_rank 16 \
    --lora_dropout 0.05
```

### 5. Computing Linear Datamodel Score (LDS)
```bash
python lds.py \
    --dataset [cifar|celeba] \
    --removal_dist [shapley|datamodel] \
    --model_behavior_key [is|diversity_score|aesthetic_score] \
    --train_db path_to_retrain/unlearn_model_behavior \
    --train_exp_name train_exp_name \
    --method [retrain|ga|lora|iu] \
    --full_db path_to_full_model_behavior \
    --null_db path_to_null_model_behavior \
    --test_db path_to_datamodel_behavior \
    --test_exp_name test_exp_name \
    --max_train_size 100 \
    --num_test_subset 100 \
    --by_class
```

## References
- [Understanding Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Lilian Weng's comprehensive blog
- [DDPM Paper](https://arxiv.org/pdf/2006.11239.pdf) - Original Denoising Diffusion Probabilistic Models paper
- [PyTorch DDPM Implementation](https://github.com/lucidrains/denoising-diffusion-pytorch)
