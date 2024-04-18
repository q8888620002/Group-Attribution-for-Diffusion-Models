
# Command line and codes for running celeba experiment

## Installation

```bash
export PYTHONPATH="$PYTHONPATH:$PWD"
```

```bash
conda create -n data_attribution python=3.11.5
conda activate data_attribution
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt # If torch is already installed, remove torch and torchvision from requirements.txt
```

<!-- ## Train -->

<!-- ```bash
python main.py
--dataset [cifar,cifar2, celeba ] \
--method [unlearning/retrain/prune_fine_tune/gd/ga] \
--removal_dist [datashapley/datamodel/uniform/None] \
--datamodel_alpha [0.5] \--removal_seed [0] \ 
--pruning_ratio [0.3]\
--pruner [magnitude]\
--thr [0.05]\
--keep_all_ckpts \
--mixed_precision [no, bf16, fp16] \ 
--gradient_accumulation_steps [1] \
``` -->

## Prepare training data

Code for preparing the training data is available in the `unconditional_generation/celeba_experiments/dataset_download.ipynb` jupyter notebook.

## Full model training

1. Precompute the VQVAE model outputs by running

```bash
CUDA_VISIBLE_DEVICES=7 \
python unconditional_generation/main.py \
--dataset celeba \
--method retrain \
--keep_all_ckpts \
--mixed_precision fp16 \
--gradient_accumulation_steps 1 \
--precompute_stage save
```

2. Train the model by running

```bash
accelerate launch --config_file unconditional_generation/celeba_experiments/deepspeed_config_dp.yaml --gpu_ids 6,7 \
unconditional_generation/main.py \
--dataset celeba \
--method retrain \
--keep_all_ckpts \
--mixed_precision fp16 \
--use_8bit_optimizer \
--gradient_accumulation_steps 1 \
--precompute_stage reuse
```

## Generate samples to see if the training was successful

1. Generate samples

```bash
pane_index=$(tmux display-message -p "#{pane_index}" | sed 's/%//');
CUDA_VISIBLE_DEVICES=$((pane_index+4)) \
python unconditional_generation/generate_samples.py \
--dataset celeba \
--n_samples 3200 \
--method retrain \
--num_inference_steps 100 \
--trained_steps 20000 \
--batch_size 16 \
--use_ema \
--seed $pane_index
```

1. Evaluate the generates images qualitatively. Or you can also calcualte the FID score by running

```bash
CUDA_VISIBLE_DEVICES=2 \
python -m pytorch_fid \
/projects/leelab3/chanwkim/data_attribution/datasets/celeba_hq_256_50_resized \
/projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/retrain/20000/ema_generated_samples/full
```

90k -> 21
70k -> 17
50k -> 13
20k -> 13
10k -> 20

## Retraining

1. Generate the retraining command file by running

```bash
python unconditional_generation/celeba_experiments/setup_train_commands.py \
--dataset celeba \
--method="retrain" \
--removal_dist="shapley" \
--num_removal_subsets=300 \
--num_subsets_per_job=1
```

Make sure to run this from the repo directory (i.e., `data_attribution/`). After running the above, the file `unconditional_generation/celeba_experiments/train.job` should be updated. A new file `unconditional_generation/celeba_experiments/commands/train/celeba/retrain/shapley/command.txt` should be created. This contains commands to run from the command line.

2. (Optional) Submit the SLURM job for running the command.
With `unconditional_generation/celeba_experiments/train.job` updated, run

```bash
cd unconditional_generation/celeba_experiments
sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

```bash
accelerate launch --config_file deepspeed_config_dp.yaml --main_process_port 29501 --gpu_ids 4,5 \
unconditional_generation/main.py \
--dataset celeba \
--method retrain \
--removal_dist shapley \
--removal_seed 0 \
--keep_all_ckpts \
--mixed_precision fp16 \
--use_8bit_optimizer \
--gradient_accumulation_steps 1 \
--precompute_stage reuse
```

## Calculate global model behavior (diversity score) for the retraining models

1. Generate celebrity images for evaluation

```bash
CUDA_VISIBLE_DEVICES=2 \
python tools/generate_celeba_images.py
# ll /projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/generated_samples
# 17*3*2*40=4080
```

``` bash
python unconditional_generation/calculate_diversity_score.py \
--celeba_images_dir /projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/generated_samples \
--generated_images_dir /projects/leelab3/chanwkim/data_attribution/datasets/celeba_hq_256_50_resized \
--num_cluster 20
# entropy: 2.327470665705897
```

``` bash
python unconditional_generation/calculate_diversity_score.py \
--celeba_images_dir /projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/generated_samples \
--generated_images_dir /projects/leelab3/chanwkim/data_attribution/datasets/celeba_hq_256 \
--num_cluster 20
# entropy: 3.165551995844454
```

``` bash
python unconditional_generation/calculate_diversity_score.py \
--celeba_images_dir /projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/generated_samples \
--generated_images_dir /projects/leelab3/chanwkim/data_attribution/datasets/celeba_hq_256_50_resized_temp \
--num_cluster 20
# entropy: 2.15769499526684
```

2. Calculate the diversity score for the retraining models
```bash
python unconditional_generation/calculate_global_scores_diversity.py \
--dataset celeba \
--removal_dist shapley \
--removal_seed 0 \
--trained_steps 20001 \
--use_ema \
--method retrain \
--num_inference_steps 100 \
--exp_name diversity_retraining \
--seed 42 \
--db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_measurements.jsonl \
# --precompute_stage reuse \
--n_samples 10


```

1. Generate the retraining command file by running

```bash
python unconditional_generation/celeba_experiments/setup_generate_commands.py \
--dataset celeba \
--method retrain \
--removal_dist shapley \
--seed_range=300 \
--num_images=1000 \
--model_per_job 1
```

2. (Optional) Submit the SLURM job for running the command.
With `unconditional_generation/celeba_experiments/generate.job` updated, run

```bash
cd unconditional_generation/celeba_experiments
sbatch -p gpu-rtx6k --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G generate.job
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G generate.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

## Prune and retrain

```bash
accelerate launch --gpu_ids 1 \
unconditional_generation/prune.py \
--load /projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/retrain/models/full/ \
--dataset celeba \
--pruning_ratio 0.1 \
--pruner magnitude \
--thr 0.05 \
--mixed_precision fp16
```

```bash
accelerate launch --config_file unconditional_generation/celeba_experiments/deepspeed_config_dp.yaml --gpu_ids 5,6 \
unconditional_generation/main.py \
--dataset celeba \
--method prune_fine_tune \
--pruning_ratio 0.3 \
--keep_all_ckpts \
--mixed_precision fp16 \
--use_8bit_optimizer \
--gradient_accumulation_steps 1 \
--precompute_stage reuse
```