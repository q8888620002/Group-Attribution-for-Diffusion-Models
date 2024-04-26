
# Command line and codes for running celeba experiment

## Installation

```bash
export PYTHONPATH="$PYTHONPATH:$PWD"
```

```bash
conda create -n data_attribution python=3.11.5
conda activate data_attribution
conda install pytorch==2.2.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
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
accelerate launch --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml --gpu_ids 0 \
unconditional_generation/main.py \
--dataset celeba \
--method retrain \
--keep_all_ckpts \
--mixed_precision fp16 \
--use_8bit_optimizer \
--gradient_accumulation_steps 1 \
--precompute_stage reuse \
--save_null_model
``` 
or if you want to use multiple GPUs

```bash
accelerate launch --config_file unconditional_generation/celeba_experiments/deepspeed_config_dp.yaml --gpu_ids 0,1 \
unconditional_generation/main.py \
--dataset celeba \
--method retrain \
--keep_all_ckpts \
--mixed_precision fp16 \
--use_8bit_optimizer \
--gradient_accumulation_steps 1 \
--precompute_stage reuse \
--save_null_model
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

2. Evaluate the generates images qualitatively. Or you can also calcualte the FID score by running

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

1. Generate the retraining command file by running the following script. Make sure to run this from the repo directory (i.e., `data_attribution/`).
`unconditional_generation/celeba_experiments/slurm/retrain_shapley.job` will be updated.

### Shapley

```bash
start=0
end=299
output_file="unconditional_generation/celeba_experiments/slurm/retrain_shapley.txt"
for (( seed=$start; seed<=$end; seed++ ))
do
  echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist shapley --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file
done
```

2. Submit the SLURM job for running the command.
With `unconditional_generation/celeba_experiments/slurm/retrain_shapley.job` updated, run

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G retrain_shapley.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

### DataShapley
```bash
start=0
end=299
output_file="unconditional_generation/celeba_experiments/slurm/retrain_datamodel_0_75.txt"
for (( seed=$start; seed<=$end; seed++ ))
do
  echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G retrain_datamodel_0_75.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

## Calculate global model behavior (diversity score) for the retraining models

1. Generate celebrity images for evaluation

```bash
CUDA_VISIBLE_DEVICES=2 \
python tools/generate_celeba_images.py
# ll /projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/generated_samples
# 17*3*2*40=4080
```

```bash
/projects/leelab3/chanwkim/data_attribution/diffusion-attr/celeba/
tar -zcvf generated_samples.tar.gz generated_samples
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

2. Calculate diversity score by running


null model
```bash
python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --trained_steps 0 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_measure_null --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_measure_null.jsonl --n_samples 1000
```

full model
```bash
python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_measure_full --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_measure_full.jsonl --n_samples 1000
```

Removal Shapley
```bash
start=0
end=299
output_file="unconditional_generation/celeba_experiments/slurm/diversity_retrain_shapley.txt"
for (( seed=$start; seed<=$end; seed++ ))
do
  echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist shapley --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_measure --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_measure.jsonl --n_samples 1000" >> $output_file
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G diversity_retrain_shapley.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

# Sync the data to the server

```bash
# -z for compression
# --delete to remove files that are not in the source
# --include, --exclude to include/exclude files
# --backup and --backup_dir to keep backup of the files

# dry run
rsync -Pavn /gscratch/scrubbed/chanwkim/diffusion-attr/ chanwkim@bicycle.cs.washington.edu:/projects/leelab3/chanwkim/data_attribution/diffusion-attr

# actual run
rsync -Pav /gscratch/scrubbed/chanwkim/diffusion-attr/ chanwkim@bicycle.cs.washington.edu:/projects/leelab3/chanwkim/data_attribution/diffusion-attr
```

