
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
end=899
output_file="unconditional_generation/celeba_experiments/slurm/retrain_shapley.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/retrain/models/shapley/shapley_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist shapley --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file
  fi
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

### Datamodel (0.75)
```bash
start=0
end=99
output_file="unconditional_generation/celeba_experiments/slurm/retrain_datamodel_0_75.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/retrain/models/datamodel/datamodel_alpha=0.75_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/retrain/models/datamodel/datamodel_alpha=0.75_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse  --opt_seed 43 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/retrain/models/datamodel/datamodel_alpha=0.75_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse  --opt_seed 44 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/" >> $output_file
  fi
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G retrain_datamodel_0_75.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

### Datamodel (0.5)
```bash
start=0
end=99
output_file="unconditional_generation/celeba_experiments/slurm/retrain_datamodel_0_5.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/retrain/models/datamodel/datamodel_alpha=0.5_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.5 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file  
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/retrain/models/datamodel/datamodel_alpha=0.5_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.5 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse  --opt_seed 43 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/retrain/models/datamodel/datamodel_alpha=0.5_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.5 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse  --opt_seed 44 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/" >> $output_file
  fi
done
```

```bash 
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G retrain_datamodel_0_5.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

### Datamodel (0.25)
```bash
start=0
end=99
output_file="unconditional_generation/celeba_experiments/slurm/retrain_datamodel_0_25.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/retrain/models/datamodel/datamodel_alpha=0.25_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.25 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file  
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/retrain/models/datamodel/datamodel_alpha=0.25_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.25 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse  --opt_seed 43 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/retrain/models/datamodel/datamodel_alpha=0.25_seed=$seed/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --removal_dist datamodel --datamodel_alpha 0.25 --removal_seed $seed --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse  --opt_seed 44 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/" >> $output_file
  fi
done
```

```bash 
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G retrain_datamodel_0_25.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```







### Leave one out
```bash
output_file="unconditional_generation/celeba_experiments/slurm/retrain_leave_one_out.txt"
> $output_file
for excluded_class in 17 63 65 95 104 122 162 177 188 204 206 228 321 329 330 335 342 368 422 423 449 451 452 509 521 573 593 598 603 615 632 636 651 805 812 858 870 982 1047 1293 1429 1538 1581 1775 1837 1838 2020 2120 2214 2632
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/retrain/models/excluded_$excluded_class/ckpt_steps_00020001.pt"
  if [ ! -f "$file_path" ]; then
    echo "Missing $excluded_class"
    echo "accelerate launch --gpu_ids 0 --config_file unconditional_generation/celeba_experiments/deepspeed_config_single.yaml unconditional_generation/main.py --dataset celeba --method retrain --excluded_class $excluded_class --mixed_precision fp16 --use_8bit_optimizer --gradient_accumulation_steps 1 --precompute_stage reuse" >> $output_file
  fi
done
```





```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G retrain_leave_one_out.job
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

retraining shapley
```bash
start=0
end=599
output_file="unconditional_generation/celeba_experiments/slurm/diversity_retrain_shapley.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_measure/celeba_retrain_models_shapley_shapley_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist shapley --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_retrain_shapley --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_retrain_shapley.jsonl --n_samples 1000" >> $output_file
  fi
done
```


```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G diversity_retrain_shapley.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

retraining datamodel (alpha=0.75)
```bash
start=0
end=99
output_file="unconditional_generation/celeba_experiments/slurm/diversity_datamodel_0_75.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_datamodel_0_75/celeba_retrain_models_datamodel_datamodel_alpha=0.75_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_75 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_datamodel_0_75.jsonl --n_samples 1000" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/diversity_datamodel_0_75/celeba_retrain_models_datamodel_datamodel_alpha=0.75_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_75 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/diversity_datamodel_0_75.jsonl --n_samples 1000 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/diversity_datamodel_0_75/celeba_retrain_models_datamodel_datamodel_alpha=0.75_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.75 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_75 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/diversity_datamodel_0_75.jsonl --n_samples 1000 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/" >> $output_file
  fi
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G diversity_datamodel_0_75.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```

retraining datamodel (alpha=0.5)

```bash
start=0
end=99
output_file="unconditional_generation/celeba_experiments/slurm/diversity_datamodel_0_5.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_datamodel_0_5/celeba_retrain_models_datamodel_datamodel_alpha=0.5_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.5 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_5 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_datamodel_0_5.jsonl --n_samples 1000" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/diversity_datamodel_0_5/celeba_retrain_models_datamodel_datamodel_alpha=0.5_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.5 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_5 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/diversity_datamodel_0_5.jsonl --n_samples 1000 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/diversity_datamodel_0_5/celeba_retrain_models_datamodel_datamodel_alpha=0.5_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.5 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_5 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/diversity_datamodel_0_5.jsonl --n_samples 1000 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/" >> $output_file
  fi
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G diversity_datamodel_0_5.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```


retraining datamodel (alpha=0.25)

```bash
start=0
end=99
output_file="unconditional_generation/celeba_experiments/slurm/diversity_datamodel_0_25.txt"
> $output_file
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_datamodel_0_25/celeba_retrain_models_datamodel_datamodel_alpha=0.25_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.25 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_25 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_datamodel_0_25.jsonl --n_samples 1000" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/diversity_datamodel_0_25/celeba_retrain_models_datamodel_datamodel_alpha=0.25_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.25 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_25 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/celeba/diversity_datamodel_0_25.jsonl --n_samples 1000 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed43/" >> $output_file
  fi
done
for (( seed=$start; seed<=$end; seed++ ))
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/diversity_datamodel_0_25/celeba_retrain_models_datamodel_datamodel_alpha=0.25_seed=${seed}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $seed"  
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --removal_dist datamodel --datamodel_alpha 0.25 --removal_seed $seed --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_datamodel_0_25 --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/celeba/diversity_datamodel_0_25.jsonl --n_samples 1000 --outdir /gscratch/scrubbed/chanwkim/diffusion-attr_seed44/" >> $output_file
  fi
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G diversity_datamodel_0_25.job
# sbatch -p gpu-rtx6k --account=aims --gres=gpu:rtx6k:1 --cpus-per-task=4 --mem=16G train.job
# sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G train_copy.job
```



Leave one out

```bash
output_file="unconditional_generation/celeba_experiments/slurm/diversity_leave_one_out.txt"
> $output_file
for excluded_class in 17 63 65 95 104 122 162 177 188 204 206 228 321 329 330 335 342 368 422 423 449 451 452 509 521 573 593 598 603 615 632 636 651 805 812 858 870 982 1047 1293 1429 1538 1581 1775 1837 1838 2020 2120 2214 2632
do
  file_path="/gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_leave_one_out/celeba_retrain_models_excluded_${excluded_class}_sample.jpg"
  if [ ! -f "$file_path" ]; then
    echo "Missing $excluded_class"
    echo "python unconditional_generation/calculate_global_scores_diversity.py --dataset celeba --excluded_class $excluded_class --trained_steps 20001 --use_ema --method retrain --num_inference_steps 100 --exp_name diversity_leave_one_out --seed 42 --db /gscratch/scrubbed/chanwkim/diffusion-attr/celeba/diversity_leave_one_out.jsonl --n_samples 1000" >> $output_file
  fi
done
```

```bash
cd unconditional_generation/celeba_experiments/slurm
sbatch -p ckpt --account=aims --gpus=1 --constraint="a40|rtx6k|a100" --cpus-per-task=4 --mem=16G diversity_leave_one_out.job
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

calculate 3 seed x 100 datamodel
calculate diveristy score.
prioritze alpha of 0.5


ll /gscratch/scrubbed/chanwkim/diffusion-attr/logs/diversity/diversity_measure -tr

cat /gscratch/scrubbed/chanwkim/diffusion-attr/logs/diversity/diversity_measure/run-18089346-144.out