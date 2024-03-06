# Experiments for text-to-image diffusion models

## Stable Diffusion fine-tuned on ArtBench Post-Impressionism

### Data
1. Download the ArtBench dataset into `src.constants.DATASET_DIR`. The version with train-test split and 256x256 resolution is used (https://artbench.eecs.berkeley.edu/files/artbench-10-imagefolder-split.tar). More details about the ArtBench are here (https://github.com/liaopeiyuan/artbench).

2. Create the meta data files by running
```
python text_to_image/artbench/create_metadata.py \
    --parent_dir PATH_TO_YOUR_ARTBENCH_DIRECTORY \
    --split train
```

### LoRA Fine-tuning Stable Diffusion
1. Generate the command file by running
```
python text_to_image/experiments/setup_train_commands.py \
    --dataset="artbench_post_impressionism" \
    --seed=42 \
    --method="retrain"
```
Make sure to run this from the repo directory (i.e., `data_attribution/`). After running the above, the file `text_to_image/experiments/train.job` should be updated. A new file `text_to_image/experiments/commands/artbench_post_impressionism/retrain/full/command.txt` should be created. This contains the command to run from the command line.

2. (Optional) Submit the SLURM job for running the command.
With `text_to_image/experiments/train.job` updated, run
```
cd text_to_image/experiments
sbatch train.job
```

3. TODO: Include command to run to get FID.

### Shapley attribution for artists
1. Generate the retraining command file by running
```
python text_to_image/experiments/setup_train_commands.py \
    --dataset="artbench_post_impressionism" \
    --seed=42 \
    --method="retrain" \
    --removal_dist="shapley" \
    --num_removal_subsets=500 \
    --num_subsets_per_job=1 \
    --removal_unit="artist"
```
Make sure to run this from the repo directory (i.e., `data_attribution/`). After running the above, the file `text_to_image/experiments/train.job` should be updated. A new file `text_to_image/experiments/commands/artbench_post_impressionism/retrain/artist_shapley/command.txt` should be created. This contains commands to run from the command line.

2. (Optional) Submit the SLURM job for running the command.
With `text_to_image/experiments/train.job` updated, run
```
cd text_to_image/experiments
sbatch train.job
```

3. TODO: Compute the model behaviors for the retrained models.
