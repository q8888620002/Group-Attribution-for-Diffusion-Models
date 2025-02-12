# Group Attribution via Sparsified Fine-tuning

This README provides instructions for training **unconditional** diffusion models, including methods such as **retraining (exact unlearning)** and **sparsified unlearning** and computing group attribution score. Please follow these steps to reproduce our results:

1. Train the diffusion model on the original dataset.
2. Apply pruning and fine-tune it on the original dataset.
3. Load the pruned model and fine-tune (sFT) it on the **removal** or **remaining** distribution.
4. Compute corresponding model behavior for both unlearned and retrained models.
5. For validation, train models using the datamodel distribution and compute their behavior.
6. Measure the performance of unlearning using linear datamodel score (LDS).


## 1. Training a Diffusion Model from Scratch with a removal distribution
To train a diffusion model, execute the following command:

```bash
python main.py

--dataset [cifar, celeba ] \

## Learning method

--method [retrain] \

## Removal distribution args

--removal_dist [datashapley/datamodel/uniform/None] \
--datamodel_alpha [0.5] \
--removal_seed [0] \

## Model sparsification args. This is needed when training a pruned model

--pruning_ratio [0.3]\
--pruner [magnitude]\
--thr [0.05]\

## Training args

--keep_all_ckpts \

## Accelerator args

--mixed_precision [no, bf16, fp16] \
--gradient_accumulation_steps [1] \
```

### Efficient Training for CelebA-HQ (256x 256)
To reduce GPU memory usage and facilitate training with CelebA-HQ dataset (e.g., on RTX 2080ti),

1. Create and set up the configuration file `deepspeed_config_dp.yaml` for [Accelerator](https://huggingface.co/docs/accelerate/en/package_reference/accelerator)
2. Run the following command to precompute the VQVAE latent embeddings, train with [8bit Adam](https://github.com/TimDettmers/bitsandbytes) and [data parallelism](https://huggingface.co/docs/accelerate/v0.27.2/en/usage_guides/deepspeed#deepspeed-config-file).

```bash
accelerate launch --config_file deepspeed_config_dp.yaml \
main.py --dataset celeba \
--method retrain \
--mixed_precision fp16 \
--use_8bit_optimizer \
--precompute_stage save
```
3. To train with precomputed latent embeddings, change `--precompute_stage save` to `reuse` in the command.

### Compute global model behavior
To compute global model behavior for a retrained model, execute the following command:
```bash

python unlearn.py

--load [path_to_full_model]
--dataset [cifar, celeba ] \
--method [retrain] \
--db [path_saved_results]
--exp_name [exp_name]

## Removal distribution args

--removal_dist [datashapley/datamodel/uniform/None] \
--datamodel_alpha [0.5] \
--removal_seed [0] \

## generation params
--n_local_samples 100
--n_noises 50
--use_ema 
--model_behavior global

```

### Prune a full model
To prune a trained diffusion model, execute the following command:
```bash
python prune.py --load model_path --pruning_ratio [0.3] --pruner [magnitude] --thr [0.05]
```
*Note that the pruned model needs to be fine-tuned (retrained) after pruning to achieve comparable performance to the original full model.

## 2. Unlearning a Removal Distribution via Gradient Descent for Full or Fine-Tuned Pruned Models
To unlearn a full model and compute its model behavior, execute the following command:

```bash
python unlearn.py

--load [full_model_path] \
--db [path_to_saved_results] \
--dataset [cifar, celeba ] \

## Unlearning methods

--method [gd] \

## Unlearning params

--ga_ratio [1.0]
--gd_steps [2000]

## Removal distribution args

--removal_dist [datashapley/datamodel/uniform/None] \
--datamodel_alpha [0.5] \
--removal_seed [0] \

## Params specification for loading the full or fine-tuned pruned model for unlearning.

--pruning_ratio [0.3]\
--pruner [magnitude]\
--thr [0.05]\

## Training args

--keep_all_ckpts \

## Accelerator args

--mixed_precision [no, bf16, fp16] \
--gradient_accumulation_steps [1] \

## model behavior

--model_behavior [global]
```

## 3. Linear datamodel score (LDS) computation

*Note: The testing set consists of the model behaviors from retrained models using the datamodel distribution, following **Step 1**.

To compute linear datamodel score (LDS) for a given unlearning method and a attribution kernel, e.g. Shapley, execute the following command

```bash
python lds.py \
--dataset [cifar/celeba] \
--removal_dist [shapley/datamodel] \
--model_behavior_key [is/diversity_score/aesthetic_score] \

## Training Set Parameters
--train_db [path_to_retrain/unlearn_model_behavior] \
--train_exp_name [train_exp_name] \
--method [retrain/ga/lora/iu] \
--full_db [path_to_full_model_behavior] \
--null_db [path_to_null_model_behavior] \
--max_train_size [100] \

## Validation Set Parameters
--test_db [path_to_datamodel_behavior] \
--test_exp_name [test_exp_name] \
--num_test_subset [100] \

--by_class
```
