#!/bin/bash

# Make Init wandb api
export WANDB_API_KEY="YOUR WANDB API KEY"

# let's source our ~/.bashrc for 'conda', then activate our environment.
. ~/.bashrc
conda activate data_att

cd /mmfs1/home/mingyulu/data_attribution
python main_new.py $*