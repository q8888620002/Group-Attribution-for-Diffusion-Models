#!/bin/bash

# Make Init wandb api
export WANDB_API_KEY="a010d8a84d6d1f4afed42df8d3e37058369030c4"

# let's source our ~/.bashrc for 'conda', then activate our environment.
. ~/.bashrc
conda activate data_att

cd /mmfs1/home/mingyulu/data_attribution
python main_new.py $*