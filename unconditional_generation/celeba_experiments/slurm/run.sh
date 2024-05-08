#!/bin/bash

# Make sure we're in a container before we continue.
if [[ -z $APPTAINER_NAME ]]; then
    echo "Error: not in an Apptainer container"
    exit 1
fi

# Now that we're sure we're in a container,
# let's source our ~/.bashrc for 'conda', then activate our environment.
. ~/.bashrc
conda activate data_attribution
export HF_DATASETS_CACHE="/gscratch/scrubbed/chanwkim/.cache/huggingface_cache"
cd /gscratch/aims/chanwkim/data_attribution

nvidia-smi

export PYTHONPATH="$PYTHONPATH:$PWD"

$*
