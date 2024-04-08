#!/bin/bash

# Make sure we're in a container before we continue.
if [[ -z $APPTAINER_NAME ]]; then
    echo "Error: not in an Apptainer container"
    exit 1
fi

# Now that we're sure we're in a container,
# let's source our ~/.bashrc for 'conda', then activate our environment.
. ~/.bashrc
conda activate diffusion-attr

cd /gscratch/aims/clin25/data_attribution
export PYTHONPATH="$PYTHONPATH:$PWD"

eval "$*"
