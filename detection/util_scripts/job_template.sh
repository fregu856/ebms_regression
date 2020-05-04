#!/bin/bash
uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $CUDA_VISIBLE_DEVICES"
cd ..
source /home/damartin/anaconda3/etc/profile.d/conda.sh
conda activate pylearn041
export LD_LIBRARY_PATH="/home/damartin/scratch/software/cuda10/lib64:/home/damartin/scratch/libs/libjpeg-turbo/build/install/lib:$LD_LIBRARY_PATH"
export PATH="/home/damartin/scratch/software/cuda10/bin:$PATH"
taskset -c $(util_scripts/gpu2cpu_affinity.py $CUDA_VISIBLE_DEVICES) python -u run_training.py $1 $2
