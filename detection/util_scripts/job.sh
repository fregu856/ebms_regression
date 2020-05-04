#!/bin/bash
uname -n; echo "Job ID: $JOB_ID"; echo "GPU: $CUDA_VISIBLE_DEVICES"
cd ../tools
source /home/bhatg/scratch/software/anaconda3/etc/profile.d/conda.sh
conda activate maskrcnn_benchmark
export LD_LIBRARY_PATH="/home/bhatg/scratch/software/cuda10/lib64:/home/damartin/scratch/libs/libjpeg-turbo/build/install/lib:$LD_LIBRARY_PATH"
export PATH="/home/bhatg/scratch/software/cuda10/bin:$PATH"
taskset -c $(../util_scripts/gpu2cpu_affinity.py $CUDA_VISIBLE_DEVICES) python -u train_net.py  --config-file "$1"
