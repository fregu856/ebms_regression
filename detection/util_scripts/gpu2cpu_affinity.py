#!/usr/bin/env python
import os
import sys
import subprocess
import numpy as np
from subprocess import Popen, PIPE
import re

ncpus = subprocess.check_output("cat /proc/cpuinfo | grep processor | wc -l", shell=True, universal_newlines=True)
ncpus = int(ncpus)

ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')

out = subprocess.check_output(["nvidia-smi", "topo", "-m"], universal_newlines=True)
out = ansi_escape.sub('', out)

affinity_column_id = 0
gpu2cpu = dict()
max_gpu_id = -1

for line in out.splitlines():
    tokens = line.strip().split('\t')
    if len(tokens) == 0:
        continue
    if affinity_column_id == 0:
        for i,t in enumerate(tokens):
            if t.strip() == 'CPU Affinity':
                affinity_column_id = i+1
        if affinity_column_id == 0:
            raise Exception("Affinity column not found")
        continue
    if tokens[0].startswith("GPU"):
        gpuid = int(tokens[0][3:])
        if gpuid > max_gpu_id:
            max_gpu_id = gpuid
        gpu2cpu[gpuid] = tokens[affinity_column_id]
ngpus = max_gpu_id + 1

aff_mat = np.zeros((ngpus, ncpus), dtype=np.float)

for gpuid,affstr in gpu2cpu.items():
    tokens = affstr.strip().split('-')
    if ',' in tokens[0]:
        tokens[0] = tokens[0].split(',')[0]
    if ',' in tokens[1]:
        tokens[1] = tokens[1].split(',')[-1]
    cpulow, cpuhigh = int(tokens[0]), int(tokens[1])
    for i in range(cpulow, cpuhigh+1):
        aff_mat[gpuid,i] = 1.0

if len(sys.argv) == 1:
    print(aff_mat)
    exit(0)

arg = sys.argv[1]
if arg == 'ENV':
    arg = os.environ['CUDA_VISIBLE_DEVICES']

arg_tokens = arg.split(',')
arg_tokens = [int(a) for a in arg_tokens]

best_cpus = np.sum(aff_mat[arg_tokens,:], axis=0)

quota_cpus = np.nonzero(best_cpus > 0)[0]
quota_cpus = [str(a) for a in quota_cpus]

out = ",".join(quota_cpus)
print(out)
