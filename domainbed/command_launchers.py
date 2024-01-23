# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

"""
A command launcher launches a list of commands on a cluster; implement your own
launcher to add support for your cluster. We've provided an example launcher
which runs all commands serially on the local machine.
"""

import subprocess
import time
import torch
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed

SLURM = 'slurm'
SLURM_LONG = 'slurm_long'

def create_temp_bash_script(command, cloud=SLURM):
    if cloud == SLURM:
        content = """#!/bin/bash
#SBATCH --job-name=dbsweep
#SBATCH --time=3:00:00
#SBATCH --partition=medium
#SBATCH --output=slurm_sweep/sweep_%j.txt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8"""
    elif cloud == SLURM_LONG:
        content = """#!/bin/bash
#SBATCH --job-name=dbsweep
#SBATCH --time=12:00:00
#SBATCH --partition=long
#SBATCH --output=slurm_sweep/sweep_%j.txt
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8"""
    else:
        raise Exception("Unsupported Cloud Name")

    content += f"\n\n{command}"
    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as temp_script:
        temp_script.write(content)
    return temp_script.name


def run_sbatch(temp_script_name):
    cmd = ["sbatch", "--wait", temp_script_name]
    result = subprocess.run(cmd, shell=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode == 0:
        print(f"Job completed: {result.stdout.strip()}")
    else:
        print(f"Error executing job: {result.stderr.strip()}; {result.stdout.strip()}")

    return result.returncode, temp_script_name

def slurm_launcher(commands):
    num_jobs = 16 # This is the number of gpus you want to use
    temp_script_names = [create_temp_bash_script(i, SLURM) for i in commands]

    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_sbatch, script_name) for script_name in temp_script_names]

        for future in as_completed(futures):
            returncode, script_name = future.result()
            if returncode == 0:
                os.remove(script_name)

def slurm_long_launcher(commands):
    num_jobs = 16 # This is the number of gpus you want to use
    temp_script_names = [create_temp_bash_script(i, SLURM_LONG) for i in commands]

    with ThreadPoolExecutor(max_workers=num_jobs) as executor:
        futures = [executor.submit(run_sbatch, script_name) for script_name in temp_script_names]

        for future in as_completed(futures):
            returncode, script_name = future.result()
            if returncode == 0:
                os.remove(script_name)

def local_launcher(commands):
    """Launch commands serially on the local machine."""
    for cmd in commands:
        subprocess.call(cmd, shell=True)

def dummy_launcher(commands):
    """
    Doesn't run anything; instead, prints each command.
    Useful for testing.
    """
    for cmd in commands:
        print(f'Dummy launcher: {cmd}')

def multi_gpu_launcher(commands):
    """
    Launch commands on the local machine, using all GPUs in parallel.
    """
    print('WARNING: using experimental multi_gpu_launcher.')

    try:
        # Get list of GPUs from env, split by ',' and remove empty string ''
        # To handle the case when there is one extra comma: `CUDA_VISIBLE_DEVICES=0,1,2,3, python3 ...`
        available_gpus = [x for x in os.environ['CUDA_VISIBLE_DEVICES'].split(',') if x != '']
    except Exception:
        # If the env variable is not set, we use all GPUs
        available_gpus = [str(x) for x in range(torch.cuda.device_count())]
    n_gpus = len(available_gpus)
    procs_by_gpu = [None]*n_gpus


    while len(commands) > 0:
        for idx, gpu_idx in enumerate(available_gpus):
            proc = procs_by_gpu[idx]
            if (proc is None) or (proc.poll() is not None):
                # Nothing is running on this GPU; launch a command.
                cmd = commands.pop(0)
                new_proc = subprocess.Popen(
                    f'CUDA_VISIBLE_DEVICES={gpu_idx} {cmd}', shell=True)
                procs_by_gpu[idx] = new_proc
                break
        time.sleep(1)

    # Wait for the last few tasks to finish before returning
    for p in procs_by_gpu:
        if p is not None:
            p.wait()

REGISTRY = {
    'local': local_launcher,
    'dummy': dummy_launcher,
    'multi_gpu': multi_gpu_launcher,
    'slurm': slurm_launcher,
    'slurm_long': slurm_long_launcher,
}

try:
    from domainbed import facebook
    facebook.register_command_launchers(REGISTRY)
except ImportError:
    pass
