#!/bin/bash

#SBATCH --job-name=compression    # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=2        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:a100_1g.10gb:1
#SBATCH --time=120:00:00
#SBATCH --time=48:05:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH -o slurm_output/slurm_output-%j

function fail {
    echo "FAIL: $@" >&2
    exit 1  # signal failure
}

source ~/miniconda3/etc/profile.d/conda.sh 

which python3 || fail "conda load fail"
conda init bash 
conda activate tmp || fail "conda load fail"
jupyter-notebook --ip 0.0.0.0 --port 9010 --NotebookApp.token=neel --no-browser
