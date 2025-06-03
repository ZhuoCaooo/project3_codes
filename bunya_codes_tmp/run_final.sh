#!/bin/bash
#SBATCH --job-name=lc_4epochs
#SBATCH --partition=gpu_cuda
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --mem=24GB
#SBATCH --account=a_z_zheng
#SBATCH --output=4epochs_%j.out  # ← CHANGED: final_%j.out → 4epochs_%j.out
#SBATCH --error=4epochs_%j.err   # ← CHANGED: final_%j.err → 4epochs_%j.err

module load python/3.11.3-gcccore-12.3.0
export PYTHONPATH=/scratch/user/uqzcao2/python_packages:$PYTHONPATH
export HF_HOME=/scratch/user/uqzcao2/huggingface_cache

cd /scratch/user/uqzcao2/lane_change_pal/lc_llm
python train_final.py