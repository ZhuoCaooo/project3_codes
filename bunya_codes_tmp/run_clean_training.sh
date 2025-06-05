#!/bin/bash --login
#SBATCH --job-name=phi2_clean_training
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --qos=gpu
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:h100:1
#SBATCH --account=a_z_zheng
#SBATCH -o clean_train-%j.output
#SBATCH -e clean_train-%j.error

# Load required modules
module load python/3.11.3-gcccore-12.3.0

# Set environment variables
export PYTHONPATH=/scratch/user/uqzcao2/python_packages:$PYTHONPATH
export TMPDIR=/scratch/user/uqzcao2/tmp
export HF_HOME=/scratch/user/uqzcao2/huggingface_cache

# Change to working directory
cd /scratch/user/uqzcao2/lane_change_pal/lc_llm

# Print job info
echo "Clean training job started at: $(date)"
echo "Running on node: $(hostname)"
echo "GPU info:"
nvidia-smi

# Run the clean training script
python clean_train_phi2.py

echo "Training finished at: $(date)"
