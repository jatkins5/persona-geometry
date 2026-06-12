#!/usr/bin/env bash
#SBATCH --job-name=exp2-manifold
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=experiments/results/exp2_manifold/slurm-%j.out
#SBATCH --error=experiments/results/exp2_manifold/slurm-%j.out
#
# Experiment 2 (reachable manifold / boundary map).
#   # calibrate the steering magnitude first (quick), then read the coherent range:
#   sbatch --time=00:30:00 experiments/run_manifold.sbatch --model llama3.1-8b --mode calibrate
#   # then the boundary map with the chosen magnitudes:
#   sbatch experiments/run_manifold.sbatch --model llama3.1-8b --mode boundary --mags 20,40,80,120,160
set -euo pipefail

PY_MODULE="python/3.11.11-5e66"
ENV_DIR="$HOME/envs/persona-geometry"
REPO_DIR="$SLURM_SUBMIT_DIR"

module load "$PY_MODULE"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export HF_HOME="/oscar/scratch/$USER/hf"
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
mkdir -p experiments/results/exp2_manifold

pip install -q matplotlib wordfreq

echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ">> running exp2_manifold.py $*"

export PYTHONPATH="$REPO_DIR/experiments:${PYTHONPATH:-}"
python experiments/exp2_manifold.py "$@"

echo ">> Done."
