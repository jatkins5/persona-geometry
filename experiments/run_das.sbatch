#!/usr/bin/env bash
#SBATCH --job-name=exp3-das
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/results/exp3_das/slurm-%j.out
#SBATCH --error=experiments/results/exp3_das/slurm-%j.out
#
# Experiment 3 (DAS persona causal interface). Trains a k-dim interchange subspace by gradient
# descent (model frozen) -- judge-free, teacher-forced CE.
#   # smoke (quick mechanics + site check):
#   sbatch --time=00:40:00 experiments/run_das.sbatch --model llama3.1-8b \
#       --n-personas 6 --k-list 1,4 --seeds 0 --steps 40 --big-n 8 --big-steps 60 \
#       --out-dir experiments/results/exp3_das_smoke
#   # full run:
#   sbatch experiments/run_das.sbatch --model llama3.1-8b
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
mkdir -p experiments/results/exp3_das

pip install -q matplotlib

echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ">> running exp3_das.py $*"

export PYTHONPATH="$REPO_DIR/experiments:${PYTHONPATH:-}"
python experiments/exp3_das.py "$@"

echo ">> Done."
