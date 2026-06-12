#!/usr/bin/env bash
#SBATCH --job-name=extract-vec
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=experiments/results/extract/slurm-%j.out
#SBATCH --error=experiments/results/extract/slurm-%j.out
#
# Re-extract persona vectors for a new model (heavy: ~n_traits*pool_n_q generations).
# Needs a gated-model HF token already stored (HF_HOME/token), e.g. for Llama.
#   sbatch experiments/run_extract_vectors.sbatch --model llama3.1-8b
set -euo pipefail

PY_MODULE="python/3.11.11-5e66"
ENV_DIR="$HOME/envs/persona-geometry"
REPO_DIR="$SLURM_SUBMIT_DIR"

module load "$PY_MODULE"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export HF_HOME="/oscar/scratch/$USER/hf"   # weights + gated-model token live here
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
mkdir -p experiments/results/extract

echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ">> running extract_persona_vectors.py $*"

export PYTHONPATH="$REPO_DIR/experiments:${PYTHONPATH:-}"
python experiments/extract_persona_vectors.py "$@"

echo ">> Done."
