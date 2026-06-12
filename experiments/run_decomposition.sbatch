#!/usr/bin/env bash
#SBATCH --job-name=exp1-decomp
#SBATCH --partition=gpu          # short job; switch to gpu-debug if the queue is busy
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=experiments/results/exp1_decomposition/slurm-%j.out
#SBATCH --error=experiments/results/exp1_decomposition/slurm-%j.out
#
# JOB 1 (quick): scaled word-vs-persona decomposition + the frequency analysis.
# Forward passes only (no generation, no judge) -- runs in minutes once the model loads.
#   sbatch experiments/run_decomposition.sbatch              # full 220-trait pool
#   sbatch experiments/run_decomposition.sbatch --n-traits 64
set -euo pipefail

PY_MODULE="python/3.11.11-5e66"
ENV_DIR="$HOME/envs/persona-geometry"
REPO_DIR="$SLURM_SUBMIT_DIR"

module load "$PY_MODULE"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export HF_HOME="/oscar/scratch/$USER/hf"   # weights cached on scratch, off home quota
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
mkdir -p experiments/results/exp1_decomposition

# Extra deps for this analysis (compute node, not the login node). No-op if already present.
pip install -q wordfreq matplotlib

echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ">> running exp1_decomposition.py $*"

# Keep CWD at the repo root so any relative --out-dir resolves intuitively; put the
# experiments dir on PYTHONPATH so `import persona_lib` / `import prompts` still work.
export PYTHONPATH="$REPO_DIR/experiments:${PYTHONPATH:-}"
python experiments/exp1_decomposition.py "$@"

echo ">> Done."
