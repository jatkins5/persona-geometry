#!/usr/bin/env bash
#SBATCH --job-name=exp1-contam
#SBATCH --partition=gpu          # gpu-he is not accessible on this account; norm-gpu has no walltime cap
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=experiments/results/exp1_contamination_steer/slurm-%j.out
#SBATCH --error=experiments/results/exp1_contamination_steer/slurm-%j.out
#
# JOB 2 (long): 7a contamination + 7c residual-steering-with-judge over the pool.
# Thousands of local generations + ~n*12 OpenRouter judge calls (~hours).
#   # smoke test first (4 traits, regular gpu partition, separate out-dir):
#   sbatch --partition=gpu --time=00:30:00 experiments/run_contamination_steer.sbatch \
#       --limit 4 --out-dir experiments/results/exp1_contam_smoke
#   # then the full pool:
#   sbatch experiments/run_contamination_steer.sbatch
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
mkdir -p experiments/results/exp1_contamination_steer

pip install -q matplotlib   # openai/python-dotenv already in the env; no-op if present

echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ">> running exp1_contamination_steer.py $*"

# Keep CWD at the repo root so any relative --out-dir resolves intuitively; put the
# experiments dir on PYTHONPATH so `import persona_lib` / `import prompts` still work.
export PYTHONPATH="$REPO_DIR/experiments:${PYTHONPATH:-}"
python experiments/exp1_contamination_steer.py "$@"

echo ">> Done."
