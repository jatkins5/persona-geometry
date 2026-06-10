#!/usr/bin/env bash
#SBATCH --job-name=persona-geo
#SBATCH --partition=gpu          # use gpu-he for long runs, gpu-debug for quick tests
#SBATCH --gres=gpu:1             # one GPU is plenty for Qwen2.5-7B (~16GB)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=runs/%x-%j.out
#SBATCH --error=runs/%x-%j.out
#
# Batch-execute the SAME notebook that you edit interactively. papermill runs
# personas.ipynb top-to-bottom on the GPU node and writes an executed copy
# (with outputs) under runs/ -- the .ipynb stays the single source of truth.
#
# Submit, optionally overriding the 'parameters' cell:
#   sbatch run_experiment.sbatch                       # defaults
#   sbatch run_experiment.sbatch SECTION=exp3 N=64     # override params
set -euo pipefail

PY_MODULE="python/3.11.11-5e66"
ENV_DIR="$HOME/envs/persona-geometry"
REPO_DIR="$SLURM_SUBMIT_DIR"
NOTEBOOK="personas.ipynb"

module load "$PY_MODULE"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

export HF_HOME="/oscar/scratch/$USER/hf"   # weights cached on scratch, off home quota
export TOKENIZERS_PARALLELISM=false

cd "$REPO_DIR"
mkdir -p runs

# Turn "KEY=VALUE" args into papermill "-p KEY VALUE" parameter overrides.
PM_ARGS=()
for kv in "$@"; do
  PM_ARGS+=( -p "${kv%%=*}" "${kv#*=}" )
done

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="runs/${NOTEBOOK%.ipynb}-${STAMP}-job${SLURM_JOB_ID}.ipynb"

echo ">> GPU:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ">> papermill $NOTEBOOK -> $OUT  ${PM_ARGS[*]:-(no overrides)}"

papermill "$NOTEBOOK" "$OUT" "${PM_ARGS[@]}" --log-output

echo ">> Done. Executed notebook with outputs: $OUT"
