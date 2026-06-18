#!/usr/bin/env bash
#SBATCH --job-name=exp2-variance
#SBATCH --partition=batch        # CPU only -- no model load, just cached vectors + PCA
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=experiments/results/analysis/slurm-%j.out
#SBATCH --error=experiments/results/analysis/slurm-%j.out
#
# Light cached-data analyses that don't need a GPU. First arg = script under experiments/.
#   sbatch experiments/run_analysis.sbatch exp2_manifold.py --mode variance --model llama3.1-8b \
#       --out-dir experiments/results/analysis
#   sbatch experiments/run_analysis.sbatch plot_das_compare.py --results dirA,dirB --out out.png
set -euo pipefail

module load python/3.11.11-5e66
# shellcheck disable=SC1091
source "$HOME/envs/persona-geometry/bin/activate"

cd "$SLURM_SUBMIT_DIR"
mkdir -p experiments/results/analysis
pip install -q matplotlib

export PYTHONPATH="$SLURM_SUBMIT_DIR/experiments:${PYTHONPATH:-}"
SCRIPT="$1"; shift
python "experiments/$SCRIPT" "$@"

echo ">> Done."
