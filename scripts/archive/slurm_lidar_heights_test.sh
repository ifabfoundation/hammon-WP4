#!/bin/bash
#SBATCH --job-name=lidar_height_test
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --partition=cpu             # adjust if your cluster uses a different default partition
#SBATCH --output=logs/lidar_height_test_%j.out
#SBATCH --error=logs/lidar_height_test_%j.err

set -euo pipefail

echo "=========================================="
echo "HAMMON - LiDAR Height Extraction (Test)"
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Zone: (none)"
echo "Runs: (none)"
echo "Limit: ${LIMIT:-20} buildings"
echo "Start time: $(date)"
echo "=========================================="

cd /home/projects/hammon
mkdir -p logs outputs/lidar
TMPDIR_LIDAR=${SLURM_TMPDIR:-/home/projects/hammon/data/tmp_lidar_${SLURM_JOB_ID:-test}}
mkdir -p "${TMPDIR_LIDAR}"
export PYTHONPATH="/home/projects/hammon/src:${PYTHONPATH:-}"

srun_env_creds() {
  export AWS_ACCESS_KEY_ID="${AWS_ACCESS_KEY_ID:-YOUR_ACCESS_KEY_HERE}"
  export AWS_SECRET_ACCESS_KEY="${AWS_SECRET_ACCESS_KEY:-YOUR_SECRET_KEY_HERE}"
}

PYTHON_BIN="/home/projects/hammon/hammon-env/bin/python"

srun_env_creds

srun "${PYTHON_BIN}" -m src.lidar.batch_runner \
  --limit "${LIMIT:-20}" \
  --temp-dir "${TMPDIR_LIDAR}" \
  --output-format geojson \
  --output outputs/lidar/heights_test.geojson

echo "Done at $(date)"
