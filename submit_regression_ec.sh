#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time=00-5:00:00
#SBATCH --signal=INT@300
#SBATCH --job-name=regression_ec_mimic
#SBATCH --output="/gpfs/scratch/x2020fpt/MIMIC-get/slurm_logs/regression_ec_mimic__%j_%u.out"
#SBATCH --mail-user=x2020fpt@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=TIME_LIMIT_90
#SBATCH --profile=all
#SBATCH --cpus-per-task=16
#SBATCH --mem=180GB

PROJECT="/gpfs/scratch/x2020fpt/MIMIC-get"

echo "Loading required modules: StdEnv/2020 gcc/9.3.0 arrow/5.0.0"
module load gcc/9.3.0 python scipy-stack

echo "Setting up python venv"
cd $PROJECT
source $PROJECT/.venv/bin/activate

PYTHON=$(which python)

echo "Job starting at $(date)"
$PYTHON /gpfs/scratch/x2020fpt/MIMIC-get/regression_ec.py && \
echo "Job done at $(date)"
