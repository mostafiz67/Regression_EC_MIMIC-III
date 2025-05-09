#!/bin/bash

PROJECT="$HOME/MIMIC"
LIGHTNING_LOGS="$PROJECT/logs/dl_logs"

module load gcc/9.3.0 arrow python scipy-stack
source .venv/bin/activate
pip install pyarrow

PYTHON=$(which python)

echo "Job starting at $(date)"
tensorboard --logdir=$LIGHTNING_LOGS --host 0.0.0.0 --load_fast false
