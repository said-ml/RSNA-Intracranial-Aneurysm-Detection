#!/bin/bash
# train.sh - Script to run RSNA project training without conda

# Optional: activate a virtual environment (if you use one)
# source ~/my_venv/bin/activate

# Set training parameters
DATA_DIR="./src/data"
CHECKPOINT_DIR="./src/training/checkpoints"
LOG_DIR="./logs"
CONFIG="./config.yaml"

# Make sure directories exist
mkdir -p $CHECKPOINT_DIR
mkdir -p $LOG_DIR

# Training command
python src/train.py \
    --data_dir $DATA_DIR \
    --checkpoint_dir $CHECKPOINT_DIR \
    --log_dir $LOG_DIR \
    --config $CONFIG \
    --batch_size 8 \
    --epochs 50 \
    --lr 1e-4 \
    --gpus 0 1  # list GPU ids if multi-GPU

echo "Training started. Logs are in $LOG_DIR"
