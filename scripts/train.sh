#!/bin/bash
# Training script for DeepLOB model

set -e

echo "Starting DeepLOB training..."

# Default arguments
HORIZON=${HORIZON:-0}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-32}
LR=${LR:-0.001}

# Run training
python src/training/train.py \
    --horizon $HORIZON \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --lr $LR \
    "$@"

echo "Training complete!"
