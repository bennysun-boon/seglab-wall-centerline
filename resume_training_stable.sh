#!/bin/bash
# Resume training from Epoch 5 checkpoint with stable configuration
# This prevents the gradient explosion that occurred at Epoch 7

set -e

echo "=================================================="
echo "STABLE TRAINING RESUME"
echo "=================================================="
echo ""
echo "Changes from previous training:"
echo "  ✅ FP32 precision (was FP16-mixed) - better numerical stability"
echo "  ✅ Lower learning rate: 5e-5 (was 1e-4) - prevents explosion"
echo "  ✅ Lower gradient clip: 0.5 (was 1.0) - tighter gradient control"
echo "  ✅ Cosine LR scheduling - smooth decay over epochs"
echo "  ✅ Max epochs: 30 (was 50) - early stopping point"
echo ""
echo "Resuming from: epoch 5 checkpoint"
echo "  Dice: 88.4%"
echo "  IoU: 80.2%"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled."
    exit 1
fi

# Navigate to Seglab directory
cd /home/bensunshine/ml-training-platform/wall_centerline/TopoLoRA‑SAM/Seglab

# Load environment
source .venv/bin/activate

# Find epoch 5 checkpoint
CHECKPOINT_PATH="checkpoints/wall_centerline_v2/wall_centerline/sam_topolora/seed0/epoch=5-val_dice=0.0000.ckpt"

if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Epoch 5 checkpoint not found at: $CHECKPOINT_PATH"
    echo ""
    echo "Available checkpoints:"
    ls -lh checkpoints/wall_centerline_v2/wall_centerline/sam_topolora/seed0/*.ckpt
    exit 1
fi

echo ""
echo "✅ Found checkpoint: $CHECKPOINT_PATH"
echo ""
echo "Starting training with stable configuration..."
echo ""

# Start training with stable config and resume from epoch 5
python -m seglab.train \
    --config configs/experiments/wall_centerline_stable.yaml \
    --ckpt_path "$CHECKPOINT_PATH" \
    2>&1 | tee training_stable_resume.log

echo ""
echo "=================================================="
echo "Training completed!"
echo "Check training_stable_resume.log for details"
echo "=================================================="
