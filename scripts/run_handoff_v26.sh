#!/usr/bin/env bash
set -e

echo "=========================================================================="
echo "🚀 STARTING STREAMLINED 3-FLOW PIPELINE TRAINING & EVALUATION (HANDOFF v2.6)"
echo "=========================================================================="

mkdir -p results
PYTHONPATH=.
export PYTHONPATH

# 1. Step 1: Train Flow 2 (MultiFrameSRResTran) End-to-End
echo ""
echo "🔥 [STEP 1/3] Training Flow 2 (MultiFrameSRResTran) with CrossFrameTemporalSRFusion + Paired HR Loss..."
.venv/bin/python train.py -m sr_restran -n flow2 \
  --checkpoint results/flow2_swa_best.pth \
  --epochs 15 --batch-size 64 --lr 2e-4 --ocr-lr-scale 0.15 --require-gpu

# 2. Step 2: Train Flow 3 (MultiFrameSRViT) End-to-End
echo ""
echo "🔥 [STEP 2/3] Training Flow 3 (MultiFrameSRViT) with CrossFrameTemporalSRFusion + Paired HR Loss..."
.venv/bin/python train.py -m sr_vit -n flow3 \
  --checkpoint results/flow3_best.pth \
  --epochs 15 --batch-size 64 --lr 2e-4 --ocr-lr-scale 0.15 --require-gpu

# 3. Step 3: Scientific Pure Greedy CTC Evaluation across 3 Flows & Uniform Ensemble
echo ""
echo "📊 [STEP 3/3] Evaluating Clean 3-Flow Pipeline..."
echo "  • Evaluating Flow 2 (results/flow2_best.pth)..."
.venv/bin/python evaluate.py -m sr_restran --weights results/flow2_best.pth --no-tta --output-json results/eval_flow2_clean.json

echo "  • Evaluating Flow 3 (results/flow3_best.pth)..."
.venv/bin/python evaluate.py -m sr_vit --weights results/flow3_best.pth --no-tta --output-json results/eval_flow3_clean.json

echo "  • Evaluating Uniform Multi-Flow Ensemble (Flow 1 + Flow 2 + Flow 3)..."
.venv/bin/python evaluate.py -m sr_restran --weights results/flow2_best.pth \
  --ensemble-weights results/crnn_best.pth results/decoupled_best.pth results/flow3_best.pth \
  --no-tta --output-json results/eval_ensemble_clean.json

echo ""
echo "=========================================================================="
echo "🎉 ALL 3 STEPS COMPLETED SUCCESSFULLY!"
echo "=========================================================================="
