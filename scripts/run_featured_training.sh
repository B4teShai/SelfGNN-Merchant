#!/bin/bash
# Run C2, C3, C4 training sequentially after feature extraction is complete.
# Run from project root: bash scripts/run_featured_training.sh
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FEAT_DIR="$ROOT/selfGNN-Feature"
DATA_DIR="$ROOT/Datasets/yelp-merchant"
RESULTS_DIR="$ROOT/Results"

# Verify feature files exist
for f in user_features.npy merchant_features.npy edge_weights.pkl; do
    if [ ! -f "$DATA_DIR/$f" ]; then
        echo "ERROR: $DATA_DIR/$f not found. Run feature_extractor.py first."
        exit 1
    fi
done
echo "Feature files verified."

COMMON="--data yelp-merchant --epoch 150 --tstEpoch 3 --patience 20 --device cuda"

echo "========================================"
echo "Config 2: Edge features only"
echo "========================================"
cd "$FEAT_DIR"
python train.py $COMMON \
    --use_edge_features \
    --save_path config2_edge \
    2>&1 | tee "$RESULTS_DIR/config2_edge_training.log"

echo "========================================"
echo "Config 3: Node features only"
echo "========================================"
python train.py $COMMON \
    --use_node_features \
    --save_path config3_node \
    2>&1 | tee "$RESULTS_DIR/config3_node_training.log"

echo "========================================"
echo "Config 4: Edge + Node features"
echo "========================================"
python train.py $COMMON \
    --use_edge_features --use_node_features \
    --save_path config4_both \
    2>&1 | tee "$RESULTS_DIR/config4_both_training.log"

echo "========================================"
echo "All training complete. Running analysis..."
echo "========================================"
cd "$ROOT"
python analysis/compare_results.py
python analysis/generate_paper_tables.py
echo "Done. Results in Results/ and paper_figures/"
