#!/bin/bash
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Unzip ---
ZIP_FILE=$(find data -maxdepth 1 -name "*.zip" | head -n 1)
if [ -z "$ZIP_FILE" ]; then
    echo "FAIL: No zip file found in data/"
    exit 1
fi

echo "Unzipping $ZIP_FILE..."
cd data
unzip -qo "$(basename "$ZIP_FILE")"
cd ..

# --- Locate directories (archive may have nested layout) ---
TRAIN_DIR=$(find data -type d -name train_images | head -n 1)
if [ -z "$TRAIN_DIR" ]; then
    echo "FAIL: train_images directory not found anywhere under data/"
    exit 1
fi

TEST_DIR=$(find data -type d -name test_images | head -n 1)
if [ -z "$TEST_DIR" ]; then
    echo "FAIL: test_images directory not found anywhere under data/"
    exit 1
fi

TRAIN_JSON=$(find data -type f -name "*train*.json" | head -n 1)
if [ -z "$TRAIN_JSON" ]; then
    echo "FAIL: Train json not found anywhere under data/"
    exit 1
fi

echo "Found train_images: $TRAIN_DIR"
echo "Found test_images:  $TEST_DIR"
echo "Found train json:   $TRAIN_JSON"

# --- Verify file counts ---
TRAIN_IMG_COUNT=$(find "$TRAIN_DIR" -type f | wc -l | tr -d ' ')
if [ "$TRAIN_IMG_COUNT" -ne 707 ]; then
    echo "FAIL: Expected 707 files in train_images, found $TRAIN_IMG_COUNT"
    exit 1
fi
echo "PASS: 707 files in train_images."

TEST_IMG_COUNT=$(find "$TEST_DIR" -type f | wc -l | tr -d ' ')
if [ "$TEST_IMG_COUNT" -ne 180 ]; then
    echo "FAIL: Expected 180 files in test_images, found $TEST_IMG_COUNT"
    exit 1
fi
echo "PASS: 180 files in test_images."
echo "PASS: Train json exists: $TRAIN_JSON"

# --- Generate GT and folds ---
echo "Generating ground truth CSV..."
python pq_eval.py gt --json "$TRAIN_JSON" --out work/gt_all.csv

echo "Generating folds CSV..."
python pq_eval.py folds --json "$TRAIN_JSON" --k 5 --seed 2026 --out work/folds.csv

# --- Verify outputs ---
echo "Verifying outputs..."
python -c '
import pandas as pd
import sys

try:
    gt = pd.read_csv("work/gt_all.csv")
    if len(gt) != 8199:
        print("FAIL: gt_all.csv has %d rows, expected 8199." % len(gt))
        sys.exit(1)

    annotator_images = gt["filament_id"].str.split("_", n=1).str[0].nunique()
    if annotator_images != 1154:
        print("FAIL: gt_all.csv has %d distinct annotator-images, expected 1154." % annotator_images)
        sys.exit(1)
    print("PASS: gt_all.csv properties verified.")

    folds = pd.read_csv("work/folds.csv")
    if len(folds) != 707:
        print("FAIL: folds.csv has %d observations, expected 707." % len(folds))
        sys.exit(1)

    sizes = sorted(folds["fold"].value_counts().tolist())
    if sizes != [141, 141, 141, 142, 142]:
        print("FAIL: Fold sizes are %s, expected [141, 141, 141, 142, 142]." % sizes)
        sys.exit(1)
    print("PASS: folds.csv properties verified (fold sizes: %s)." % sizes)

except Exception as e:
    print("FAIL: Verification crashed - %s" % e)
    sys.exit(1)
'
