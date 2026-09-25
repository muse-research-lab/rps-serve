#!/usr/bin/env bash
set -u

python3 train-estimator.py
python3 train-classifier.py

echo "Experment 4 (Model Training) done."