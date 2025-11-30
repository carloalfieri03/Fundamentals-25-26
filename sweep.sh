#!/bin/bash

# Define the parameter arrays
depths=(1 2 3)
widths=(16 32 64 128)

# Iterate through all combinations
for depth in "${depths[@]}"; do
    for width in "${widths[@]}"; do

        echo "Running experiment with depth=$depth, width=$width"
        python src/experiment.py net.depth=$depth net.width=$width
    done
done