#!/bin/bash

# Base paths
output_base_dir="/project/g/r13922043/hw2/output/1030_0"

for checkpoint in $(ls "$output_base_dir"/epoch_* | sort -V); do
    epoch=$(basename "$checkpoint" | grep -oP '\d+')
    output_dir="${output_base_dir}/epoch_${epoch}"
    echo "epoch_${epoch}"
    python evaluation/grade_hw2_3.py --json_path "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input.json" --input_dir "/project/g/r13922043/hw2_data/textual_inversion" --output_dir "$output_dir"
    done
