#!/bin/bash

# Base paths
output_base_dir="/project/g/r13922043/hw2/output/1030_0"
max_epochs=20

# Iterate through epoch numbers from 0 to max_epochs
for epoch in $(seq 0 $max_epochs); do
    output_dir="${output_base_dir}/epoch_${epoch}"

    # Check if the output directory exists
    if [ -d "$output_dir" ]; then
        echo "Processing ${output_dir}..."
        
        # Run the evaluation script
        # /tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input_1.json
        # /project/g/r13922043/hw2_data/textual_inversion/input.json
        python3 evaluation/grade_hw2_3.py --json_path "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input_0.json" --input_dir "/project/g/r13922043/hw2_data/textual_inversion" --output_dir "$output_dir"
    else
        echo "Directory ${output_dir} does not exist. Skipping..."
    fi
done