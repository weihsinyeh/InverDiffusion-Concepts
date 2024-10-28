#!/bin/bash

# Base paths
checkpoint_dir="/project/g/r13922043/hw2/checkpoints/1028_data_aug"
output_base_dir="/project/g/r13922043/hw2/output/1028_data_aug"

# Iterate over all checkpoints in the specified directory
for checkpoint in "$checkpoint_dir"/*.ckpt; do
    # Extract epoch number from the checkpoint filename (assuming a consistent naming pattern like 'fine_tuned_{epoch}.ckpt')
    epoch=$(basename "$checkpoint" | grep -oP '\d+')

    # Set output directory based on epoch number
    output_dir="${output_base_dir}/epoch${epoch}"

    # Create output directory if it doesn't exist
    mkdir -p "$output_dir/0"
    cd stable-diffusion
    # Run txt2img commands with different prompts
    CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py --ckpt "$checkpoint" --prompt "A <new1> shepherd posing proudly on a hilltop with Mount Fuji in the background." --outdir "$output_dir/0" --skip_grid --prompt_num 0 --source_num 0
    CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py --ckpt "$checkpoint" --prompt "A <new1> perched on a park bench with the Colosseum looming behind." --outdir "$output_dir/0" --skip_grid --prompt_num 1 --source_num 0
    CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py --ckpt "$checkpoint" --prompt "A dog shepherd posing proudly on a hilltop with Mount Fuji in the background." --outdir "$output_dir/0" --skip_grid --prompt_num 2 --source_num 0
    CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py --ckpt "$checkpoint" --prompt "A dog perched on a park bench with the Colosseum looming behind." --outdir "$output_dir/0" --skip_grid --prompt_num 3 --source_num 0
    cd ..
    # Run the evaluation script
    python evaluation/grade_hw2_3.py --json_path "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input_0.json" --input_dir "/project/g/r13922043/hw2_data/textual_inversion" --output_dir "$output_dir"
done

echo "All checkpoints processed."
