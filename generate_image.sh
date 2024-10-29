#!/bin/bash

# Base paths
checkpoint_dir="/project/g/r13922043/hw2/checkpoints/1029_0"
output_base_dir="/project/g/r13922043/hw2/output/1029_0"

# Check if an epoch argument is passed
if [ -n "$1" ]; then
    # If an epoch is specified as the first argument, process only that epoch's checkpoint
    epoch="$1"
    checkpoint="$checkpoint_dir/fine_tuned_${epoch}.ckpt"
    
    if [ ! -f "$checkpoint" ]; then
        echo "Checkpoint for epoch $epoch not found!"
        exit 1
    fi
    
    # Process the specified checkpoint
    output_dir="${output_base_dir}/epoch${epoch}"
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
else
    # If no epoch is specified, iterate over all checkpoints in ascending order
    for checkpoint in $(ls "$checkpoint_dir"/*.ckpt | sort -V); do
        epoch=$(basename "$checkpoint" | grep -oP '\d+')
        output_dir="${output_base_dir}/epoch${epoch}"
        
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
fi

echo "All checkpoints processed."
