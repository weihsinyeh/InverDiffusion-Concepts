cd stable-diffusion
CUDA_VISIVLE_DEVICES=0 python scripts/txt2img.py --ckpt "/project/g/r13922043/hw2/checkpoints/1028_data_aug/fine_tuned_10.ckpt" --prompt "A <new1> shepherd posing proudly on a hilltop with Mount Fuji in the background."  --outdir "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/output_folder_example_data_augmentation_ep10/0" --skip_grid --prompt_num 0 --source_num 0
CUDA_VISIVLE_DEVICES=0 python scripts/txt2img.py --ckpt "/project/g/r13922043/hw2/checkpoints/1028_data_aug/fine_tuned_10.ckpt" --prompt "A <new1> perched on a park bench with the Colosseum looming behind." --outdir "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/output_folder_example_data_augmentation_ep10/0" --skip_grid --prompt_num 1 --source_num 0
CUDA_VISIVLE_DEVICES=0 python scripts/txt2img.py --ckpt "/project/g/r13922043/hw2/checkpoints/1028_data_aug/fine_tuned_10.ckpt" --prompt "A dog shepherd posing proudly on a hilltop with Mount Fuji in the background."  --outdir "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/output_folder_example_data_augmentation_ep10/0" --skip_grid --prompt_num 2 --source_num 0
CUDA_VISIVLE_DEVICES=0 python scripts/txt2img.py --ckpt "/project/g/r13922043/hw2/checkpoints/1028_data_aug/fine_tuned_10.ckpt" --prompt "A dog perched on a park bench with the Colosseum looming behind."  --outdir "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/output_folder_example_data_augmentation_ep10/0" --skip_grid --prompt_num 3 --source_num 0
cd ..
python evaluation/grade_hw2_3.py --json_path "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/stable-diffusion/input_0.json" --input_dir "/project/g/r13922043/hw2_data/textual_inversion" --output_dir "/tmp2/r13922043/dlcv-fall-2024-hw2-weihsinyeh/output_folder_example_data_augmentation_ep10"
