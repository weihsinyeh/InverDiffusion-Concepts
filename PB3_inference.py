import json
import os
import subprocess

def main(input_json_path, output_dir, checkpoint_path)
    with open(input_json_path, 'r') as f:
        data = json.load(f)

    for source_num, details in data.items():
        token_name = details["token_name"]
        prompts = details["prompt"]
        
        for prompt_num, prompt_text in enumerate(prompts):
            prompt_output_dir = os.path.join(output_dir, source_num)
            os.makedirs(prompt_output_dir, exist_ok=True)

            command = [ "python3", "scripts/txt2img.py",
                        "--ckpt", checkpoint_path,
                        "--prompt", prompt_text,
                        "--outdir", prompt_output_dir,
                        "--skip_grid",
                        "--prompt_num", str(prompt_num),
                        "--source_num", source_num]

            subprocess.run(command)
            print(f"Generated image for source_num={source_num}, prompt_num={prompt_num}")

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_json_path', type=str, help='path to the directory of predefined noises')
    parser.add_argument('output_dir', type=str, help='path to your output folder (e.g. “~/output_folder”)')
    parser.add_argument('checkpoint_path', type=str, help='path to the pretrained model weight (e.g. “~/hw2/personalization/model.ckpt”)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parser()
    main(args.input_json_path, args.output_folder, args.pretrained_model)