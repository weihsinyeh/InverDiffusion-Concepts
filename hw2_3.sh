#!/bin/bash

# TODO - run your inference Python3 code

# $ bash hw2_3.sh $1 $2 $3
# $ bash hw2_3.sh "/project/g/r13922043/hw2_data/textual_inversion/input.json" "./PB3_output_folder" "./stable-diffusion/ldm/models/stable-diffusion-v1/model.ckpt"
# $1: path to the json file containing the testing prompt 
# (e.g. “~/hw2_data/textual_inversion/input.json”)
# $2: path to your output folder (e.g. “~/output_folder”)
# $3: path to the pretrained model weight (e.g. “~/hw2/personalization/model.ckpt”)
python3 stable-diffusion/PB3_inference.py $1 $2 $3

