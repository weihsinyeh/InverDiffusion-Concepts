#!/bin/bash

# TODO - run your inference Python3 code 
# bash hw2_2.sh $1 $2 $3
 
 # $1: path to the directory of predefined noises (e.g. “~/hw2/DDIM/input_noise”)
 # $2: path to the directory for your 10 generated images (e.g. “~/hw2/DDIM/output_images”)
 # $3: path to the pretrained model weight(e.g. “~/hw2/DDIM/UNet.pt”

 python3 PB2/PB2_inference.py $1 $2 $3