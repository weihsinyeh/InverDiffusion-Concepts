#!/bin/bash
# bash hw2_1.sh ./PB1_output
# TODO - run your inference Python3 code
if [ -z "$1" ]; then
    echo "Usage: $0 <path_to_output_directory>"
    exit 1
fi

OUTPUT_DIR="$1"

# first make the directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/mnistm" "$OUTPUT_DIR/svhn"

# $ python3 PB1_inference.py ./PB1_output
python3 PB1/PB1_inference.py "$OUTPUT_DIR"