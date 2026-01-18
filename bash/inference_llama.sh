#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1
INPUT_FILE=/path/to/your/input.json
MODEL=/path/to/your/model


echo "Running QA script with fixed parameters:"
echo "Model: $MODEL"
echo "Input file: $INPUT_FILE"


python inference_llama.py --input "$INPUT_FILE"   --model "$MODEL"  