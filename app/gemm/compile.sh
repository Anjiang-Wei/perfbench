#!/bin/bash

# Check if exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Error: Exactly one argument is required"
    exit 1
fi

# Input argument
input=$1
input="${input%/}"

# Check if the argument contains "src"
if [[ "$input" == *"src"* ]]; then
    # Replace "src" with "bin"
    output=${input/src/bin}
    # Check if the directory exists
    if [ -d "$output" ]; then
        echo "Directory $output exists. Deleting it."
        rm -rf "$output"
    fi
    mkdir $output
    echo "$output contains the compiled binary"
else
    echo "Error: Input argument does not contain 'src'"
    exit 1
fi

if [ -z "${LG_RT_DIR}" ]; then
    echo "Error: LG_RT_DIR environment variable is not set"
    exit 1
else
    echo "LG_RT_DIR is set to ${LG_RT_DIR}"
fi

# Check the value of HOSTNAME and set the variable accordingly
if [[ "$HOSTNAME" == *"lassen"* ]]; then
    gpu_arch="volta"
elif [[ "$HOSTNAME" == *"stanford"* ]]; then
    gpu_arch="pascal"
else
    echo "Error: HOSTNAME does not contain 'lassen' or 'stanford'"
    exit 1
fi

# Echo the set variable
echo "gpu_arch is set to: $gpu_arch"

# Detect CUDA path
if [ -d "/usr/local/cuda" ]; then
  CUDA_PATH="/usr/local/cuda"
  echo CUDA_PATH FOUND: $CUDA_PATH
else
  echo "CUDA not found!"
  exit 1
fi

g++ ${input}/main.cpp ${input}/taco-generated.cpp -o ${output}/main -std=c++14 -O2 \
    -I$LG_RT_DIR -I$LG_RT_DIR/../bindings/regent -L$LG_RT_DIR/../bindings/regent -lrealm -llegion -lregent \
    -lpthread -ldl -lrt -lz \
    -I${CUDA_PATH}/include -L${CUDA_PATH}/lib64 -lcudart -lcuda \
    -Iinclude \
    -IopenBLAS/install/include -LopenBLAS/install/lib -lopenblas

abs_input=$(realpath "$input")
for file in ${input}/*.{py,lsf,sh} ${input}/mapping*; do
    if [ -e "$file" ]; then
        ln -s "$abs_input/$(basename $file)" "${output}/$(basename $file)"
    fi
done
