#!/bin/bash

# Check if exactly one argument is passed
if [ "$#" -ne 1 ]; then
    echo "Error: Exactly one argument is required"
    exit 1
fi

# Input argument
input=$1

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

# Hack: -ffuture 0 is a workaround for blocking on a future with the trace loop
build_option="-fflow 0 -fopenmp 0 -fcuda 1 -fcuda-offline 1 -fgpu-arch $gpu_arch -findex-launch 1"
SAVEOBJ=1 STANDALONE=1 OBJNAME=${output}/pennant ${LG_RT_DIR}/../language/regent.py $input/pennant.rg $build_option
cp ${input}/*.{py,lsf,sh} ${output}
