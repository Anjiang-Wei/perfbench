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
    gpu_arch="sm_70" # volta
elif [[ "$HOSTNAME" == *"stanford"* ]]; then
    gpu_arch="sm_60" # pascal
else
    echo "Error: HOSTNAME does not contain 'lassen' or 'stanford'"
    exit 1
fi

# Echo the set variable
echo "gpu_arch is set to: $gpu_arch"

# Find the path to nvcc
NVCC_PATH=$(which nvcc)
# Remove '/bin/nvcc' from the path
CUDA_PATH=${NVCC_PATH%/bin/nvcc}

# Detect CUDA path
if [ -d $CUDA_PATH ]; then
    echo CUDA_PATH FOUND: $CUDA_PATH
else
    echo "CUDA not found!"
    exit 1
fi

# Assume Legion is built with --cmake
common_flags="-std=c++14 -O2 \
    -I$LG_RT_DIR -I$LG_RT_DIR/../language/build/runtime \
    -L$LG_RT_DIR/../language/build/lib -lrealm -llegion -lregent \
    -lpthread -ldl -lrt -lz \
    -I${CUDA_PATH}/include -L${CUDA_PATH}/lib64 -lcudart -lcuda -lcublas \
    -Iinclude \
    -IopenBLAS/install/include -LopenBLAS/install/lib -lopenblas \
    -w"

if ls ${input}/*.cu 1>/dev/null 2>&1; then
    # If there are .cu files, use nvcc to compile
    echo "CUDA code: using nvcc to compile"
    nvcc ${input}/*.cpp ${input}/*.cu src/* -o ${output}/main $common_flags -arch=$gpu_arch -D TACO_USE_CUDA
    cp -v openBLAS/install/lib/libopenblas.so.0 $LG_RT_DIR/../language/build/lib/lib*.so.1 ${output}
else
    echo "CPU code: using g++ to compile"
    g++ ${input}/*.cpp src/*.cpp -o ${output}/main $common_flags
    cp -v openBLAS/install/lib/libopenblas.so.0 $LG_RT_DIR/../language/build/lib/lib*.so.1 ${output}
fi

abs_input=$(realpath "$input")
for file in ${input}/*.{py,lsf,sh} ${input}/mapping*; do
    if [ -e "$file" ]; then
        ln -s "$abs_input/$(basename $file)" "${output}/$(basename $file)"
    fi
done
