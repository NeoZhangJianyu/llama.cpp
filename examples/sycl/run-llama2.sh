#!/bin/bash

#  MIT license
#  Copyright (C) 2024 Intel Corporation
#  SPDX-License-Identifier: MIT

INPUT2="Building a website can be done in 10 simple steps:\nStep 1:"
source /opt/intel/oneapi/setvars.sh

if [ $# -gt 0 ]; then
    GGML_SYCL_DEVICE=$1
else
    GGML_SYCL_DEVICE=0
fi
echo "use $GGML_SYCL_DEVICE as main GPU"
#export GGML_SYCL_DEBUG=1
./build/bin/main -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 100 -e -ngl 33 -s 0 -mg $GGML_SYCL_DEVICE
#./build/bin/main -m models/llama-2-7b.Q4_0.gguf -p "${INPUT2}" -n 5 -e -ngl 33 -t 1 -s 0

