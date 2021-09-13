#!/bin/bash

module purge
module load compiler/gcc/10.2.0 tools/cmake/3.15.3

echo "Running the kernel"

srun --partition=s_standard --nodes=1 sde64 -cpx -- ./build/kernel
