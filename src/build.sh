#!/bin/bash

module purge
module load compiler/gcc/10.2.0 tools/cmake/3.15.3

echo "Deleting the current build artifacts"
rm -r -f ./build
mkdir build
cd build

echo "Generating Makefile"

srun --partition=s_standard --nodes=1 sde64 -cpx -- cmake ..

echo "Executing Makefile"

srun --partition=s_standard --nodes=1 sde64 -cpx -- make

echo "finished