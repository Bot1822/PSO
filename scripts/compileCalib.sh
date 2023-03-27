#!/bin/bash

echo "Compiling calibration..."
# 进入工作目录
cd /home/piggy_georgy/work_spaces/PSO
rm -rf build
mkdir build
cd build

cmake ..
make -j8
