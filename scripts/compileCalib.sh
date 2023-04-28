#!/bin/bash

echo "Compiling calibration..."
# 进入工作目录
SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo "Current dir: ${SCRIPT_DIR}"
cd ${SCRIPT_DIR}/../
rm -rf build
mkdir build
cd build

cmake ..
make -j8
