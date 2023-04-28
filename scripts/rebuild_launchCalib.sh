#!/bin/bash

echo "Rebuilding and launching calibration..."
# 进入工作目录
SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo "Current dir: ${SCRIPT_DIR}"
cd ${SCRIPT_DIR}/../
./compileCalib.sh
./launchCalib.sh