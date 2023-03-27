#!/bin/bash

echo "Rebuilding and launching calibration..."
# 进入工作目录
cd /home/piggy_georgy/work_spaces/PSO/scripts
./compileCalib.sh
./launchCalib.sh