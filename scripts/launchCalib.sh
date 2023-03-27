#!/bin/bash

echo "Launching calibration..."
# 进入工作目录
cd /home/piggy_georgy/work_spaces/PSO

# 删除上次的标定结果
rm ./results/project_imgs/*.jpg

# 启动本次标定
./bin/main
