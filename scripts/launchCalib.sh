#!/bin/bash

echo "Launching calibration..."
# 进入工作目录
SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo "Current dir: ${SCRIPT_DIR}"
cd ${SCRIPT_DIR}/../

# 删除上次的标定结果
rm ./results/project_imgs/*.jpg

# 启动本次标定
./bin/main
