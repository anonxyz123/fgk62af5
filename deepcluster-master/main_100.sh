# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

DIR="/datasets01/imagenet_full_size/061417/train"
ARCH="vgg16"
LR=0.05
WD=-5
K=1000
WORKERS=12
EXP="/home/${USER}/dc_exps/100"

mkdir -p ${EXP}

CUDA_VISIBLE_DEVICES=1 python main.py ${DIR} --exp ${EXP} --arch ${ARCH} \
  --lr ${LR} --wd ${WD} --k ${K} --verbose --workers ${WORKERS} --sobel
