#!/bin/bash
# bash ./scripts-search/NAS-Bench-201/train-a-net.sh resnet 16 5
echo script name: $0
echo $# arguments
if [ "$#" -ne 3 ] ;then
  echo "Input illegal number of parameters " $#
  echo "Need 3 parameters for network, channel, num-of-cells"
  exit 1
fi
if [ "$TORCH_HOME" = "" ]; then
  echo "Must set TORCH_HOME envoriment variable for data dir saving"
  exit 1
else
  echo "TORCH_HOME : $TORCH_HOME"
fi

model=$1
channel=$2
num_cells=$3

save_dir=./output/NAS-BENCH-201-4/

srun -K -N1 --gpus=1 --gpus-per-task=1 --cpus-per-gpu=4 --mem-per-gpu=16G -p batch\
     --container-mounts=/netscratch:/netscratch,/ds:/ds,`pwd`:`pwd` \
     --container-image=/netscratch/enroot/dlcc_pytorch_20.10.sqsh \
     --container-workdir=`pwd` \
     --export="NCCL_SOCKET_IFNAME=bond,NCCL_IB_HCA=mlx5" \
      python3 ./exps/NAS-Bench-201/main.py \
	--mode specific-${model} --save_dir ${save_dir} --max_node 6 \
	--datasets cifar100 \
	--use_less 0 \
	--splits         0 \
	--xpaths $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
		 $TORCH_HOME/cifar.python \
	--channel ${channel} --num_cells ${num_cells} \
	--workers 4 \
	--seeds 777 888 999