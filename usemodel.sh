#!/bin/bash

#SBATCH --job-name=usemodel
#SBATCH --partition=ADA6000,L40S,A100 # 用sinfo命令可以看到所有队列，默认为debug队列


#SBATCH --nodes=1           # 申请的节点数量，单机训练无需调整
#SBATCH --ntasks-per-node=1 # 若多卡或多进程，请调整此参数
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16  # 每个进程的CPU数量
#SBATCH --gres=gpu:1        # 若使用2块卡，则gres=gpu:2

#SBATCH --output=./logs/use_model_%j.out     # 设置标准输出的文件路径
#SBATCH --error=./logs/use_model_%j.err      # 设置标准错误的文件路径

python -u use_model.py
