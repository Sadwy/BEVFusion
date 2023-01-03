#!/usr/bin/env bash

CONFIG=$1  # 配置文件
GPUS=$2  # 使用的GPU数量，传入值=8
PORT=$((RANDOM + 10000))  # 随机选择端口

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3}

"""第一行指令：python包搜索路径
$PYTHONPATH是环境变量，用于指定python程序搜索module时的路径列表
$0是变量，表示正在执行的文件名
dirname是Linux指令，返回文件所在路径，此处返回的结果是'tools/'
dirname，需要使用$()将dirname返回的结果转换为变量
/..是返回上一级目录
此处是让$PYTHONPATH增加了主路径/.sh文件所在路径的上一层，即'BEVFusion/'
类似于 PYTHONPATH='BEVFusion/':$PYTHONPATH
指令末尾有反斜杠\，说明3行总共是一条指令，此时$PYTHONPATH的赋值生效范围
仅仅限于这一条指令，该指令结束后赋值失效
"""

"""第二行指令：torch.distributed.launch及其可选参数
# 感觉此处加不加-m参数没有影响（这是主观臆断，可能错误）
torch.distributed.launch：分布式训练包
--nproc_per_node：每个node的进程数量，推荐等于GPU数量
--master_port：分布式训练时节点之间的通信端口
"""

"""第三行指令：torch.distributed.launch的位置参数
第一个是训练文件train.py，$(dirname "$0")的值是"tools/"
剩下的参数都将传给train.py作为其参数
${@:3}的意思是，将.sh文件接收到的第3个及后面的所有参数传到这里。具体尚未深入了解
${@:3}表示，.sh文件第3个及后面的所有参数会传给train.py作为其参数
"""
