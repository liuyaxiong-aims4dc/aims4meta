#!/bin/bash
# MS-BART训练监控脚本 (绝对路径版本)

echo "================================================================================"
echo "MS-BART训练监控"
echo "================================================================================"

# 检查训练进程
echo ""
echo "【训练进程状态】"
TRAIN_PID=$(ps aux | grep "L5_MSBART_02_train.py" | grep -v grep | awk '{print $2}')
if [ -z "$TRAIN_PID" ]; then
    echo "  状态: 未运行"
else
    echo "  状态: 运行中"
    echo "  PID: $TRAIN_PID"
    ps aux | grep "L5_MSBART_02_train.py" | grep -v grep | awk '{printf "  CPU: %s%%, 内存: %s%%, 运行时间: %s\n", $3, $4, $10}'
fi

# 检查GPU使用情况
echo ""
echo "【GPU使用情况】"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s, 使用率: %s%%, 显存: %s/%s MB\n", $1, $2, $3, $4, $5}'
else
    echo "  nvidia-smi未安装"
fi

# 检查模型输出目录
echo ""
echo "【模型输出目录】"
MODELS_DIR="/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MS-BART/models"
if [ -d "$MODELS_DIR" ]; then
    echo "  目录: $MODELS_DIR"
    ls -lh "$MODELS_DIR" | tail -n +2 | awk '{printf "  %s  %s  %s\n", $9, $5, $6" "$7}'
else
    echo "  目录不存在"
fi

# 检查训练日志
echo ""
echo "【训练日志】"
LOG_DIR="/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MS-BART/models/logs"
if [ -d "$LOG_DIR" ]; then
    LATEST_LOG=$(ls -t "$LOG_DIR"/events.out.tfevents.* 2>/dev/null | head -n 1)
    if [ -n "$LATEST_LOG" ]; then
        echo "  最新日志: $(basename $LATEST_LOG)"
        echo "  大小: $(ls -lh "$LATEST_LOG" | awk '{print $5}')"
    else
        echo "  未找到训练日志"
    fi
else
    echo "  日志目录不存在"
fi

# 检查数据文件
echo ""
echo "【数据文件】"
DATA_DIR="/stor3/AIMS4Meta/工作区/工作流/多层鉴定/L5_从头鉴定/MS-BART/data"
if [ -d "$DATA_DIR" ]; then
    echo "  目录: $DATA_DIR"
    ls -lh "$DATA_DIR"/*.pkl | awk '{printf "  %s: %s\n", $9, $5}'
else
    echo "  数据目录不存在"
fi

echo ""
echo "================================================================================"
echo "监控完成"
echo "================================================================================"
