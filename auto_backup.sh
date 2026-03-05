#!/bin/bash

# AIMS4Meta 自动备份脚本
# 每天凌晨 2:00 自动备份到 GitHub

cd /stor3/AIMS4Meta

# 添加所有修改
git add -A

# 检查是否有未提交的更改
if git diff --staged --quiet; then
    # 没有更改，退出
    exit 0
fi

# 添加时间戳提交
DATE=$(date '+%Y-%m-%d %H:%M')
git commit -m "自动备份: $DATE"

# 推送到 GitHub
git push origin master
