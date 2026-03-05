#!/bin/bash
# SIRIUS 登录脚本
# 使用账号密码登录以启用所有在线功能

# 自动获取项目自带的SIRIUS路径（login_sirius.sh所在目录 -> 多层鉴定 -> 工作流 -> AIMS4Meta -> 源代码/SIRIUS）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SIRIUS="$(cd "$SCRIPT_DIR/../../.." && pwd)/源代码/SIRIUS/sirius-6.3.3-linux-x64/sirius/bin/sirius"

# 账号密码
USERNAME="fanhl@whut.edu.cn"
PASSWORD="Kongtong@518936"

echo "========================================="
echo "SIRIUS 登录"
echo "========================================="
echo ""

echo "账号: $USERNAME"
echo "正在登录..."
echo ""

# 执行登录（使用环境变量）
export SIRIUS_USERNAME="$USERNAME"
export SIRIUS_PASSWORD="$PASSWORD"
$SIRIUS login --user-env SIRIUS_USERNAME --password-env SIRIUS_PASSWORD

if [ $? -eq 0 ]; then
    echo ""
    echo "========================================="
    echo "✓ 登录成功！"
    echo "========================================="
    echo ""
    echo "现在可以使用 SIRIUS 的所有功能："
    echo "  - 分子式预测 (formulas/sirius)"
    echo "  - 结构数据库搜索 (structures/CSI:FingerID)"
    echo "  - 化合物分类 (canopus)"
    echo "  - 谱图库搜索 (spectra-search)"
    echo "  - 自定义数据库 (custom-db)"
    echo ""
    echo "查看账号信息："
    echo "  $SIRIUS login --show"
    echo ""
else
    echo ""
    echo "========================================="
    echo "✗ 登录失败"
    echo "========================================="
    echo ""
    echo "请检查："
    echo "  1. 网络连接是否正常"
    echo "  2. 账号密码是否正确"
    echo "  3. SIRIUS 服务器是否可访问"
    echo ""
    exit 1
fi
