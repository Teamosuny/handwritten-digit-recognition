#!/bin/bash
# 启动脚本，确保在正确的虚拟环境中运行Python程序

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 激活虚拟环境
source "$SCRIPT_DIR/pytorch_env/bin/activate"

# 运行Python脚本
python test.py

# 可选：退出虚拟环境（脚本结束后会自动退出）
deactivate