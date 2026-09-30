#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"
if [ ! -x .venv/bin/python ]; then
  echo '虚拟环境不存在，请先运行 bash setup-local.sh'
  exit 1
fi
.venv/bin/python -m local_checks.check
.venv/bin/python -m pytest local_checks -q
echo '本地验证通过。结果在本项目 artifacts 目录；这不是正式模型评测或训练结果。'
