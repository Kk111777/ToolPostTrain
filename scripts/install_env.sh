#!/usr/bin/env bash
set -euo pipefail

# Recreate the ProjectB modern runtime from the official verl v0.9.1 lock.
# This script never touches the legacy .conda environment and never runs training.

PROJECT_ROOT="${PROJECT_ROOT:-/root/autodl-tmp/ProjectB}"
UPSTREAM_DIR="${UPSTREAM_DIR:-$PROJECT_ROOT/upstream/verl-v0.9.1}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-modern}"
TOOLS_DIR="${TOOLS_DIR:-$PROJECT_ROOT/tools}"
UV_BIN="${UV_BIN:-$TOOLS_DIR/uv}"
PYPI_INDEX="${PYPI_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
EXPECTED_COMMIT="1876b06d0a3e4e71e06230be10af14492ca8a75b"

if [[ ! -x "$UV_BIN" ]]; then
    mkdir -p "$TOOLS_DIR"
    curl -LsSf --connect-timeout 8 --max-time 60 https://astral.sh/uv/install.sh \
        | UV_INSTALL_DIR="$TOOLS_DIR" sh
fi

test -d "$UPSTREAM_DIR/.git"
test "$(git -C "$UPSTREAM_DIR" rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C "$UPSTREAM_DIR" describe --tags --exact-match HEAD)" = "v0.9.1"

export UV_PROJECT_ENVIRONMENT="$VENV_DIR"
export UV_LINK_MODE="copy"
cd "$UPSTREAM_DIR"

# Official frozen runtime. The package index is only a transport mirror; versions
# and hashes remain controlled by the checked-in uv.lock.
"$UV_BIN" sync --frozen --python 3.12 --extra vllm --default-index "$PYPI_INDEX"

# Upstream v0.9.1's lock selects numpy 2.4.6 while mistral-common 1.11.3
# declares numpy<2.4 for Python 3.12. Keep the project constraint numpy>=2.0
# and apply the smallest compatibility correction without changing dependencies.
"$UV_BIN" pip install --python "$VENV_DIR/bin/python" --no-deps \
    --default-index "$PYPI_INDEX" numpy==2.3.5

"$UV_BIN" pip check --python "$VENV_DIR/bin/python"
echo "Environment packages prepared. GPU/import validation remains separate."
