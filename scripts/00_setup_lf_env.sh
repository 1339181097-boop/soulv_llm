#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

CONDA_EXE="${CONDA_EXE:-$HOME/miniconda3/bin/conda}"
ENV_NAME="${ENV_NAME:-lf}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PROJECT_ROOT="${PROJECT_ROOT:-$DEFAULT_PROJECT_ROOT}"
LLAMAFACTORY_HOME="${LLAMAFACTORY_HOME:-/root/llama-factory}"
LLAMAFACTORY_REPO="${LLAMAFACTORY_REPO:-https://github.com/hiyouga/LLaMA-Factory.git}"
LLAMAFACTORY_REF="${LLAMAFACTORY_REF:-v0.9.4}"
INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "conda executable not found: $CONDA_EXE"
  echo "Set CONDA_EXE explicitly, for example: CONDA_EXE=/root/miniconda3/bin/conda"
  exit 1
fi

if ! command -v git >/dev/null 2>&1; then
  echo "git is required to clone or reuse the official LLaMA-Factory repo."
  exit 1
fi

if ! "$CONDA_EXE" run -n "$ENV_NAME" python -V >/dev/null 2>&1; then
  "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements-lf.txt"

if [[ ! -d "$LLAMAFACTORY_HOME/.git" ]]; then
  git clone --depth 1 --branch "$LLAMAFACTORY_REF" "$LLAMAFACTORY_REPO" "$LLAMAFACTORY_HOME"
else
  echo "Using existing LLaMA-Factory checkout at $LLAMAFACTORY_HOME"
fi

"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install -e "$LLAMAFACTORY_HOME"

if [[ -f "$LLAMAFACTORY_HOME/requirements/metrics.txt" ]]; then
  "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install \
    -r "$LLAMAFACTORY_HOME/requirements/metrics.txt"
fi

if [[ -f "$LLAMAFACTORY_HOME/requirements/deepspeed.txt" ]]; then
  "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install \
    -r "$LLAMAFACTORY_HOME/requirements/deepspeed.txt"
fi

if [[ "$INSTALL_FLASH_ATTN" == "1" ]]; then
  "$CONDA_EXE" run -n "$ENV_NAME" python -m pip install -r "$PROJECT_ROOT/requirements-lf-flashattn.txt"
fi

echo "lf environment is ready."
echo "  conda env: $ENV_NAME"
echo "  python: $PYTHON_VERSION"
echo "  llamafactory_home: $LLAMAFACTORY_HOME"
