#!/usr/bin/env bash
# Create and provision the bridge-eval environment (conda or venv fallback).
# Usage: bash setup_env.sh
set -euo pipefail

ENV_NAME="bridge-eval"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# redeal is installed from GitHub (it is not on PyPI), so git is required.
if ! command -v git >/dev/null 2>&1; then
  echo "ERROR: git not found on PATH (needed to install redeal from GitHub)." >&2
  exit 1
fi

MINICONDA_ACTIVATE="${HOME}/code/miniconda3/bin/activate"
if [ -f "${MINICONDA_ACTIVATE}" ]; then
  # shellcheck disable=SC1090
  source "${MINICONDA_ACTIVATE}"
fi

if command -v conda >/dev/null 2>&1; then
  # ── conda path ──────────────────────────────────────────────────────────────
  eval "$(conda shell.bash hook)"

  if conda env list | grep -qE "^${ENV_NAME}\s"; then
    echo "Environment '${ENV_NAME}' already exists; updating packages."
  else
    echo "Creating conda environment '${ENV_NAME}' (python 3.10)..."
    conda create -n "${ENV_NAME}" python=3.10 -y
  fi

  conda activate "${ENV_NAME}"
  echo "Installing Python dependencies..."
  pip install -r "${SCRIPT_DIR}/requirements.txt"
  ACTIVATE_CMD="conda activate ${ENV_NAME}"
else
  # ── venv fallback ────────────────────────────────────────────────────────────
  echo "conda not found; falling back to Python venv."
  VENV_DIR="${SCRIPT_DIR}/.venv"

  if [ ! -d "${VENV_DIR}" ]; then
    echo "Creating venv at ${VENV_DIR}..."
    python3 -m venv "${VENV_DIR}"
  else
    echo "venv already exists at ${VENV_DIR}; updating packages."
  fi

  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  echo "Installing Python dependencies..."
  pip install --upgrade pip -q
  pip install -r "${SCRIPT_DIR}/requirements.txt"
  ACTIVATE_CMD="source ${VENV_DIR}/bin/activate"
fi

echo
echo "Verifying core imports (one package at a time)..."
PACKAGES=(redeal endplay pydantic openai matplotlib pandas)
missing=()
for pkg in "${PACKAGES[@]}"; do
  if version=$(python -c "import ${pkg} as m; print(getattr(m, '__version__', '?'))" 2>/dev/null); then
    printf '  OK   %-12s %s\n' "${pkg}" "${version}"
  else
    printf '  FAIL %-12s (import failed)\n' "${pkg}"
    missing+=("${pkg}")
  fi
done

echo
if [ "${#missing[@]}" -ne 0 ]; then
  echo "ERROR: ${#missing[@]} package(s) failed to import: ${missing[*]}" >&2
  echo "Re-run the install: pip install -r requirements.txt" >&2
  echo "(redeal comes from git+https://github.com/anntzer/redeal, not PyPI)" >&2
  exit 1
fi

echo "All core packages import OK."
echo
echo "Activate with:  ${ACTIVATE_CMD}"
