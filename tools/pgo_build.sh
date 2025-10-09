#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
BUILD_DIR="${PROJECT_ROOT}/build"
PGO_BUILD_DIR="${BUILD_DIR}/pgo"
PROFILE_DIR="${PGO_BUILD_DIR}/profiles"

usage() {
  cat <<'USAGE'
利用方法: ./tools/pgo_build.sh <mode>
  release        通常の Release ビルド (-O3/-flto)
  pgo-generate   PGO プロファイル生成ビルドとプロファイル収集
  pgo-use        収集済みプロファイルを用いた最適化ビルド
  clean          PGO 用のビルド成果物とプロファイルを削除
USAGE
}

ensure_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    if ! command -v "${PYTHON}" >/dev/null 2>&1; then
      echo "指定された PYTHON='${PYTHON}' が実行できません" >&2
      exit 1
    fi
    echo "${PYTHON}"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo python3
  elif command -v python >/dev/null 2>&1; then
    echo python
  else
    echo "Python が見つかりません" >&2
    exit 1
  fi
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

action="$1"

case "${action}" in
  release)
    mkdir -p "${BUILD_DIR}"
    cmake -S "${PROJECT_ROOT}" -B "${BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release
    cmake --build "${BUILD_DIR}"
    ;;
  pgo-generate)
    mkdir -p "${PGO_BUILD_DIR}"
    mkdir -p "${PROFILE_DIR}"
    PY_CMD=$(ensure_python)
    cmake -S "${PROJECT_ROOT}" -B "${PGO_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
      -DTLG_PGO_MODE=generate \
      -DTLG_PGO_PROFILE_DIR="${PROFILE_DIR}"
    cmake --build "${PGO_BUILD_DIR}"
    "${PY_CMD}" "${PROJECT_ROOT}/test/run_roundtrip.py" \
      --tlgconv "${PGO_BUILD_DIR}/tlgconv" \
      --images "${PROJECT_ROOT}/test/images"
    ;;
  pgo-use)
    if [[ ! -d "${PROFILE_DIR}" ]]; then
      echo "プロファイルディレクトリ ${PROFILE_DIR} が存在しません。まず pgo-generate を実行してください" >&2
      exit 1
    fi
    mkdir -p "${PGO_BUILD_DIR}"
    cmake -S "${PROJECT_ROOT}" -B "${PGO_BUILD_DIR}" -DCMAKE_BUILD_TYPE=Release \
      -DTLG_PGO_MODE=use \
      -DTLG_PGO_PROFILE_DIR="${PROFILE_DIR}"
    cmake --build "${PGO_BUILD_DIR}"
    ;;
  clean)
    if [[ -d "${PGO_BUILD_DIR}" ]]; then
      rm -rf "${PGO_BUILD_DIR}"
    fi
    ;;
  *)
    usage
    exit 1
    ;;
esac
