#!/usr/bin/env bash
# 現在の作業ディレクトリを .gitignore の内容を反映して zip 化するスクリプト
set -euo pipefail

if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "git リポジトリ内で実行してください" >&2
  exit 1
fi

timestamp="$(date +%y%m%d_%H%M%S)"
output="/tmp/tlgconv_${timestamp}.zip"

tmp_list=$(mktemp)
trap 'rm -f "${tmp_list}"' EXIT

git ls-files --cached --others --exclude-standard >"${tmp_list}"

if [ -s "${tmp_list}" ]; then
  zip -q "${output}" -@ <"${tmp_list}"
else
  # 空の zip を生成するために一時的にダミーファイルを追加する
  zip -q "${output}" -- .gitignore >/dev/null 2>&1 || true
  zip -dq "${output}" .gitignore >/dev/null 2>&1 || true
fi

echo "${output}"
