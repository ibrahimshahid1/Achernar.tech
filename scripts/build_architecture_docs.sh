#!/usr/bin/env bash
# Build current standalone specifications and searchable text companions.
set -euo pipefail

doc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
doc_pdftotext="${PDFTOTEXT_BIN:-pdftotext}"
doc_text_python="${PDFTEXT_PYTHON:-python3}"
doc_target="${1:-current}"

case "$doc_target" in
  current) doc_sources=(achernar_v6 achernar_data_collection_v6) ;;
  v6) doc_sources=(achernar_v6) ;;
  collection) doc_sources=(achernar_data_collection_v6) ;;
  v5) doc_sources=(achernar_v5) ;;
  *) printf '%s\n' 'Usage: build_architecture_docs.sh [current|v6|collection|v5]' >&2; exit 2 ;;
esac

if ! command -v latexmk >/dev/null 2>&1; then
  printf '%s\n' 'Missing latexmk. Install a TeX distribution before building the architecture docs.' >&2
  exit 1
fi
if ! command -v "$doc_pdftotext" >/dev/null 2>&1; then
  if ! "$doc_text_python" -c 'import pypdf' >/dev/null 2>&1; then
    printf '%s\n' 'Text extraction requires pdftotext or PDFTEXT_PYTHON pointing to Python with pypdf.' >&2
    exit 1
  fi
fi

cd "$doc_repo_root"
for doc_name in "${doc_sources[@]}"; do
  doc_build_dir="$doc_repo_root/tmp/pdfs/$doc_name-build"
  mkdir -p "$doc_build_dir"
  latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error \
    -pdflatex='pdflatex -no-shell-escape %O %S' \
    -outdir="$doc_build_dir" "$doc_name.tex"
  if command -v "$doc_pdftotext" >/dev/null 2>&1; then
    "$doc_pdftotext" -layout "$doc_build_dir/$doc_name.pdf" "$doc_build_dir/$doc_name.txt"
  else
    "$doc_text_python" - "$doc_build_dir/$doc_name.pdf" "$doc_build_dir/$doc_name.txt" <<'PY'
import sys
from pathlib import Path
from pypdf import PdfReader
reader = PdfReader(sys.argv[1])
Path(sys.argv[2]).write_text("\n\f\n".join(page.extract_text(extraction_mode="layout") for page in reader.pages), encoding="utf-8")
PY
  fi
  cp "$doc_build_dir/$doc_name.pdf" "$doc_repo_root/$doc_name.pdf"
  cp "$doc_build_dir/$doc_name.txt" "$doc_repo_root/$doc_name.txt"
  printf 'Built %s.pdf and %s.txt. Review all rendered pages before publishing.\n' "$doc_name" "$doc_name"
done
