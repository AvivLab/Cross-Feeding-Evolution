#!/usr/bin/env bash
# Render figures/Used/mccm_two_conditions_tikz.tex -> figures/Used/mccm_two_conditions.pdf
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -d "$DRAFT_DIR/figures/Used" ]]; then
  FIG_DIR="$DRAFT_DIR/figures/Used"
else
  FIG_DIR="$DRAFT_DIR/figures"
fi
BUILD_DIR="$FIG_DIR/.build_mccm_two_conditions"
STEM="mccm_two_conditions"
WRAPPER="$FIG_DIR/mccm_two_conditions_standalone.tex"
OUT_PDF="$FIG_DIR/${STEM}.pdf"

if [[ ! -f "$WRAPPER" ]]; then
  echo "Missing wrapper: $WRAPPER" >&2
  exit 1
fi

mkdir -p "$BUILD_DIR"
latexmk -pdf -interaction=nonstopmode -cd \
  -output-directory="$BUILD_DIR" \
  -jobname="$STEM" \
  "$WRAPPER"

if [[ ! -f "$BUILD_DIR/${STEM}.pdf" ]]; then
  echo "Standalone build did not produce $BUILD_DIR/${STEM}.pdf" >&2
  exit 1
fi

cp -f "$BUILD_DIR/${STEM}.pdf" "$OUT_PDF"

PDFCROP="${PDFCROP:-pdfcrop}"
if command -v "$PDFCROP" >/dev/null 2>&1; then
  tmp="$(mktemp -t mccm_two_conditions_crop.XXXXXX.pdf)"
  "$PDFCROP" --hires --margins "3 10 3 3" "$OUT_PDF" "$tmp"
  mv -f "$tmp" "$OUT_PDF"
else
  echo "Warning: pdfcrop not found; figure may include extra margins." >&2
fi

latexmk -c -cd -output-directory="$BUILD_DIR" -jobname="$STEM" "$WRAPPER" >/dev/null 2>&1 || true

echo "Wrote: $OUT_PDF"
