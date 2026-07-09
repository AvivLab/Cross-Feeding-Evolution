#!/usr/bin/env bash
# Render each chemostat snapshot variant -> figures/Used/chemostat_snap/<variant>.pdf
# (standalone TikZ + pdfcrop; used by Supp. Fig. S3 so diagrams match S2 without nested-TikZ artifacts).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRAFT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
if [[ -d "$DRAFT_DIR/figures/Used" ]]; then
  FIG_DIR="$DRAFT_DIR/figures/Used"
else
  FIG_DIR="$DRAFT_DIR/figures"
fi
OUT_DIR="$FIG_DIR/chemostat_snap"
BUILD_DIR="$FIG_DIR/.build_chemostat_snap"
BODY_TEX="$BUILD_DIR/chemostat_snap_export_body.tex"
PDFCROP="${PDFCROP:-pdfcrop}"

VARIANTS=(
  init inflow taskA taskB death dup dupmut mut outflow endgen score collapse
)

if [[ ! -f "$FIG_DIR/chemostat_snap_standalone.tex" ]]; then
  echo "Missing wrapper: $FIG_DIR/chemostat_snap_standalone.tex" >&2
  exit 1
fi

mkdir -p "$OUT_DIR" "$BUILD_DIR"

build_one() {
  local variant="$1"
  local kind="$2"
  local jobname="chemostat_snap_${variant}"
  local driver="$BUILD_DIR/_${jobname}.tex"
  local out_pdf="$OUT_DIR/${variant}.pdf"

  if [[ "$kind" == "energy" ]]; then
    printf '%s\n' '\ChemostatSnapEnergyPanel' >"$BODY_TEX"
  elif [[ "$kind" == "energy_noeq" ]]; then
    printf '%s\n' '\ChemostatSnapEnergyPanelNoEq' >"$BODY_TEX"
  else
    printf '%s\n' "\\ChemostatSnapshot{}{${variant}}" >"$BODY_TEX"
  fi

  cat >"$driver" <<'EOF'
\makeatletter
\def\input@path{{../}}
\makeatother
\input{chemostat_snap_standalone.tex}
EOF

  latexmk -pdf -interaction=nonstopmode -cd \
    -output-directory="$BUILD_DIR" \
    -jobname="$jobname" \
    "$driver"

  rm -f "$driver"

  if [[ ! -f "$BUILD_DIR/${jobname}.pdf" ]]; then
    echo "Standalone build did not produce $BUILD_DIR/${jobname}.pdf" >&2
    exit 1
  fi

  cp -f "$BUILD_DIR/${jobname}.pdf" "$out_pdf"

  if command -v "$PDFCROP" >/dev/null 2>&1; then
    local tmp
    tmp="$(mktemp -t chemostat_snap_crop.XXXXXX.pdf)"
    "$PDFCROP" --hires --margins "1 1 1 1" "$out_pdf" "$tmp"
    mv -f "$tmp" "$out_pdf"
  else
    echo "Warning: pdfcrop not found; ${variant}.pdf may include extra margins." >&2
  fi

  echo "Wrote: $out_pdf"
}

for variant in "${VARIANTS[@]}"; do
  build_one "$variant" snapshot
done
build_one energy energy
build_one energy_noeq energy_noeq

rm -f "$BODY_TEX"
latexmk -c -cd -output-directory="$BUILD_DIR" >/dev/null 2>&1 || true
