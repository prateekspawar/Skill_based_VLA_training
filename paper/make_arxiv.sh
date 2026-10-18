#!/usr/bin/env bash
# Build the arXiv upload package: paper/arxiv/skill_level_experts_arxiv.tar.gz
# Contents: main.tex, main.bbl (arXiv does not run BibTeX), the figure PDFs the paper includes. The package is
# test-compiled in a clean directory with pdflatex only, the way arXiv processes it.
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex > /dev/null   # refresh main.bbl
STAGE=$(mktemp -d)
mkdir -p "$STAGE/figures" arxiv
cp main.tex main.bbl "$STAGE/"
for f in $(grep -oE 'figures/[A-Za-z0-9_]+\.pdf' main.tex | sort -u); do cp "$f" "$STAGE/$f"; done
tar -czf arxiv/skill_level_experts_arxiv.tar.gz -C "$STAGE" .
TEST=$(mktemp -d)
tar -xzf arxiv/skill_level_experts_arxiv.tar.gz -C "$TEST"
(cd "$TEST" && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null \
              && pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null)
grep -E "Output written|undefined|Overfull|^! " "$TEST/main.log" || true
cp "$TEST/main.pdf" arxiv/test_compile.pdf
echo "package: $(tar -tzf arxiv/skill_level_experts_arxiv.tar.gz | grep -v '/$' | tr '\n' ' ')"
rm -rf "$STAGE" "$TEST"
