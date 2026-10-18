# Paper draft: Skill-Level Experts with Grounded Targets

`main.tex` + `references.bib` (plain LaTeX, two-column article; to be moved to the target venue's template).
Experiment 3 and the author list are marked `\todo{...}` (red in the PDF); bibliography entries with
`note = {VERIFY}` must be checked against the published versions.

Build (IEEEtran; needs texlive-latex-extra and texlive-publishers, or Overleaf). The compiled `main.pdf` is committed:

    latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main

Numbers come from the evaluation logs under `outputs/` (paired sign tests computed per comparison); see
docs/NOTES.md section 7 for the runs; per-episode results and sign tests are in `data/`.
