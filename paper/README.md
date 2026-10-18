# Paper draft: Skill-Level Experts with Grounded Targets

`main.tex` + `references.bib` (plain LaTeX, two-column article; to be moved to the target venue's template).
Experiment 3 and the author list are marked `\todo{...}` (red in the PDF); bibliography entries with
`note = {VERIFY}` must be checked against the published versions.

Build (TeX Live is not installed on this machine; Overleaf works with the two files as they are):

    latexmk -pdf main.tex      # or: pdflatex main && bibtex main && pdflatex main && pdflatex main

Numbers come from the evaluation logs under `outputs/` (paired sign tests computed per comparison); see
README.md section 7 for the runs.
