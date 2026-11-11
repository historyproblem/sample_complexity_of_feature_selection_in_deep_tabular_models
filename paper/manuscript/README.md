# Pruning manuscript

`main.tex` is the entry point and `main.pdf` is its compiled rendering.
The original Methodology, including Algorithm 1, is retained in `main.tex`.
`iterative_pruning.tex` adds the iterative procedure as a separate subsection.
Author and conference placeholders remain in the draft.

The added experiment tables and figures are reproducible from the CSV files
in `../data`; see [data provenance](../data/README.md). From the repository
root on the server, regenerate them with:

```sh
.venv/bin/python paper/scripts/build_results.py
```

This requires Matplotlib. On local Windows, use the corresponding Python
environment (`python paper/scripts/build_results.py`). The original notebook
is only needed to re-import its measurements, not to rebuild the figures.

Build from the repository root with a TeX Live installation providing
`latexmk`, pdfLaTeX, BibTeX, and the `acmart` class:

```sh
./paper/manuscript/build.sh
```

The directory includes all bibliography, TeX, and figure inputs. Intermediate
LaTeX files are ignored. Without `latexmk`, run `pdflatex main.tex`,
`bibtex main`, and `pdflatex main.tex` twice from this directory.
