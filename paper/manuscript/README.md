# Pruning manuscript

Snapshot of the current manuscript on 7 November 2026. `main.tex` is the entry
point; `main.pdf` is the existing 15-page rendering saved with this snapshot.
The two implementation TODOs and author placeholders are retained as part of
the draft.

Build from the repository root with a TeX Live installation providing
`latexmk`, pdfLaTeX, BibTeX, and the `acmart` class:

```sh
./paper/manuscript/build.sh
```

The directory includes all bibliography, TeX, and figure inputs. Intermediate
LaTeX files are ignored. The snapshot contains the existing PDF without
regenerating it.
