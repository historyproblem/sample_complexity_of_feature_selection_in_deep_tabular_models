#!/bin/sh
set -eu

cd "$(dirname "$0")"

if ! command -v latexmk >/dev/null 2>&1; then
    tex_bin="$HOME/Library/TinyTeX/bin/universal-darwin"
    if [ -x "$tex_bin/latexmk" ]; then
        PATH="$tex_bin:$PATH"
        export PATH
    else
        echo "latexmk is required (TeX Live, MacTeX or TinyTeX)." >&2
        exit 1
    fi
fi

exec latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error main.tex
