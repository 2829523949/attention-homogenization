#!/bin/bash
cd "$(dirname "$0")"
pdflatex main
bibtex main
pdflatex main
pdflatex main
echo "Compiled: main.pdf"
