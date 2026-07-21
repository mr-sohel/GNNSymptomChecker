# How to Compile the IEEE LaTeX Paper

## File Structure
```
GNNSymptomChecker/
├── IEEE_Paper_GNN_Symptom_Checker.tex   ← Main LaTeX source
├── figures/
│   ├── fig_system_architecture.jpg
│   ├── fig_bipartite_graph.jpg
│   ├── fig_confusion_matrix.jpg
│   └── fig_roc_curve.jpg
└── HOW_TO_COMPILE.md
```

## Option 1: Overleaf (Recommended — no local install needed)
1. Go to https://www.overleaf.com and create a free account
2. Click "New Project" → "Upload Project"
3. Upload a ZIP containing:
   - `IEEE_Paper_GNN_Symptom_Checker.tex`
   - The entire `figures/` folder
4. Overleaf auto-detects IEEEtran and compiles with pdflatex
5. Click "Recompile" — your PDF will appear in seconds

## Option 2: Local Compile (macOS with MacTeX)
```bash
# Install MacTeX if not present
brew install --cask mactex

# Compile (2 passes for cross-references)
cd "/Users/akash/Development/Knowledge Engineering/GNNSymptomChecker"
pdflatex IEEE_Paper_GNN_Symptom_Checker.tex
pdflatex IEEE_Paper_GNN_Symptom_Checker.tex
```

## Option 3: VS Code + LaTeX Workshop
1. Install VS Code extension "LaTeX Workshop"
2. Install MacTeX (see above)
3. Open the .tex file → Ctrl+Alt+B to build

## Required LaTeX Packages (all included in MacTeX/Overleaf)
- IEEEtran (document class)
- amsmath, amssymb — mathematics
- graphicx — figures
- booktabs, multirow, makecell — tables
- pgfplots, tikz — training loss curve
- listings — code blocks
- hyperref — cross-references
- balance, xcolor — formatting
