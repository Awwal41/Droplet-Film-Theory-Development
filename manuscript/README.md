# Updated manuscript (November 2026)

This folder contains the current DFM manuscript and SI only.

- `article.pdf`, `SI.pdf` — compiled papers
- `article.tex`, `SI.tex`, `sn-bibliography.bib` — source
- `Figures/` — figures used in the paper

The DFM is trained on observed well status (loaded/unloaded). The true critical rate is never used as a regression target. After fitting, \(\hat Q_{\mathrm{crit}}(X)\) is evaluated from the closed-form physics expression plus a log-space residual.

Numerical predictions and the DFM implementation are in `../results/`.
