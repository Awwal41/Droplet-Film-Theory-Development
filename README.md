# DFM reproduction

Code and numerical results for the Droplet-Film Model. The true critical rate is never observed. Models are trained on well status; \(\hat Q_{\mathrm{crit}}\) is then evaluated from the fitted expression.

## Layout

| Path | What it is |
|---|---|
| `dfm_model.py` | DFM / LDM / LFM (physics + residual), fit by logistic loss on \(\ln(Q/Q_{\mathrm{crit}})\) |
| `evaluate.py` | Scores from cached predictions, or `--refit` the DFM family |
| `data/project_dataset.csv` | Wells used in the paper |
| `results/` | Saved out-of-fold and test predictions |

Near-loaded wells are withheld from training and counted as Loaded in every accuracy.

## Evaluate the saved results

```bash
pip install -r requirements.txt
python evaluate.py
```

Expected test-set table:

| Model | CV | Test (44) | Unloaded (23) | Loaded (21) |
|---|---|---|---|---|
| DFM | 85.8 ± 3.5 | 90.9 | 91.3 | 90.5 |
| QLattice | 82.4 ± 5.2 | 88.6 | 87.0 | 90.5 |
| TabPFN | 91.8 ± 4.7 | 84.1 | 82.6 | 85.7 |
| LDM | 85.1 ± 3.6 | 88.6 | 91.3 | 85.7 |
| LFM | 81.1 ± 4.6 | 86.4 | 91.3 | 81.0 |

## Refit DFM, LDM, LFM

```bash
python evaluate.py --refit
```

This re-estimates the physics parameters (differential evolution) and the residual. QLattice and TabPFN are not refit here; their probabilities are stored in `results/qlattice_tabpfn_probabilities.csv`.

## How Qcrit is obtained

1. Physics expression (droplet + film, mixed by a global \(\alpha\)) gives \(Q_{\mathrm{phys}}(X)\).
2. Residual: \(\hat Q_{\mathrm{crit}}(X) = Q_{\mathrm{phys}}(X)\,\exp(b + \mathbf{w}^\top\tilde{\mathbf{x}})\).
3. Parameters are chosen so unloaded wells have \(\ln(Q_{\mathrm{operating}}/\hat Q_{\mathrm{crit}}) > 0\) and loaded wells the reverse.
4. After fitting, \(\hat Q_{\mathrm{crit}}(X)\) can be evaluated for any well. Classify unloaded if \(Q_{\mathrm{operating}} > \hat Q_{\mathrm{crit}}\).
