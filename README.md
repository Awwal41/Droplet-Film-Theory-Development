# DFM reproduction

Code and numerical results for the Droplet-Film Model. The true critical rate is never observed. Models are trained on well status; \(\hat Q_{\mathrm{crit}}\) is then evaluated from the fitted expression.

## Layout

| Path | What it is |
|---|---|
| `dfm_model.py` | DFM / LDM / LFM (physics + residual), fit by logistic loss on \(\ln(Q/Q_{\mathrm{crit}})\) |
| `data_models.py` | QLattice and TabPFN classifiers (P(unloaded) at the measured rate) |
| `evaluate.py` | Scores from cached predictions, or `--refit` all five models |
| `data/project_dataset.csv` | Wells used in the paper |
| `results/` | Saved out-of-fold and test predictions |

Near-loaded wells are withheld from training and counted as Loaded in every accuracy.

## Evaluate the saved results

```bash
pip install -r requirements.txt
python evaluate.py
```

Cached scoring uses only numpy, pandas, scipy, and scikit-learn. `feyn` and `tabpfn` are needed only to refit QLattice and TabPFN.

Expected test-set table:

| Model | CV | Test (44) | Unloaded (23) | Loaded (21) |
|---|---|---|---|---|
| DFM | 85.8 ± 3.5 | 90.9 | 91.3 | 90.5 |
| QLattice | 82.4 ± 5.2 | 88.6 | 87.0 | 90.5 |
| TabPFN | 91.8 ± 4.7 | 84.1 | 82.6 | 85.7 |
| LDM | 85.1 ± 3.6 | 88.6 | 91.3 | 85.7 |
| LFM | 81.1 ± 4.6 | 86.4 | 91.3 | 81.0 |

## Refit

```bash
python evaluate.py --refit
```

This re-estimates DFM, LDM, LFM, QLattice, and TabPFN on the same 148/44 split (Near wells in the test set only). QLattice is the slowest step. The QLattice search is constrained to use \(\log Q\) (`query_string="'logQ' + _"`, 20 epochs, max complexity 7). TabPFN is fit on \(\log Q\) plus the ten well features.

## How Qcrit is obtained (DFM)

1. Physics expression (droplet + film, mixed by a global \(\alpha\)) gives \(Q_{\mathrm{phys}}(X)\).
2. Residual: \(\hat Q_{\mathrm{crit}}(X) = Q_{\mathrm{phys}}(X)\,\exp(b + \mathbf{w}^\top\tilde{\mathbf{x}})\).
3. Parameters are chosen so unloaded wells have \(\ln(Q_{\mathrm{operating}}/\hat Q_{\mathrm{crit}}) > 0\) and loaded wells the reverse.
4. After fitting, \(\hat Q_{\mathrm{crit}}(X)\) can be evaluated for any well. Classify unloaded if \(Q_{\mathrm{operating}} > \hat Q_{\mathrm{crit}}\).

QLattice and TabPFN do not return a critical rate for every well. They are classified by \(P(\mathrm{unloaded}) > 0.5\) at the measured operating rate.
