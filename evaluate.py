"""Evaluate DFM / LDM / LFM / QLattice / TabPFN from cached results, or refit them.

Usage
-----
python evaluate.py              # scores from results/*.csv (no fitting)
python evaluate.py --refit      # refit DFM, LDM, LFM, QLattice, and TabPFN
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold

from dfm_model import (
    FEATURES, LABEL, NAMES, NEAR_BAND, SEED, DFM, classify, as_loaded, film_share,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "project_dataset.csv"
RES = ROOT / "results"


def load_wells():
    df = pd.read_csv(DATA)
    df = df[df["Test status"] != "Questionable"].copy()
    df["y"] = df["Test status"].map(LABEL)
    df = df.dropna(subset=FEATURES + ["Gasflowrate", "y"]).reset_index(drop=True)
    X = df[FEATURES].values.astype(float)
    Q = df["Gasflowrate"].values.astype(float)
    y = df.y.values.astype(int)
    idx_lu = np.where(y != 0)[0]
    idx_tr, idx_te = train_test_split(idx_lu, test_size=0.2, stratify=y[idx_lu], random_state=SEED)
    idx_te = np.r_[idx_te, np.where(y == 0)[0]]
    return df, X, Q, y, idx_tr, idx_te


def score(true, pred):
    return float(np.mean(as_loaded(true) == as_loaded(pred)))


def predict_from_qcr(Q, qcr):
    return pd.Series(classify(np.log(Q / qcr), NEAR_BAND)).map(NAMES)


def predict_from_p(p):
    return np.where(np.asarray(p) > 0.5, "Unloaded", "Loaded")


def report(name, true, pred, cv_mean=None, cv_std=None):
    true = pd.Series(np.asarray(true, dtype=object), dtype=object)
    pred = pd.Series(np.asarray(pred, dtype=object), dtype=object)
    u = true == "Unloaded"
    ln = true.isin(["Loaded", "Near L.U"])
    row = {
        "Model": name,
        "CV": None if cv_mean is None else f"{100*cv_mean:.1f} ± {100*cv_std:.1f}",
        "Test": 100 * score(true, pred),
        "Unloaded (23)": 100 * score(true[u], pred[u]),
        "Loaded (21)": 100 * score(true[ln], pred[ln]),
    }
    return row


def evaluate_cache():
    pred = pd.read_csv(RES / "model_predictions.csv")
    probs = pd.read_csv(RES / "qlattice_tabpfn_probabilities.csv")
    pred = pred.merge(probs[["Well no", "Qlattice model", "TabPFN model"]], on="Well no", suffixes=("_qcr", ""))
    te = pred[pred.split == "test"]
    tr = pred[pred.split == "train"]
    rows = []
    for name, col in [("DFM", "Droplet-Film model"), ("QLattice", "Qlattice model"), ("TabPFN", "TabPFN model")]:
        if name == "DFM":
            pt = predict_from_qcr(te["Test flow rate"], te[col])
            fold_acc = [
                score(f["Well status"], predict_from_qcr(f["Test flow rate"], f[col]))
                for _, f in tr.groupby("fold")
            ]
        else:
            pt = predict_from_p(te[col])
            fold_acc = [
                score(f["Well status"], predict_from_p(f[col]))
                for _, f in tr.groupby("fold")
            ]
        rows.append(report(name, te["Well status"].values, pt, np.mean(fold_acc), np.std(fold_acc)))
    base = pd.read_csv(RES / "ldm_lfm_predictions.csv")
    bte, btr = base[base.split == "test"], base[base.split == "train"]
    for name, col in [("LDM", "Liquid droplet model"), ("LFM", "Liquid film model")]:
        pt = predict_from_qcr(bte["Test flow rate"], bte[col])
        fold_acc = [
            score(f["Well status"], predict_from_qcr(f["Test flow rate"], f[col]))
            for _, f in btr.groupby("fold")
        ]
        rows.append(report(name, bte["Well status"], pt, np.mean(fold_acc), np.std(fold_acc)))
    return pd.DataFrame(rows)


def evaluate_refit():
    df, X, Q, y, idx_tr, idx_te = load_wells()
    Xtr, Qtr, ytr = X[idx_tr], Q[idx_tr], y[idx_tr]
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, ytr))
    true_te = pd.Series([NAMES[v] for v in y[idx_te]])
    by_name = {}

    for name, alpha in [("DFM", None), ("LDM", 0.0), ("LFM", 1.0)]:
        print("fitting", name, flush=True)
        acc = []
        for a, b in folds:
            m = DFM(alpha=alpha).fit(Xtr[a], Qtr[a], ytr[a])
            pred = classify(np.log(Qtr[b] / m.qcr(Xtr[b])))
            acc.append(np.mean(as_loaded(pred) == as_loaded(ytr[b])))
        m = DFM(alpha=alpha).fit(Xtr, Qtr, ytr)
        pred = pd.Series(classify(np.log(Q[idx_te] / m.qcr(X[idx_te])))).map(NAMES)
        by_name[name] = report(name, true_te, pred, np.mean(acc), np.std(acc))
        if name == "DFM":
            phi = film_share(X, m.p)
            print(f"fitted alpha={1/(1+np.exp(-m.p[5])):.4f}  k={m.k:.3f}  "
                  f"film-share median={np.median(phi):.3f}")

    try:
        from data_models import QLatticeModel, TabPFNModel
    except ImportError as err:
        print("Install feyn and tabpfn to refit QLattice/TabPFN:", err)
    else:
        for name, Model in [("QLattice", QLatticeModel), ("TabPFN", TabPFNModel)]:
            print("fitting", name, flush=True)
            acc = []
            for a, b in folds:
                m = Model().fit(Xtr[a], Qtr[a], ytr[a])
                pred = predict_from_p(m.p_unloaded(Xtr[b], Qtr[b]))
                true = pd.Series([NAMES[v] for v in ytr[b]])
                acc.append(score(true, pred))
            m = Model().fit(Xtr, Qtr, ytr)
            pred = predict_from_p(m.p_unloaded(X[idx_te], Q[idx_te]))
            by_name[name] = report(name, true_te, pred, np.mean(acc), np.std(acc))
            if name == "QLattice":
                print("QLattice formula:", m.formula())

    order = ["DFM", "QLattice", "TabPFN", "LDM", "LFM"]
    return pd.DataFrame([by_name[k] for k in order if k in by_name])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--refit", action="store_true",
                   help="refit DFM, LDM, LFM, QLattice, and TabPFN (several minutes)")
    args = p.parse_args()
    table = evaluate_refit() if args.refit else evaluate_cache()
    print(table.to_string(index=False, formatters={
        "Test": "{:.1f}".format, "Unloaded (23)": "{:.1f}".format, "Loaded (21)": "{:.1f}".format,
    }))


if __name__ == "__main__":
    main()
