"""QLattice and TabPFN classifiers. Fit from well status; score by P(Unloaded) at the measured rate."""
import numpy as np
import pandas as pd

from dfm_model import SEED

QL_COLS = ["Dia", "Dev", "z", "GasDens", "LiquidDens", "PT", "ff", "film"]


def ql_frame(X, Q):
    d = pd.DataFrame(X[:, [0, 1, 3, 4, 5, 7, 8, 9]], columns=QL_COLS)
    d.insert(0, "logQ", np.log(Q))
    return d


class TabPFNModel:
    """TabPFN on log(Q) + the ten well features. Unloaded if P > 0.5."""

    def fit(self, X, Q, y):
        from tabpfn import TabPFNClassifier

        self.m = TabPFNClassifier(device="cpu", random_state=SEED).fit(
            np.column_stack([np.log(Q), X]), y
        )
        self.ci = list(self.m.classes_).index(1)
        return self

    def p_unloaded(self, X, Q):
        return self.m.predict_proba(np.column_stack([np.log(Q), X]))[:, self.ci]


class QLatticeModel:
    """QLattice symbolic classifier. The search is constrained to use log(Q)."""

    def __init__(self, n_epochs=20, max_complexity=7):
        self.n_epochs, self.max_complexity = n_epochs, max_complexity

    def fit(self, X, Q, y):
        import feyn

        d = ql_frame(X, Q)
        d["unloaded"] = (y == 1).astype(int)
        ql = feyn.QLattice(random_seed=SEED)
        self.m = ql.auto_run(
            d,
            output_name="unloaded",
            kind="classification",
            n_epochs=self.n_epochs,
            max_complexity=self.max_complexity,
            query_string="'logQ' + _",
        )[0]
        return self

    def p_unloaded(self, X, Q):
        return self.m.predict(ql_frame(X, Q))

    def formula(self):
        return str(self.m.sympify(signif=3))
