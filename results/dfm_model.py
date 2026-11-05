import numpy as np, pandas as pd, warnings, time
from scipy.optimize import differential_evolution, minimize
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score
warnings.filterwarnings("ignore")

CSV = r"C:\Users\awwalola\Downloads\project_dataset.csv"; SEED = 42
FEATURES = ['Dia','Dev(deg)','Area (m2)','z','GasDens','LiquidDens','g (m/s2)','P/T','friction_factor','critical_film_thickness']
LABEL = {'Loaded': -1, 'Near L.U': 0, 'Unloaded': 1}
NAMES = {v: k for k, v in LABEL.items()}

df = pd.read_csv(CSV)
df = df[df["Test status"] != "Questionable"].copy(); df["y"] = df["Test status"].map(LABEL)
df = df.dropna(subset=FEATURES + ["Gasflowrate", "y"]).reset_index(drop=True)
X = df[FEATURES].values.astype(float); Q = df["Gasflowrate"].values.astype(float); y = df.y.values.astype(int)
# Near L.U wells are never trained on: models learn Loaded vs Unloaded, and all Near L.U go to the test set
idx_lu = np.where(y != 0)[0]
idx_tr, idx_te = train_test_split(idx_lu, test_size=.2, stratify=y[idx_lu], random_state=SEED)
idx_te = np.r_[idx_te, np.where(y == 0)[0]]

def classify(m, band): return np.where(m > band, 1, np.where(m < -band, -1, 0))

# ======================= DFM: physics + residual =======================
def phys_terms(X, p):
    Dia, t, Ar, z, G, L, g, PT, ff, film = X.T
    lp1, p2, p3, lp4, p5 = p[:5]
    r = np.clip(Dia/film, 1e-8, 1e6)
    t1 = 2*g*Dia*(1-3*r+4*r*r)*(L-G)*np.cos(np.radians(t))/(ff*G) * np.exp(lp4)   # film
    t2 = np.abs(np.sin(np.radians(p5*t)))**p3 * np.abs(L-G)**p2 / G**2           # droplet
    return np.exp(lp1), t1, t2, PT/np.abs(z)

def qcr_phys(X, p, alpha=None):
    c, t1, t2, s = phys_terms(X, p)
    A = 1/(1+np.exp(-p[5])) if alpha is None else alpha
    return c*np.sqrt(np.maximum(np.abs(A*t1+(1-A)*t2)*s, 1e-12))

def film_share(X, p):
    """Fraction of the combined term contributed by the film term (0 = droplet-only, 1 = film-only)."""
    c, t1, t2, s = phys_terms(X, p); A = 1/(1+np.exp(-p[5]))
    return np.abs(A*t1)/(np.abs(A*t1) + np.abs((1-A)*t2) + 1e-300)

def logistic_loss(m, y):
    t = np.where(y == 1, 1.0, np.where(y == -1, 0.0, 0.5))
    return np.mean(np.logaddexp(0, m) - t*m)

def resid_feats(X):
    dev = np.radians(X[:, 1])
    return np.column_stack([np.log(np.abs(X[:, [0, 3, 4, 5, 7, 8, 9]]) + 1e-12), np.cos(dev), np.sin(dev)])

PB = [(-20, 20), (0, 3), (0, 3), (-10, 10), (0.1, 3), (-6, 6)]
DFM_SLOPE = None   # None: free logistic slope (best accuracy, wide Qcr spread); 4: Qcr stays close to Q (~79 % CV)
class DFM:
    """Physics Qcr + learned log-space residual. alpha=None: global alpha (DFM); 0: droplet only (LDM); 1: film only (LFM)."""
    def __init__(self, lam=0.1, slope=None, alpha=None):
        self.lam, self.slope, self.alpha = lam, slope if slope is not None else DFM_SLOPE, alpha
    def fit(self, X, Q, y):
        self.p = PhysicsOnly(self.alpha).fit(X, Q, y).p
        F = resid_feats(X); self.mu, self.sd = F.mean(0), F.std(0) + 1e-12; F = (F - self.mu)/self.sd
        m0 = np.log(Q/qcr_phys(X, self.p, self.alpha))
        k = (lambda t: self.slope) if self.slope else np.exp
        def obj(th):
            return logistic_loss(k(th[0])*(m0 - th[1] - F @ th[2:]), y) + self.lam*np.sum(th[2:]**2)/len(y)
        th = minimize(obj, np.r_[np.log(4), 0, np.zeros(F.shape[1])], method="L-BFGS-B").x
        self.b, self.w, self.k = th[1], th[2:], k(th[0])
        return self
    def qcr(self, X, Q=None):
        return qcr_phys(X, self.p, self.alpha)*np.exp(self.b + (resid_feats(X) - self.mu)/self.sd @ self.w)
    def p_unloaded(self, X, Q):
        return 1/(1 + np.exp(-self.k*np.log(Q/self.qcr(X))))

class PhysicsOnly:
    """Physics Qcr without the residual: alpha=0 -> droplet only (LDM), 1 -> film only (LFM), None -> global alpha (DFM)."""
    def __init__(self, alpha=None): self.alpha = alpha
    def fit(self, X, Q, y):
        B = PB if self.alpha is None else PB[:5]
        full = (lambda p: p) if self.alpha is None else (lambda p: np.r_[p, 0.0])
        self.p = full(differential_evolution(lambda p: logistic_loss(4*np.log(Q/qcr_phys(X, full(p), self.alpha)), y), B,
                                             maxiter=200, popsize=15, seed=SEED, tol=1e-8).x)
        return self
    def qcr(self, X, Q=None): return qcr_phys(X, self.p, self.alpha)

# ============ data-driven models: implied Qcr = rate where P(Unloaded) crosses 0.5 ============
def implied_qcr(p_unloaded, X, Q, n_grid=41):
    """p_unloaded(X, Q) -> prob. Returns the 0.5-crossing in Q closest to the measured Q (log-interpolated)."""
    grid = np.linspace(np.log(Q.min()/4), np.log(Q.max()*4), n_grid)
    XX, QQ = np.repeat(X, n_grid, 0), np.exp(np.tile(grid, len(X)))
    P = p_unloaded(XX, QQ).reshape(len(X), n_grid) - 0.5
    out = np.empty(len(X))
    for i in range(len(X)):
        s = np.where(np.sign(P[i, :-1]) != np.sign(P[i, 1:]))[0]
        if len(s) == 0:
            out[i] = np.exp(grid[0] if P[i, 0] > 0 else grid[-1]); continue
        a, b = P[i, s], P[i, s+1]
        xc = grid[s] + (grid[s+1]-grid[s])*a/(a-b)
        out[i] = np.exp(xc[np.argmin(np.abs(xc - np.log(Q[i])))])
    return out

class TabPFNModel:
    def fit(self, X, Q, y):
        from tabpfn import TabPFNClassifier
        self.m = TabPFNClassifier(device="cpu", random_state=SEED).fit(np.column_stack([np.log(Q), X]), y)
        self.ci = list(self.m.classes_).index(1)
        return self
    def p_unloaded(self, X, Q): return self.m.predict_proba(np.column_stack([np.log(Q), X]))[:, self.ci]
    def qcr(self, X, Q): return implied_qcr(self.p_unloaded, X, Q)

QL_COLS = ['Dia', 'Dev', 'z', 'GasDens', 'LiquidDens', 'PT', 'ff', 'film']
def ql_frame(X, Q):
    d = pd.DataFrame(X[:, [0, 1, 3, 4, 5, 7, 8, 9]], columns=QL_COLS); d.insert(0, "logQ", np.log(Q)); return d

class QLatticeModel:
    def __init__(self, n_epochs=20, max_complexity=7): self.n_epochs, self.max_complexity = n_epochs, max_complexity
    def fit(self, X, Q, y):
        import feyn
        d = ql_frame(X, Q); d["unloaded"] = (y == 1).astype(int)
        ql = feyn.QLattice(random_seed=SEED)
        self.m = ql.auto_run(d, output_name="unloaded", kind="classification", n_epochs=self.n_epochs,
                             max_complexity=self.max_complexity, query_string="'logQ' + _")[0]
        return self
    def p_unloaded(self, X, Q): return self.m.predict(ql_frame(X, Q))
    def qcr(self, X, Q): return implied_qcr(self.p_unloaded, X, Q)
