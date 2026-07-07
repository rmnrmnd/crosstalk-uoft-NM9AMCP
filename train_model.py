"""Main training and submission pipeline for the CrossTALK workshop.

Example:
    $ python train_model.py
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import average_precision_score

import src.dataset
import src.eval
import src.path_and_constants

_P = src.path_and_constants.Paths()
_C = src.path_and_constants.Constants()

# --------------------------------------------------------------------------- configuration
QUICK_TEST = 1       # small/fast smoke test
SEED = 42
EPOCHS = 3 if QUICK_TEST else 18
BATCH = 1024
K = 200
SEEDS = [42] if QUICK_TEST else [42, 7, 2025]               # seed-ensemble each base for stability
LR, WEIGHT_DECAY, DROPOUT = 1e-3, 1e-5, 0.3
VAL_FOLD = 4                                                # BB1-disjoint holdout fold
TABPFN_CTX = 10000
DEV = "cuda" if torch.cuda.is_available() else "cpu"

if DEV != "cuda":
    print("NO GPU! NO TRAINING!")
    exit()

# 9 single fingerprints + 2 mixed pairs
MODEL_INPUTS = [[fp] for fp in src.dataset.FINGERPRINTS] + [["ECFP4", "TOPTOR"], ["FCFP4", "AVALON"]]
MODEL_NAMES = ["+".join(s) for s in MODEL_INPUTS]

torch.manual_seed(SEED)
np.random.seed(SEED)


def download_data() -> None:
    """Downloads dataset files from Google Drive if they don't exist locally."""
    import gdown
    os.makedirs("data", exist_ok=True)
    for filepath, file_id in _C.file_ids.items():
        if not os.path.exists(filepath):
            print(f"Downloading {filepath}...")
            gdown.download(id=file_id, output=filepath, quiet=False)


def load_spec(path, spec, max_rows=None):
    """Load fingerprint(s) via the template loader, log1p-scale the counts, return CSR float32."""
    X = src.dataset.load_data(path, spec, y_col=None, max_rows=max_rows).tocsr().astype(np.float32)
    X.data = np.log1p(X.data)
    return X


def parse_bb1(path, max_rows=None):
    """BB1 synthon = field 2 of DEL_ID 'L<lib>-<bb1>-<bb2>-<bb3>' (leakage-aware grouping)."""
    s = pd.read_parquet(path, columns=["DEL_ID"])["DEL_ID"]
    if max_rows is not None:
        s = s.iloc[:max_rows]
    return s.str.split("-", expand=True)[1].to_numpy()


# model + engine
class WideDeep(nn.Module):
    """Linear 'wide' path + three parallel 'deep' MLP branches of differing depth."""

    def __init__(self, d, p=DROPOUT):
        super().__init__()
        self.wide = nn.Linear(d, 1)
        self.deeps = nn.ModuleList()
        for hid in [(1024, 512, 256), (1024, 512), (1024,)]:
            layers, prev = [], d
            for h in hid:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
                prev = h
            layers += [nn.Linear(prev, 1)]
            self.deeps.append(nn.Sequential(*layers))

    def forward(self, x):
        return (self.wide(x) + sum(d(x) for d in self.deeps)).squeeze(1)


def _dense(M, batch_idx):
    return torch.from_numpy(np.ascontiguousarray(M[batch_idx].toarray(), np.float32)).to(DEV)


@torch.no_grad()
def predict(model, M, idx, bs=8192):
    model.eval()
    out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), bs):
        b = idx[i:i + bs]
        out[i:i + len(b)] = torch.sigmoid(model(_dense(M, b))).cpu().numpy()
    return out


def rank01(s):
    return np.argsort(np.argsort(s)) / (len(s) - 1)


def train_one(Xfp, y, train_idx, val_idx, epochs=EPOCHS, seed=SEED):
    """Train one WideDeep; if val_idx given, keep the best-AUPRC epoch."""
    torch.manual_seed(seed)
    model = WideDeep(Xfp.shape[1]).to(DEV)
    pos = float(y[train_idx].sum())
    neg = len(train_idx) - pos
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1)], device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    best = {"ap": -1.0, "pred": None, "ep": epochs, "state": None}
    for ep in range(epochs):
        model.train()
        order = np.random.default_rng(seed + ep).permutation(train_idx)
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(model(_dense(Xfp, b)), torch.from_numpy(y[b]).to(DEV)).backward()
            opt.step()
        sch.step()
        if val_idx is not None:
            vp = predict(model, Xfp, val_idx)
            ap = average_precision_score(y[val_idx], vp)
            if ap > best["ap"]:
                best = {"ap": ap, "pred": vp, "ep": ep + 1,
                        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
    if val_idx is not None:
        model.load_state_dict(best["state"])
    return model, best


# ensembler
def tabpfn_stack(P_ctx, y_ctx, P_query, ctx=TABPFN_CTX, chunk=4000):
    """TabPFN-v3 in-context stacker over the raw base predictions. Returns None if unavailable."""
    try:
        os.environ.setdefault("TABPFN_DISABLE_TELEMETRY", "1")
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
    except Exception as e:
        print(f" TabPFN unavailable: {e}")
        return None
    rng = np.random.default_rng(SEED)
    if ctx < len(y_ctx):
        pos, neg = np.where(y_ctx == 1)[0], np.where(y_ctx == 0)[0]
        npos = max(1, round(ctx * len(pos) / len(y_ctx)))
        sel = np.concatenate([rng.choice(pos, min(npos, len(pos)), replace=False),
                              rng.choice(neg, min(ctx - npos, len(neg)), replace=False)])
    else:
        sel = np.arange(len(y_ctx))
    clf = TabPFNClassifier.create_default_for_version(
        ModelVersion.V3, device=DEV, ignore_pretraining_limits=True)
    clf.fit(P_ctx[sel], y_ctx[sel].astype(int))
    out = np.empty(len(P_query), np.float32)
    for i in range(0, len(P_query), chunk):
        out[i:i + chunk] = clf.predict_proba(P_query[i:i + chunk])[:, 1]
    del clf
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return out


def blend(P, tabpfn_ctx=None):
    """Rank-average of rank-average + median (+ TabPFN when available) — the score-8 blend."""
    ranks = np.array([rank01(P[:, j]) for j in range(P.shape[1])])
    stack_ranks = [rank01(ranks.mean(0)), rank01(np.median(ranks, 0))]
    if tabpfn_ctx is not None:
        s_tab = tabpfn_stack(tabpfn_ctx[0], tabpfn_ctx[1], P)
        if s_tab is not None:
            stack_ranks.append(rank01(s_tab))
    return np.mean(stack_ranks, axis=0)


download_data()
t0 = time.time()
n_tr = 15000 if QUICK_TEST else None
n_te = 3000 if QUICK_TEST else None

y = pd.read_parquet(_P.train_path, columns=["DELLabel"])["DELLabel"].to_numpy().astype(np.float32)
y = y[:n_tr] if n_tr else y
test_ids = pq.ParquetFile(_P.test_path).read(columns=["RandomID"]).to_pandas()["RandomID"].to_numpy()
test_ids = test_ids[:n_te] if n_te else test_ids
N, NT = len(y), len(test_ids)
all_idx = np.arange(N)

# leakage-aware BB1-disjoint fold split (fold VAL_FOLD held out)
bb1 = parse_bb1(_P.train_path, n_tr)
fold = np.full(N, -1, dtype=np.int64)
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
for i, (_, vi) in enumerate(sgkf.split(np.zeros(N, np.int8), y, bb1)):
    fold[vi] = i
tr_idx, val_idx = all_idx[fold != VAL_FOLD], all_idx[fold == VAL_FOLD]
y_val = y[val_idx]
print(f"[split] BB1-disjoint fold {VAL_FOLD}: train={len(tr_idx):,} val={len(val_idx):,} "
        f"test={NT:,} pos-rate={y.mean():.4%} device={DEV}", flush=True)

# 11 base models, each seed-ensembled: fold-VAL preds (P_val) + all-data test preds (P_test)
P_val = np.zeros((len(val_idx), len(MODEL_INPUTS)), np.float32)
P_test = np.zeros((NT, len(MODEL_INPUTS)), np.float32)
base_states, base_dims = [], []
for j, spec in enumerate(MODEL_INPUTS):
    t = time.time()
    Xtr = load_spec(_P.train_path, spec, n_tr)
    Xte = load_spec(_P.test_path, spec, n_te)
    base_dims.append(Xtr.shape[1])
    vp, tp, aps, seed_states = np.zeros(len(val_idx), np.float32), np.zeros(NT, np.float32), [], []
    for s in SEEDS:
        _, best = train_one(Xtr, y, tr_idx, val_idx, seed=s)
        vp += best["pred"]
        aps.append(best["ap"])
        m_full, _ = train_one(Xtr, y, all_idx, None, epochs=best["ep"], seed=s)
        tp += predict(m_full, Xte, np.arange(NT))
        seed_states.append({k: v.detach().cpu() for k, v in m_full.state_dict().items()})
        del m_full
        if DEV == "cuda":
            torch.cuda.empty_cache()
    P_val[:, j] = vp / len(SEEDS)
    P_test[:, j] = tp / len(SEEDS)
    base_states.append(seed_states)
    print(f"  base {MODEL_NAMES[j]:14s} fold{VAL_FOLD} AUPRC={np.mean(aps):.4f} "
            f"P@{K}={src.eval.precision_at_k(y_val, P_val[:, j], K):.4f}  ({round(time.time()-t)}s)", flush=True)
    del Xtr, Xte
    if DEV == "cuda":
        torch.cuda.empty_cache()

# 3-stacker blend
tabpfn_ctx = (P_val, y_val)
val_blend = blend(P_val, tabpfn_ctx=tabpfn_ctx)
final = blend(P_test, tabpfn_ctx=tabpfn_ctx)

# local validation
n_boot = 100 if QUICK_TEST else 300
p_mean, p_lo, p_hi = src.eval.bootstrap_ci(y_val, val_blend, src.eval.precision_at_k, k=K, n_iterations=n_boot)
print(f"\nLocal Validation Split Metrics (BB1-disjoint holdout, {len(y_val):,} molecules):")
print(f"  Precision@{K:<15d}: {src.eval.precision_at_k(y_val, val_blend, K):.4f} "
        f"(95% CI: [{p_lo:.4f}, {p_hi:.4f}])")
print(f"  Hits@{K:<20d}: {int(src.eval.hits_at_k(y_val, val_blend, K) * (y_val == 1).sum())}")
print(f"  ROC-AUC{'':<18s}: {src.eval.roc_auc(y_val, val_blend):.4f}")

# save final model
os.makedirs(os.path.dirname(_P.model_path), exist_ok=True)
torch.save({
    "format": "crosstalk-widedeep-stack-v3",
    "model_inputs": MODEL_INPUTS, "base_dims": base_dims, "base_states": base_states,
    "tabpfn_ctx_P": P_val, "tabpfn_ctx_y": y_val,
}, _P.model_path)
print(f"\nSaved final model to {_P.model_path}")

# submission
pd.DataFrame({"RandomID": test_ids, "DELLabel": final}).to_csv(
    _P.submission_path, index=False, float_format="%.10f")
print(f"Generated submission file {_P.submission_path}  ({len(final):,} rows, {round(time.time()-t0)}s total)")
print("Pipeline complete.")

