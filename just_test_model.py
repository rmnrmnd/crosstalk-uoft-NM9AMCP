"""Inference-only pipeline: 

load models/best_model.pkl and generate submission.csv.
"""

import os
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

import src.dataset
import src.eval
import src.path_and_constants

_P = src.path_and_constants.Paths()
_C = src.path_and_constants.Constants()

# config
SEED = 42
K = 200
DROPOUT = 0.3
TABPFN_CTX = 10000
DEV = "cuda" if torch.cuda.is_available() else "cpu"

if DEV != "cuda":
    print("NO GPU! NO TRAINING!")
    exit()

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

test_ids = pq.ParquetFile(_P.test_path).read(columns=["RandomID"]).to_pandas()["RandomID"].to_numpy()
NT = len(test_ids)

# Loading (base model weights + TabPFN context from the training run)
ckpt = torch.load(_P.model_path, map_location="cpu", weights_only=False)
assert ckpt["format"] == "crosstalk-widedeep-stack-v3", f"unexpected checkpoint format {ckpt['format']!r}"
MODEL_INPUTS = ckpt["model_inputs"]
MODEL_NAMES = ["+".join(s) for s in MODEL_INPUTS]
P_val, y_val = ckpt["tabpfn_ctx_P"], ckpt["tabpfn_ctx_y"]
print(f"[load] {_P.model_path}: {len(MODEL_INPUTS)} bases x {len(ckpt['base_states'][0])} seeds "
        f"test={NT:,} device={DEV}", flush=True)

# 11 base models predictions
P_test = np.zeros((NT, len(MODEL_INPUTS)), np.float32)
for j, spec in enumerate(MODEL_INPUTS):
    t = time.time()
    Xte = load_spec(_P.test_path, spec)
    assert Xte.shape[1] == ckpt["base_dims"][j], f"dim mismatch for {MODEL_NAMES[j]}"
    tp = np.zeros(NT, np.float32)
    for state in ckpt["base_states"][j]:
        m_full = WideDeep(ckpt["base_dims"][j]).to(DEV)
        m_full.load_state_dict(state)
        tp += predict(m_full, Xte, np.arange(NT))
        del m_full
        if DEV == "cuda":
            torch.cuda.empty_cache()
    P_test[:, j] = tp / len(ckpt["base_states"][j])
    print(f"  base {MODEL_NAMES[j]:14s} done  ({round(time.time()-t)}s)", flush=True)
    del Xte
    torch.cuda.empty_cache()

# 3-stacker blend
tabpfn_ctx = (P_val, y_val)
val_blend = blend(P_val, tabpfn_ctx=tabpfn_ctx)
final = blend(P_test, tabpfn_ctx=tabpfn_ctx)

# local validation (recomputed from the stored fold predictions)
print(f"\nLocal Validation Split Metrics (BB1-disjoint holdout, {len(y_val):,} molecules):")
print(f"  Precision@{K:<15d}: {src.eval.precision_at_k(y_val, val_blend, K):.4f}")
print(f"  Hits@{K:<20d}: {int(src.eval.hits_at_k(y_val, val_blend, K) * (y_val == 1).sum())}")
print(f"  ROC-AUC{'':<18s}: {src.eval.roc_auc(y_val, val_blend):.4f}")

# submission
pd.DataFrame({"RandomID": test_ids, "DELLabel": final}).to_csv(
    _P.submission_path, index=False, float_format="%.10f")
print(f"Generated submission file {_P.submission_path}  ({len(final):,} rows, {round(time.time()-t0)}s total)")
print("Inference complete.")
