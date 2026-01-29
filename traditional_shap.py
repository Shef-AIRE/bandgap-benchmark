#!/usr/bin/env python3
"""
10-fold SHAP bar plots for traditional ML models (ALL features).

Feature blocks:
  0–92     : self-atom encoding
  92–184   : neighbour-atom encoding
  184–220  : distance-to-neighbour encoding

Each bar shows mean(|SHAP|) across folds,
with error bars = std across folds.
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import KFold

# Make repo imports work regardless of working directory
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import get_cfg_defaults
from loaddata.cifdata import CIFData
from loaddata.dataloader import extract_features
from traditional_ml import _build_estimator

FEATURE_BLOCKS = [
    ("Group #", 19),
    ("Period #", 7),
    ("Electronegativity", 10),
    ("Covalent radius", 10),
    ("Valence electrons", 12),
    ("First ionization energy", 10),
    ("Electron affinity", 10),
    ("Block", 4),
    ("Atomic volume", 10),
]

SEGMENTS = [
    ("self", slice(0, 92)),
    ("nbr", slice(92, 184)),
    ("dist", slice(184, 220)),
]


def load_xy(cfg):
    data = json.loads(Path(cfg.DATASET.TRAIN).read_text())
    rows = [{"mpids": k, "bg": v["bg"]} for k, v in data.items()]
    df = pd.DataFrame(rows)

    ds = CIFData(
        df[["mpids", "bg"]],
        cfg.MODEL.CIF_FOLDER,
        cfg.MODEL.INIT_FILE,
        cfg.MODEL.MAX_NBRS,
        cfg.MODEL.RADIUS,
        cfg.SOLVER.RANDOMIZE,
    )
    X, y = extract_features(ds)
    return np.asarray(X), np.asarray(y)


def shap_mean_abs_per_fold(model_name, model, X_tr, X_val, seed, bg_k, eval_n, nsamples):
    rng = np.random.default_rng(seed)

    if model_name == "random_forest":
        sv = shap.TreeExplainer(model).shap_values(X_val)
        return np.mean(np.abs(np.asarray(sv)), axis=0)

    if model_name == "linear_regression":
        sv = shap.LinearExplainer(model, X_tr, feature_perturbation="interventional").shap_values(X_val)
        return np.mean(np.abs(np.asarray(sv)), axis=0)

    if model_name == "svm":
        k = min(bg_k, X_tr.shape[0])
        background = shap.kmeans(X_tr, k=k)
        idx = rng.choice(X_val.shape[0], size=min(eval_n, X_val.shape[0]), replace=False)
        X_eval = X_val[idx]
        explainer = shap.KernelExplainer(model.predict, background)
        sv = explainer.shap_values(X_eval, nsamples=nsamples)
        return np.mean(np.abs(np.asarray(sv)), axis=0)

    raise ValueError(f"Unsupported model for SHAP: {model_name}")

def plot_segment_bar_all(mean_imp, std_imp, seg_name, seg_slice, out_pdf):
    s0, s1 = seg_slice.start, min(seg_slice.stop, len(mean_imp))
    if s0 >= s1:
        return

    m = mean_imp[s0:s1]
    idx_global = np.arange(s0, s1)

    plt.figure(figsize=(12, 4.0), dpi=300)
    x = np.arange(len(m))
    plt.bar(x, m)   # ← 不再传 yerr

    plt.xticks(x, idx_global, rotation=90)
    plt.xlabel("Feature index")
    plt.ylabel("mean(|SHAP|) across folds")
    plt.title(f"Feature importance (ALL) — {seg_name}")

    # Draw brackets for chemistry feature blocks on self/nbr segments
    if seg_name in {"self", "nbr"}:
        ax = plt.gca()
        maxval = float(m.max()) if m.size else 1.0
        base = -0.12 * (maxval if maxval > 0 else 1.0)
        # Expand y-limits to make room for brackets and labels
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(min(ymin, base - 0.1 * abs(base)), ymax)
        xpos = 0
        for name, width in FEATURE_BLOCKS:
            x0, x1 = xpos, xpos + width - 1
            xc = (x0 + x1) / 2
            ax.annotate(
                "",
                xy=(x0, base),
                xytext=(x1, base),
                arrowprops=dict(
                    arrowstyle="-",
                    connectionstyle="bar,fraction=0.1",
                    color="black",
                    lw=0.8,
                ),
            )
            ax.text(xc, base - 0.025 * abs(base), name, ha="center", va="top", fontsize=9)
            xpos += width

    plt.tight_layout()
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()



def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--model", default=None, help="svm | random_forest | linear_regression")
    p.add_argument("--outdir", default="figs")
    p.add_argument("--folds", type=int, default=None)
    # KernelSHAP controls (svm only)
    p.add_argument("--bg-k", type=int, default=50)
    p.add_argument("--eval-n", type=int, default=200)
    p.add_argument("--nsamples", default="auto")
    args = p.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.cfg)
    cfg.freeze()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    X, y = load_xy(cfg)

    model_name = args.model or cfg.MODEL.NAME
    n_folds = args.folds or int(cfg.SOLVER.NUM_FOLDS)
    seed = int(cfg.SOLVER.SEED)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_imps = []

    for fold_idx, (tr_idx, va_idx) in enumerate(kf.split(X), start=1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_va, y_va = X[va_idx], y[va_idx]

        model = _build_estimator(model_name, seed)
        model.fit(X_tr, y_tr)

        imp = shap_mean_abs_per_fold(
            model_name=model_name,
            model=model,
            X_tr=X_tr,
            X_val=X_va,
            seed=seed + fold_idx,
            bg_k=args.bg_k,
            eval_n=args.eval_n,
            nsamples=args.nsamples,
        )
        fold_imps.append(imp)

    fold_imps = np.stack(fold_imps, axis=0)  # (K, n_features)
    mean_imp = fold_imps.mean(axis=0)
    std_imp = fold_imps.std(axis=0, ddof=1)

    print(f"Model: {model_name}, folds={n_folds}, n_features={mean_imp.shape[0]}")
    for seg_name, seg in SEGMENTS:
        out_pdf = outdir / f"shap_bar_{model_name}_{seg_name}_all.pdf"
        plot_segment_bar_all(mean_imp, std_imp, seg_name, seg, out_pdf)

    print(f"Saved to: {outdir.resolve()}")


if __name__ == "__main__":
    main()
