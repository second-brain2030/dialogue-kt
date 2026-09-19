#!/usr/bin/env python3
"""
Track B — BKT / DKT-multi-KC / DKT-SEM / BKT+Qwen runner on the AiQGen/HKEAA dataset.
Outputs: results/aiqgen_comparison.csv, hybrid_comparison.csv, guess_slip_by_zhu.csv, bkt_params.json
BKT via pyBKT; DKT-multi/SEM via local PyTorch. Does NOT import training.py.
"""
import json
import os
import time
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from pyBKT.models import Model as BKT

# Local model imports (no pykt dependency)
from dialogue_kt.models.dkt_multi_kc import DKTMultiKC
from dialogue_kt.models.dkt_sem import DKTSem
from dialogue_kt.kt_data_loading import DKTDataset, DKTCollator, get_dataloader
from dialogue_kt.utils import device

_REPO = Path(__file__).parent.parent

def _load_splits(csv, kc_json):
    """70/15/15 split; return (fit_df, test_df, kc_dict)."""
    df = pd.read_csv(csv, converters={c: literal_eval for c in ["dialogue","meta_data","annotation"]}).sample(frac=1, random_state=221)
    n = len(df); fit = pd.concat([df[:int(.7*n)], df[int(.7*n):int(.85*n)]]).reset_index(drop=True)
    with open(kc_json) as f: kc_dict = json.load(f)
    return fit, df[int(.85*n):], kc_dict

def load_aiqgen_splits():
    fit, test, _ = _load_splits(_REPO/"data/annotated/aiqgen_aiqgen.csv",
                                _REPO/"data/annotated/kc_dict_aiqgen_aiqgen.json")
    return fit, test

def load_kc_dict():
    with open(_REPO / "data/annotated/kc_dict_aiqgen_aiqgen.json") as f: return json.load(f)


def _dataset_to_bkt_df(dataset: DKTDataset) -> pd.DataFrame:
    rows, order_id = [], 0
    for s in dataset.data:
        for kc, label in zip(s["kc_ids_flat"], s["labels_flat"]):
            rows.append({"user_id": s["dialogue_idx"], "skill_name": str(kc),
                         "correct": label, "order_id": order_id}); order_id += 1
    return pd.DataFrame(rows)


def run_bkt(train_df, test_df, kc_dict):
    """Fit BKT, evaluate on test split, return (metrics_dict, params_dict)."""
    t0 = time.time()
    train_ds = DKTDataset(train_df, kc_dict, None, None)
    test_ds  = DKTDataset(test_df,  kc_dict, None, None)
    bkt_train = _dataset_to_bkt_df(train_ds)
    bkt_test  = _dataset_to_bkt_df(test_ds)

    model = BKT(seed=42, num_fits=5)
    model.fit(data=bkt_train)

    eval_df = model.evaluate(data=bkt_test, metric=["accuracy", "auc"])
    acc = float(eval_df["accuracy"].mean() if isinstance(eval_df, pd.DataFrame) else eval_df[0])
    auc = float(eval_df["auc"].mean()      if isinstance(eval_df, pd.DataFrame) else eval_df[1])
    params_df  = model.params().reset_index()
    kc_names   = list(kc_dict.keys())
    params_dict = {kc_names[int(skill)]: {
        "prior": float(v.iloc[0]) if len(v := sk[sk["param"]=="prior"]["value"]) else None,
        "learns": float(v.iloc[0]) if len(v := sk[sk["param"]=="learns"]["value"]) else None,
        "guesses": float(v.iloc[0]) if len(v := sk[sk["param"]=="guesses"]["value"]) else None,
        "slips": float(v.iloc[0]) if len(v := sk[sk["param"]=="slips"]["value"]) else None,
    } for skill in params_df["skill"].unique()
      for sk in [params_df[params_df["skill"] == skill]]}
    return {
        "model": "BKT", "auc": round(auc, 4), "accuracy": round(acc, 4),
        "n_train": len(bkt_train), "n_test": len(bkt_test),
        "elapsed_s": round(time.time() - t0, 1)
    }, params_dict


# ── DKT helpers ─────────────────────────────────────────────────────────────

def _eval_dkt(model, loader, kc_dict):
    """Single eval pass; returns (auc, accuracy) or (None, acc) if one class."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"][:, 1:]
            if labels.numel() == 0: continue
            y = model(batch)[:, :-1]
            probs = torch.gather(y, 2, batch["kc_ids"][:, 1:]).sum(2) / batch["num_kcs"][:, 1:]
            mask = labels != -100
            all_labels.extend(labels[mask].tolist())
            all_preds.extend(probs[mask].tolist())
    acc = accuracy_score(all_labels, [round(p) for p in all_preds])
    if len(set(all_labels)) < 2: return None, acc
    return roc_auc_score(all_labels, all_preds), acc


def _train_dkt(model, train_loader, epochs=40, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr); loss_fn = torch.nn.BCELoss()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch["labels"][:, 1:].float()
            preds  = torch.gather(model(batch)[:, :-1], 2, batch["kc_ids"][:, 1:]).sum(2) / batch["num_kcs"][:, 1:]
            mask   = batch["labels"][:, 1:] != -100
            if mask.sum() == 0: continue
            loss = loss_fn(preds[mask], labels[mask])
            opt.zero_grad(); loss.backward(); opt.step()


def run_dkt_multi(train_df, test_df, kc_dict):
    t0 = time.time()
    collator = DKTCollator(flatten_kcs=False)
    train_ds = DKTDataset(train_df, kc_dict, None, None)
    test_ds  = DKTDataset(test_df,  kc_dict, None, None)
    model = DKTMultiKC(num_kcs=len(kc_dict), emb_size=16).to(device)
    _train_dkt(model, get_dataloader(train_ds, collator, 8, True))
    auc, acc = _eval_dkt(model, get_dataloader(test_ds, collator, 8, False), kc_dict)
    return {"model": "DKT-multi-KC", "auc": round(auc, 4) if auc else "N/A",
            "accuracy": round(acc, 4) if acc else "N/A",
            "n_train": len(train_ds.data), "n_test": len(test_ds.data),
            "elapsed_s": round(time.time() - t0, 1)}


def run_dkt_sem(train_df, test_df, kc_dict):
    from sentence_transformers import SentenceTransformer
    t0 = time.time()
    sbert = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    kcs_sorted = [k for k, _ in sorted(kc_dict.items(), key=lambda kv: kv[1])]
    kc_emb = torch.tensor(sbert.encode(kcs_sorted), dtype=torch.float).to(device)
    collator = DKTCollator(flatten_kcs=False)
    train_ds = DKTDataset(train_df, kc_dict, kc_emb, sbert)
    test_ds  = DKTDataset(test_df,  kc_dict, kc_emb, sbert)
    model = DKTSem(emb_size=16, kc_emb_matrix=kc_emb).to(device)
    _train_dkt(model, get_dataloader(train_ds, collator, 8, True))
    auc, acc = _eval_dkt(model, get_dataloader(test_ds, collator, 8, False), kc_dict)
    return {"model": "DKT-SEM", "auc": round(auc, 4) if auc else "N/A",
            "accuracy": round(acc, 4) if acc else "N/A",
            "n_train": len(train_ds.data), "n_test": len(test_ds.data),
            "elapsed_s": round(time.time() - t0, 1)}


# ── BKT Hybrid (MVP 5) ──────────────────────────────────────────────────────

def run_bkt_hybrid():
    """Fit BKT on hybrid CSV (Qwen demonstrated_level as KC tag). Return metrics dict."""
    fit_df, test_df, kc_dict = _load_splits(
        _REPO / "data/annotated/aiqgen_hybrid.csv",
        _REPO / "data/annotated/kc_dict_aiqgen_hybrid.json")
    metrics, _ = run_bkt(fit_df, test_df, kc_dict)
    metrics["model"] = "BKT+Qwen"
    return metrics


# ── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_REPO / "results", exist_ok=True)
    kc_dict = load_kc_dict()
    fit_df, test_df = load_aiqgen_splits()
    print(f"Data split: fit={len(fit_df)}, test={len(test_df)} sequences\n")

    results = []
    print("── BKT ──────────────────────────────────")
    bkt_metrics, bkt_params = run_bkt(fit_df, test_df, kc_dict)
    results.append(bkt_metrics); print(bkt_metrics)
    with open(_REPO / "results/bkt_params.json", "w") as fp:
        json.dump(bkt_params, fp, indent=2, ensure_ascii=False)

    print("── DKT-multi-KC ─────────────────────────")
    dkt_metrics = run_dkt_multi(fit_df, test_df, kc_dict)
    results.append(dkt_metrics); print(dkt_metrics)

    print("── DKT-SEM ──────────────────────────────")
    sem_metrics = run_dkt_sem(fit_df, test_df, kc_dict)
    results.append(sem_metrics); print(sem_metrics)

    print("── BKT + Qwen hybrid ────────────────────")
    hybrid_metrics = run_bkt_hybrid()
    results.append(hybrid_metrics); print(hybrid_metrics)

    pd.DataFrame(results).to_csv(_REPO / "results/aiqgen_comparison.csv", index=False)
    print("Wrote results/aiqgen_comparison.csv")

    # Write hybrid_comparison.csv: binary BKT vs Qwen-hybrid BKT side by side
    pd.DataFrame([bkt_metrics, hybrid_metrics]).to_csv(
        _REPO / "results/hybrid_comparison.csv", index=False)
    print("Wrote results/hybrid_comparison.csv")

    gs_df = pd.DataFrame([{"zhu_level": kc, **p} for kc, p in bkt_params.items()])
    gs_df.to_csv(_REPO / "results/guess_slip_by_zhu.csv", index=False)
    print(f"Wrote results/guess_slip_by_zhu.csv\n{gs_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
