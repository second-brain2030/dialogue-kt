#!/usr/bin/env python3
"""
Track B — BKT / DKT-multi-KC / DKT-SEM runner on the AiQGen/HKEAA dataset.
Outputs: results/aiqgen_comparison.csv, guess_slip_by_zhu.csv, bkt_params.json.
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

# ── data loading ────────────────────────────────────────────────────────────

def load_aiqgen_splits():
    """Return (train_df, val_df, test_df) from the annotated CSV."""
    csv = _REPO / "data/annotated/aiqgen_aiqgen.csv"
    df = pd.read_csv(csv, converters={col: literal_eval for col in ["dialogue", "meta_data", "annotation"]}).sample(frac=1, random_state=221)
    n = len(df)
    return df[:int(.7 * n)], df[int(.7 * n):int(.85 * n)], df[int(.85 * n):]


def load_kc_dict():
    with open(_REPO / "data/annotated/kc_dict_aiqgen_aiqgen.json") as f:
        return json.load(f)


# ── BKT ─────────────────────────────────────────────────────────────────────

def _dataset_to_bkt_df(dataset: DKTDataset) -> pd.DataFrame:
    """Convert DKTDataset to pyBKT long format."""
    rows, order_id = [], 0
    for sample in dataset.data:
        for kc, label in zip(sample["kc_ids_flat"], sample["labels_flat"]):
            rows.append({"user_id": sample["dialogue_idx"],
                         "skill_name": str(kc), "correct": label, "order_id": order_id})
            order_id += 1
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

    # Extract per-skill params — pyBKT skill index → KC name via kc_dict order
    params_df = model.params().reset_index()
    kc_names = list(kc_dict.keys())
    params_dict = {}
    for skill in params_df["skill"].unique():
        sk = params_df[params_df["skill"] == skill]
        get = lambda p: float(v.iloc[0]) if len(v := sk[sk["param"] == p]["value"]) else None
        params_dict[kc_names[int(skill)]] = {
            "prior": get("prior"), "learns": get("learns"),
            "guesses": get("guesses"), "slips": get("slips"),
        }
    return {
        "model": "BKT", "auc": round(auc, 4), "accuracy": round(acc, 4),
        "n_train": len(bkt_train), "n_test": len(bkt_test),
        "elapsed_s": round(time.time() - t0, 1)
    }, params_dict


# ── DKT helpers ─────────────────────────────────────────────────────────────

def _eval_dkt(model, loader, kc_dict):
    """Single eval pass; returns (accuracy, auc) or (None, None) if no data."""
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for batch in loader:
            labels = batch["labels"][:, 1:]           # shift: predict next turn
            kc_ids = batch["kc_ids"]                  # B x L x K
            num_kcs = batch["num_kcs"]
            if labels.numel() == 0:
                continue
            y = model(batch)                          # B x L x num_kcs
            # Gather KC probs for next turn, average across KCs
            corr_probs = torch.gather(y[:, :-1], 2,
                kc_ids[:, 1:]).sum(2) / num_kcs[:, 1:]
            mask = labels != -100
            all_labels.extend(labels[mask].tolist())
            all_preds.extend(corr_probs[mask].tolist())
    if len(set(all_labels)) < 2:      # need both classes for AUC
        return None, accuracy_score(all_labels, [round(p) for p in all_preds])
    return (roc_auc_score(all_labels, all_preds),
            accuracy_score(all_labels, [round(p) for p in all_preds]))


def _train_dkt(model, train_loader, epochs=40, lr=1e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCELoss()
    for _ in range(epochs):
        model.train()
        for batch in train_loader:
            labels = batch["labels"][:, 1:].float()
            y = model(batch)[:, :-1]
            kc_ids = batch["kc_ids"][:, 1:]
            num_kcs = batch["num_kcs"][:, 1:]
            preds = torch.gather(y, 2, kc_ids).sum(2) / num_kcs
            mask = batch["labels"][:, 1:] != -100
            if mask.sum() == 0:
                continue
            loss = loss_fn(preds[mask], labels[mask])
            opt.zero_grad(); loss.backward(); opt.step()


def run_dkt_multi(train_df, test_df, kc_dict):
    """Train DKT-multi-KC and return metrics dict."""
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
    """Train DKT-SEM with multilingual SBERT KC embeddings and return metrics dict."""
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


# ── main ────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_REPO / "results", exist_ok=True)
    kc_dict = load_kc_dict()
    train_df, val_df, test_df = load_aiqgen_splits()
    # Use train+val for fitting (small dataset), hold test out
    fit_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    print(f"Data split: fit={len(fit_df)}, test={len(test_df)} sequences\n")

    results = []
    print("── BKT ──────────────────────────────────")
    bkt_metrics, bkt_params = run_bkt(fit_df, test_df, kc_dict)
    results.append(bkt_metrics); print(bkt_metrics)
    with open(_REPO / "results/bkt_params.json", "w") as fp:
        json.dump(bkt_params, fp, indent=2, ensure_ascii=False)
    print("  Saved bkt_params.json")

    print("── DKT-multi-KC ─────────────────────────")
    dkt_metrics = run_dkt_multi(fit_df, test_df, kc_dict)
    results.append(dkt_metrics); print(dkt_metrics)

    print("── DKT-SEM ──────────────────────────────")
    sem_metrics = run_dkt_sem(fit_df, test_df, kc_dict)
    results.append(sem_metrics); print(sem_metrics)

    pd.DataFrame(results).to_csv(_REPO / "results/aiqgen_comparison.csv", index=False)
    print("Wrote results/aiqgen_comparison.csv")
    gs_df = pd.DataFrame([{"zhu_level": kc, **p} for kc, p in bkt_params.items()])
    gs_df.to_csv(_REPO / "results/guess_slip_by_zhu.csv", index=False)
    print(f"Wrote results/guess_slip_by_zhu.csv\n{gs_df.to_string(index=False)}")


if __name__ == "__main__":
    main()
