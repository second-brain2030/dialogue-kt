#!/usr/bin/env python3
"""
Track C — BKT forward-trajectory projection + student clustering.
Reads bkt_params.json (from aiqgen_train), rolls p(L) forward N=10, k-means
clusters on mastery-gap vectors, replays factorial A/B/C hint-policy per cluster.
Outputs: trajectory_forecast.csv, student_clusters.csv, factorial_by_cluster.csv.
"""
import json
import sys
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

_REPO = Path(__file__).parent.parent
_AITA = _REPO.parent
sys.path.insert(0, str(_AITA))  # expose ai_tutor_simulation

MASTERY_THRESHOLD = 0.90
N_ROLLOUT = 10
K_CLUSTERS = 3


# ── BKT update equations ────────────────────────────────────────────────────

def _bkt_posterior(p_l, correct: bool, guess: float, slip: float) -> float:
    """Bayes-update p(L) given one observed response."""
    if correct:
        p_cor_given_l  = 1.0 - slip
        p_cor_given_nl = guess
    else:
        p_cor_given_l  = slip
        p_cor_given_nl = 1.0 - guess
    numerator   = p_l * p_cor_given_l
    denominator = numerator + (1.0 - p_l) * p_cor_given_nl
    return numerator / denominator if denominator > 0 else p_l


def _bkt_transition(p_l_post, learns: float) -> float:
    """Apply transition: p(L_{t+1}) = p(L|obs) + (1-p(L|obs)) * p(T)."""
    return p_l_post + (1.0 - p_l_post) * learns


def replay_sequence(turns: list, params: dict) -> dict:
    """
    Replay BKT on a student's observed turns.
    Returns {kc_name: p_L_final} for each KC that appears in the sequence.
    """
    kc_state = {kc: params[kc]["prior"] for kc in params}
    for turn in turns:
        kc = turn["kcs"][0]
        if kc not in params:
            continue
        p = params[kc]
        p_l = kc_state[kc]
        p_l = _bkt_posterior(p_l, turn["correct"], p["guesses"], p["slips"])
        p_l = _bkt_transition(p_l, p["learns"])
        kc_state[kc] = p_l
    return kc_state


def roll_forward(p_l_start: float, learns: float) -> int:
    """
    Roll p(L) forward from p_l_start using only learning transitions.
    Returns first step N where p(L_N) > MASTERY_THRESHOLD, or N_ROLLOUT if not reached.
    """
    p_l = p_l_start
    for step in range(1, N_ROLLOUT + 1):
        p_l = _bkt_transition(p_l, learns)
        if p_l >= MASTERY_THRESHOLD:
            return step
    return N_ROLLOUT   # censored: not mastered within N_ROLLOUT


# ── main routines ────────────────────────────────────────────────────────────

def load_sequences():
    """Load annotated CSV and return list of (student_id, turns)."""
    csv = _REPO / "data/annotated/aiqgen_aiqgen.csv"
    df = pd.read_csv(csv, converters={
        col: literal_eval for col in ["dialogue", "meta_data", "annotation"]
    })
    seqs = []
    for _, row in df.iterrows():
        sid = row["meta_data"]["student_id"]
        turns = [
            {"kcs": row["annotation"][f"turn {i+1}"]["kcs"],
             "correct": row["annotation"][f"turn {i+1}"]["correct"]}
            for i in range(len(row["dialogue"]))
        ]
        seqs.append((sid, turns))
    return seqs


def build_trajectories(seqs, bkt_params):
    """Return forecast DataFrame and gap matrix (students × KCs)."""
    kc_names = sorted(bkt_params.keys())
    rows = []
    gap_matrix = []
    student_ids = []
    for sid, turns in seqs:
        final_state = replay_sequence(turns, bkt_params)
        gap_vec = []
        for kc in kc_names:
            p_final = final_state.get(kc, bkt_params[kc]["prior"])
            otm = roll_forward(p_final, bkt_params[kc]["learns"])
            rows.append({"student": sid, "kc": kc,
                         "p_L_final": round(p_final, 4),
                         "opportunities_to_mastery": otm})
            gap_vec.append(1.0 - p_final)
        gap_matrix.append(gap_vec)
        student_ids.append(sid)
    return pd.DataFrame(rows), np.array(gap_matrix), student_ids, kc_names


def cluster_students(gap_matrix, student_ids, kc_names):
    """k-means on mastery-gap vectors; return cluster DataFrame."""
    k = min(K_CLUSTERS, len(student_ids))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(gap_matrix)
    cluster_rows = []
    for sid, cluster, gap_vec in zip(student_ids, labels, gap_matrix):
        top_kc = kc_names[int(np.argmax(gap_vec))]
        cluster_rows.append({"student": sid, "cluster": int(cluster),
                              "top_gap_kc": top_kc,
                              **{f"gap_{kc}": round(g, 4)
                                 for kc, g in zip(kc_names, gap_vec)}})
    df = pd.DataFrame(cluster_rows).sort_values("cluster")
    return df, km.cluster_centers_


def replay_factorial_per_cluster(cluster_df, gap_matrix, student_ids, bkt_params, kc_names):
    """Map each cluster to BKT StudentBKT kwargs and replay the 2×2 experiment."""
    from ai_tutor_simulation.src.factorial_runner import run_2x2_experiment, CONDITION_LABELS
    rows = []
    for cid in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cid]["student"].tolist()
        idxs = [student_ids.index(m) for m in members]
        avg_gap = gap_matrix[idxs].mean(axis=0)
        dom_kc = kc_names[int(np.argmax(avg_gap))]
        p = bkt_params[dom_kc]
        kwargs = {
            "p_l0":    round(1.0 - float(avg_gap[kc_names.index(dom_kc)]), 4),
            "p_guess": round(float(p["guesses"]) if p["guesses"] else 0.15, 4),
            "p_slip":  round(float(p["slips"])   if p["slips"]   else 0.10, 4),
        }
        print(f"  Cluster {cid} (n={len(members)}, dom KC: {dom_kc}): {kwargs}")
        exp = run_2x2_experiment(n_students=200, max_steps=12, random_state=42, student_kwargs=kwargs)
        for cond, label in CONDITION_LABELS.items():
            m = exp[cond]
            rows.append({"cluster": cid, "dominant_kc": dom_kc, "condition": label,
                         "mean_trials_to_mastery": round(m["mean_trials_to_mastery"], 2),
                         "mastery_rate": round(m["mastery_rate"], 1),
                         "leakage_rate": round(m["leakage_rate"], 1)})
    return pd.DataFrame(rows)


def main():
    with open(_REPO / "results/bkt_params.json") as f:
        bkt_params = json.load(f)

    seqs = load_sequences()
    print(f"Loaded {len(seqs)} student sequences")

    forecast_df, gap_matrix, student_ids, kc_names = build_trajectories(seqs, bkt_params)
    forecast_df.to_csv(_REPO / "results/trajectory_forecast.csv", index=False)
    print(f"Wrote results/trajectory_forecast.csv ({len(forecast_df)} rows)")
    print(forecast_df.to_string(index=False))

    cluster_df, centers = cluster_students(gap_matrix, student_ids, kc_names)
    cluster_df.to_csv(_REPO / "results/student_clusters.csv", index=False)
    print(f"\nWrote results/student_clusters.csv")
    print(cluster_df[["student", "cluster", "top_gap_kc"]].to_string(index=False))

    # Interpretation sentence per cluster
    print("\n── Cluster interpretations ──────────────────────────────────")
    for cid in sorted(cluster_df["cluster"].unique()):
        members = cluster_df[cluster_df["cluster"] == cid]
        top_kc = members["top_gap_kc"].mode()[0]
        avg_otm = forecast_df[
            forecast_df["student"].isin(members["student"]) &
            (forecast_df["kc"] == top_kc)
        ]["opportunities_to_mastery"].mean()
        print(f"  Cluster {cid} (n={len(members)}): needs ~{avg_otm:.1f} more opportunities on '{top_kc}'")

    print("\n── Factorial replay per cluster ─────────────────────────────")
    fact_df = replay_factorial_per_cluster(
        cluster_df, gap_matrix, student_ids, bkt_params, kc_names)
    fact_df.to_csv(_REPO / "results/factorial_by_cluster.csv", index=False)
    print(f"Wrote results/factorial_by_cluster.csv")
    print(fact_df.to_string(index=False))


if __name__ == "__main__":
    main()
