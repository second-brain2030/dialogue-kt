#!/usr/bin/env python3
"""
Track A — AiQGen/HKEAA data adapter for dialogue-kt.

Joins HKEAA student responses with KC labels (Zhu cognitive level as KC tag),
groups by student (grade_label × paper_id), and writes:
  data/annotated/aiqgen_aiqgen.csv   — annotated dialogue sequences
  data/annotated/kc_dict_aiqgen_aiqgen.json  — KC name → integer index
  data/aiqgen_hkeaa.jsonl            — flat per-turn JSONL for inspection

Correct/incorrect signal: zhu_alignment=='meets_target' → True,
  'below_target'/'different_operation' → False; others skipped.
KC tag: item's zhu_level (重整 / 伸展 / 評鑑) — 3-class cognitive demand.
"""
import json
import os
from ast import literal_eval
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── paths resolved relative to this file so cwd doesn't matter ────────────
_REPO = Path(__file__).parent.parent          # dialogue-kt/
_AITA = _REPO.parent                          # AiTA/

RESPONSES_PATH = _AITA / "student_data/Sample_CHI_paper_v1_zhu.jsonl"
KC_LABELS_PATH = _AITA / "dsh-mvp/data/manifests/kc_hkeaa_a_v54.jsonl"
OUT_CSV     = _REPO / "data/annotated/aiqgen_aiqgen.csv"
OUT_KC_DICT = _REPO / "data/annotated/kc_dict_aiqgen_aiqgen.json"
OUT_JSONL   = _REPO / "data/aiqgen_hkeaa.jsonl"

CORRECT_ALN   = {"meets_target"}
INCORRECT_ALN = {"below_target", "different_operation"}


# ── helpers ────────────────────────────────────────────────────────────────

def load_kc_labels(path) -> dict:
    """Return {paper_id:question_id → kc_item} from kc_hkeaa_a_v54.jsonl."""
    items = {}
    for line in open(path, encoding="utf-8"):
        kc = json.loads(line)
        key = kc["paper_id"] + ":" + kc["question_id"]
        items[key] = kc
    return items


def _parse_correct(zhu_alignment: str):
    """Return True/False/None (None = skip this row)."""
    if zhu_alignment in CORRECT_ALN:
        return True
    if zhu_alignment in INCORRECT_ALN:
        return False
    return None


def build_student_sequences(responses_path, kc_items: dict) -> list:
    """
    Group HKEAA rows by (grade_label, paper_id) → student sequence.
    Each group becomes one dialogue row.  Sequences with < 2 turns are
    dropped (DKTDataset requires ≥ 2).
    """
    groups = defaultdict(list)
    for line in open(responses_path, encoding="utf-8"):
        d = json.loads(line)
        item_key = d["paper_id"] + ":" + d["question_id"]
        if item_key not in kc_items:
            continue
        kc = kc_items[item_key]
        correct = _parse_correct(d.get("zhu_alignment", ""))
        if correct is None:
            continue
        obs = d["student_response_observations"]
        answer = str(obs[0]["student_answer_verbatim"]) if obs else ""
        sid = d["grade_label"] + "|" + d["paper_id"]
        q_num = int(d["question_id"]) if d["question_id"].isdigit() else 0
        groups[sid].append({
            "qid": d["question_id"],
            "qid_num": q_num,
            "correct": correct,
            "kcs": [kc["zhu_level"]],   # Zhu level as KC tag
            "teacher": d.get("question_template", "")[:200],
            "student": answer[:200],
            "zhu_level": kc["zhu_level"],
        })

    sequences = []
    for sid, turns in groups.items():
        turns.sort(key=lambda x: x["qid_num"])
        if len(turns) >= 2:
            sequences.append({"student_id": sid, "turns": turns})
    return sequences


def to_dialogue_row(seq: dict) -> dict:
    """Convert student sequence → dialogue-kt annotated DataFrame row."""
    turns = seq["turns"]
    dialogue = [
        {"turn": i + 1, "teacher": t["teacher"], "student": t["student"]}
        for i, t in enumerate(turns)
    ]
    annotation = {
        f"turn {i + 1}": {"correct": t["correct"], "kcs": t["kcs"]}
        for i, t in enumerate(turns)
    }
    meta_data = {
        "student_id": seq["student_id"],
        "num_turns": len(turns),
        "correct_rate": round(sum(1 for t in turns if t["correct"]) / len(turns), 3),
    }
    return {"dialogue": dialogue, "annotation": annotation, "meta_data": meta_data}


def write_flat_jsonl(sequences: list, path):
    """Write one row per (student, question) turn for human inspection."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seq in sequences:
            for turn in seq["turns"]:
                f.write(json.dumps({
                    "turn_id": seq["student_id"] + ":q" + turn["qid"],
                    "speaker": "student",
                    "utterance": turn["student"],
                    "kc_tags": turn["kcs"],
                    "correct": int(turn["correct"]),
                    "zhu_level": turn["zhu_level"],
                }, ensure_ascii=False) + "\n")


def verify_with_kt_data_loading(csv_path) -> int:
    """Load the CSV through DKTDataset and return total data points (DoD check)."""
    import sys
    sys.path.insert(0, str(_REPO))
    from dialogue_kt.kt_data_loading import DKTDataset
    df = pd.read_csv(csv_path, converters={
        col: literal_eval for col in ["dialogue", "meta_data", "annotation"]
    })
    kc_dict_path = str(csv_path).replace("aiqgen_aiqgen.csv", "kc_dict_aiqgen_aiqgen.json")
    with open(kc_dict_path) as fk:
        kc_dict = json.load(fk)
    ds = DKTDataset(df, kc_dict, None, None)
    return len(ds.data)


# ── main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(_REPO / "data/annotated", exist_ok=True)
    os.makedirs(_REPO / "data", exist_ok=True)
    os.makedirs(_REPO / "results", exist_ok=True)

    kc_items = load_kc_labels(KC_LABELS_PATH)
    sequences = build_student_sequences(RESPONSES_PATH, kc_items)
    print(f"Built {len(sequences)} student sequences:")
    for seq in sequences:
        rates = f"{sum(t['correct'] for t in seq['turns'])}/{len(seq['turns'])} correct"
        print(f"  {seq['student_id']}: {len(seq['turns'])} turns, {rates}")

    # annotated CSV
    rows = [to_dialogue_row(seq) for seq in sequences]
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote {OUT_CSV}  ({len(df)} rows)")

    # KC dict (Zhu level name → integer index)
    all_kcs = sorted({kc for seq in sequences for t in seq["turns"] for kc in t["kcs"]})
    kc_dict = {kc: i for i, kc in enumerate(all_kcs)}
    with open(OUT_KC_DICT, "w", encoding="utf-8") as fk:
        json.dump(kc_dict, fk, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT_KC_DICT}  {kc_dict}")

    # flat JSONL
    write_flat_jsonl(sequences, OUT_JSONL)
    total_turns = sum(len(s["turns"]) for s in sequences)
    print(f"Wrote {OUT_JSONL}  ({total_turns} rows)")

    # DoD verification
    n = verify_with_kt_data_loading(OUT_CSV)
    print(f"\nDoD check: DKTDataset loaded {n} data points (sequences with ≥2 turns).")


if __name__ == "__main__":
    main()
