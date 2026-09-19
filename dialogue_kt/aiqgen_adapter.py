#!/usr/bin/env python3
"""
Track A — AiQGen/HKEAA data adapter for dialogue-kt.
Default (--signal binary): zhu_alignment→correct, zhu_level as KC tag.
  Writes: data/annotated/aiqgen_aiqgen.csv, kc_dict_aiqgen_aiqgen.json, data/aiqgen_hkeaa.jsonl
Hybrid  (--signal qwen): loads aiqgen_qwen_extraction.jsonl, demonstrated_level as KC tag,
  correct=(demo>=demand). Writes: data/annotated/aiqgen_hybrid.csv, kc_dict_aiqgen_hybrid.json
"""
import json, os, sys
from ast import literal_eval
from collections import defaultdict
from pathlib import Path
import pandas as pd

_REPO = Path(__file__).parent.parent
_AITA = _REPO.parent
RESPONSES_PATH  = _AITA / "student_data/Sample_CHI_paper_v1_zhu.jsonl"
KC_LABELS_PATH  = _AITA / "dsh-mvp/data/manifests/kc_hkeaa_a_v54.jsonl"
OUT_CSV         = _REPO / "data/annotated/aiqgen_aiqgen.csv"
OUT_KC_DICT     = _REPO / "data/annotated/kc_dict_aiqgen_aiqgen.json"
OUT_JSONL       = _REPO / "data/aiqgen_hkeaa.jsonl"
EXTRACTION_PATH = _REPO / "data/annotated/aiqgen_qwen_extraction.jsonl"
OUT_HYBRID      = _REPO / "data/annotated/aiqgen_hybrid.csv"
OUT_KC_HYBRID   = _REPO / "data/annotated/kc_dict_aiqgen_hybrid.json"
ZHU_ORDER       = ["複述", "解釋", "重整", "伸展", "評鑑", "創意"]
CORRECT_ALN     = {"meets_target"}
INCORRECT_ALN   = {"below_target", "different_operation"}


def load_kc_labels(path) -> dict:
    """Return {paper_id:question_id → kc_item}."""
    items = {}
    for line in open(path, encoding="utf-8"):
        kc = json.loads(line)
        items[kc["paper_id"] + ":" + kc["question_id"]] = kc
    return items


def _parse_correct(aln: str):
    if aln in CORRECT_ALN:   return True
    if aln in INCORRECT_ALN: return False
    return None


def _make_turn(d, kc, correct, kc_tag):
    obs = d["student_response_observations"]
    return {"qid": d["question_id"],
            "qid_num": int(d["question_id"]) if d["question_id"].isdigit() else 0,
            "correct": correct, "kcs": [kc_tag],
            "teacher": d.get("question_template", "")[:200],
            "student": str(obs[0]["student_answer_verbatim"])[:200] if obs else "",
            "zhu_level": kc["zhu_level"]}


def build_student_sequences(responses_path, kc_items: dict,
                             ext_map: dict = None) -> list:
    """
    Build per-student sequences.
    ext_map=None → binary mode (zhu_alignment signal, demanded KC tag)
    ext_map=dict → hybrid mode (Qwen demonstrated_level as KC tag + correct signal)
    """
    groups = defaultdict(list)
    for line in open(responses_path, encoding="utf-8"):
        d = json.loads(line)
        item_key = d["paper_id"] + ":" + d["question_id"]
        if item_key not in kc_items:
            continue
        kc  = kc_items[item_key]
        sid = d["grade_label"] + "|" + d["paper_id"]
        if ext_map is None:
            correct = _parse_correct(d.get("zhu_alignment", ""))
            if correct is None:
                continue
            kc_tag = kc["zhu_level"]
        else:
            ext = ext_map.get(sid + ":q" + d["question_id"])
            if ext is None:
                continue
            demo    = ext["demonstrated_level"]
            correct = ZHU_ORDER.index(demo) >= ZHU_ORDER.index(kc["zhu_level"])
            kc_tag  = demo
        groups[sid].append(_make_turn(d, kc, correct, kc_tag))
    sequences = []
    for sid, turns in groups.items():
        turns.sort(key=lambda x: x["qid_num"])
        if len(turns) >= 2:
            sequences.append({"student_id": sid, "turns": turns})
    return sequences


def to_dialogue_row(seq: dict) -> dict:
    turns    = seq["turns"]
    dialogue = [{"turn": i+1, "teacher": t["teacher"], "student": t["student"]}
                for i, t in enumerate(turns)]
    annotation = {f"turn {i+1}": {"correct": t["correct"], "kcs": t["kcs"]}
                  for i, t in enumerate(turns)}
    meta_data  = {"student_id": seq["student_id"], "num_turns": len(turns),
                  "correct_rate": round(sum(1 for t in turns if t["correct"]) / len(turns), 3)}
    return {"dialogue": dialogue, "annotation": annotation, "meta_data": meta_data}


def write_flat_jsonl(sequences: list, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seq in sequences:
            for t in seq["turns"]:
                f.write(json.dumps({"turn_id": seq["student_id"] + ":q" + t["qid"],
                    "speaker": "student", "utterance": t["student"],
                    "kc_tags": t["kcs"], "correct": int(t["correct"]),
                    "zhu_level": t["zhu_level"]}, ensure_ascii=False) + "\n")


def verify_with_kt_data_loading(csv_path) -> int:
    sys.path.insert(0, str(_REPO))
    from dialogue_kt.kt_data_loading import DKTDataset
    df = pd.read_csv(csv_path, converters={c: literal_eval for c in ["dialogue","meta_data","annotation"]})
    kc_dict_path = str(csv_path).replace("aiqgen_aiqgen.csv", "kc_dict_aiqgen_aiqgen.json")
    with open(kc_dict_path) as fk:
        kc_dict = json.load(fk)
    return len(DKTDataset(df, kc_dict, None, None).data)


def main():
    for d in [_REPO / "data/annotated", _REPO / "data", _REPO / "results"]:
        os.makedirs(d, exist_ok=True)

    signal = "binary"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--signal" and i + 1 < len(sys.argv[1:]):
            signal = sys.argv[i + 2]

    kc_items = load_kc_labels(KC_LABELS_PATH)

    if signal == "qwen":
        ext_map   = {json.loads(l)["turn_id"]: json.loads(l)
                     for l in open(EXTRACTION_PATH, encoding="utf-8")}
        sequences = build_student_sequences(RESPONSES_PATH, kc_items, ext_map)
        out_csv, out_kc = OUT_HYBRID, OUT_KC_HYBRID
        print(f"[hybrid/qwen] Built {len(sequences)} sequences")
    else:
        sequences = build_student_sequences(RESPONSES_PATH, kc_items)
        out_csv, out_kc = OUT_CSV, OUT_KC_DICT
        for seq in sequences:
            rates = f"{sum(t['correct'] for t in seq['turns'])}/{len(seq['turns'])} correct"
            print(f"  {seq['student_id']}: {len(seq['turns'])} turns, {rates}")

    df = pd.DataFrame([to_dialogue_row(s) for s in sequences])
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}  ({len(df)} rows)")

    all_kcs = sorted({kc for seq in sequences for t in seq["turns"] for kc in t["kcs"]})
    kc_dict = {kc: i for i, kc in enumerate(all_kcs)}
    with open(out_kc, "w", encoding="utf-8") as fk:
        json.dump(kc_dict, fk, ensure_ascii=False, indent=2)
    print(f"Wrote {out_kc}  {kc_dict}")

    if signal != "qwen":
        write_flat_jsonl(sequences, OUT_JSONL)
        print(f"Wrote {OUT_JSONL}  ({sum(len(s['turns']) for s in sequences)} rows)")
        n = verify_with_kt_data_loading(out_csv)
        print(f"DoD check: DKTDataset loaded {n} data points.")


if __name__ == "__main__":
    main()
