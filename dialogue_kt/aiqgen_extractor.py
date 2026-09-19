#!/usr/bin/env python3
"""
Qwen3-30B-AWQ zero-shot extraction of demonstrated Zhu cognitive level per turn.
Reads  : data/aiqgen_hkeaa.jsonl  (110 turns)
Writes : data/annotated/aiqgen_qwen_extraction.jsonl
Row: turn_id, demanded_level, demonstrated_level, confidence, reason, extraction_status
  status: "ok" | "fallback"  (fallback = Pydantic fail after 2 retries → copy zhu_level)
Usage: python3 -m dialogue_kt.aiqgen_extractor [--force]
"""
import json, re, sys, urllib.request
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator

_REPO     = Path(__file__).parent.parent
_AITA     = _REPO.parent
JSONL_IN  = _REPO / "data/aiqgen_hkeaa.jsonl"
KC_PATH   = _AITA / "dsh-mvp/data/manifests/kc_hkeaa_a_v54.jsonl"
CACHE_OUT = _REPO / "data/annotated/aiqgen_qwen_extraction.jsonl"
ZHU_LEVELS   = ["複述", "解釋", "重整", "伸展", "評鑑", "創意"]
QWEN_URL     = "http://10.103.1.2:9001/v1/chat/completions"
QWEN_MODEL   = "Qwen3-32B-AWQ"
MAX_RETRIES  = 2
THINK_RE     = re.compile(r"<think>.*?</think>", re.DOTALL)


# ── Pydantic output schema ─────────────────────────────────────────────────

class ZhuExtraction(BaseModel):
    demonstrated_level: str
    confidence: float
    reason: str

    @field_validator("demonstrated_level")
    @classmethod
    def must_be_zhu(cls, v):
        if v not in ZHU_LEVELS:
            raise ValueError(f"not a valid Zhu level: {v}")
        return v

    @field_validator("confidence")
    @classmethod
    def clamp_conf(cls, v): return max(0.0, min(1.0, float(v)))

    @field_validator("reason")
    @classmethod
    def truncate_reason(cls, v): return v[:60]


# ── KC lookup ──────────────────────────────────────────────────────────────

def load_kc_map(path) -> dict:
    """Return {paper_id:question_id → {question, official_answer}}."""
    result = {}
    for line in open(path, encoding="utf-8"):
        row = json.loads(line)
        key = row["paper_id"] + ":" + row["question_id"]
        result[key] = {"question": str(row.get("question") or "")[:300],
                       "official_answer": str(row.get("official_answer") or "")[:300]}
    return result


def _kc_key(turn_id: str) -> str:
    # "第一級|2020-DSE-CHI-Paper1:q4" → "2020-DSE-CHI-Paper1:4"
    rest = turn_id.split("|", 1)[-1]            # "2020-DSE-CHI-Paper1:q4"
    paper, _, q = rest.rpartition(":")
    return paper + ":" + q.lstrip("q")


# ── LLM call ──────────────────────────────────────────────────────────────

SYSTEM = ("你是中文閱讀理解評估助手。根據題目、參考答案和學生答案，判斷學生此題的"
          "實際認知操作層次（朱作仁六層次）。只輸出JSON，不加任何解釋。")

USER_TMPL = (
    "Input variables:\n"
    "  question_text: {question_text}\n"
    "  student_answer: {student_answer}\n"
    "  demanded_level: {demanded_level}\n"
    "  official_answer_excerpt: {official_answer_excerpt}\n\n"
    "Output JSON keys:\n"
    "  demonstrated_level: one of {levels}\n"
    "  confidence: float 0.0–1.0\n"
    "  reason: ≤40 Chinese characters\n"
)


def _call_qwen(messages: list) -> str:
    payload = {"model": QWEN_MODEL, "messages": messages, "temperature": 0.0,
               "max_tokens": 200, "response_format": {"type": "json_object"},
               "chat_template_kwargs": {"enable_thinking": False}}
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = urllib.request.Request(QWEN_URL, data=body,
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return str(data["choices"][0]["message"]["content"] or "")


def _parse_json(raw: str) -> Optional[dict]:
    cleaned = THINK_RE.sub("", raw).strip()
    m = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def extract_one(turn: dict, kc_map: dict) -> dict:
    """Run Qwen extraction for one turn; fall back to zhu_level on failure."""
    kc = kc_map.get(_kc_key(turn["turn_id"]), {})
    user_msg = USER_TMPL.format(
        question_text=kc.get("question", ""), student_answer=turn["utterance"][:300],
        demanded_level=turn["zhu_level"], official_answer_excerpt=kc.get("official_answer", ""),
        levels=", ".join(ZHU_LEVELS),
    )
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user",   "content": user_msg}]

    last_err = ""
    for _ in range(MAX_RETRIES):
        try:
            raw = _call_qwen(messages)
            parsed = _parse_json(raw)
            if parsed is None:
                last_err = "no JSON found"; continue
            # Accept LLM's preferred key names, map to our schema
            demo   = parsed.get("demonstrated_level") or parsed.get("level") or ""
            conf   = parsed.get("confidence", 0.5)
            reason = parsed.get("reason") or parsed.get("explanation") or ""
            ext = ZhuExtraction(demonstrated_level=demo, confidence=conf, reason=reason)
            return {"turn_id": turn["turn_id"], "demanded_level": turn["zhu_level"],
                    "demonstrated_level": ext.demonstrated_level, "confidence": ext.confidence,
                    "reason": ext.reason, "extraction_status": "ok"}
        except Exception as e:
            last_err = str(e)

    # Fallback: fail transparently (do not silently convert to binary correct)
    print(f"  [FALLBACK] {turn['turn_id']}: {last_err}", file=sys.stderr)
    return {"turn_id": turn["turn_id"], "demanded_level": turn["zhu_level"],
            "demonstrated_level": turn["zhu_level"], "confidence": 0.0,
            "reason": f"fallback:{last_err[:40]}", "extraction_status": "fallback"}


# ── main ──────────────────────────────────────────────────────────────────

def main():
    force = "--force" in sys.argv
    CACHE_OUT.parent.mkdir(parents=True, exist_ok=True)

    existing = {}
    if CACHE_OUT.exists() and not force:
        for line in open(CACHE_OUT, encoding="utf-8"):
            row = json.loads(line); existing[row["turn_id"]] = row

    kc_map = load_kc_map(KC_PATH)
    turns  = [json.loads(l) for l in open(JSONL_IN, encoding="utf-8")]
    print(f"Turns to process: {len(turns)}  Already cached: {len(existing)}")

    results = dict(existing)
    for i, turn in enumerate(turns):
        if turn["turn_id"] in results:
            continue
        print(f"  [{i+1}/{len(turns)}] {turn['turn_id'][:60]}")
        results[turn["turn_id"]] = extract_one(turn, kc_map)

    ordered = [results[t["turn_id"]] for t in turns if t["turn_id"] in results]
    with open(CACHE_OUT, "w", encoding="utf-8") as f:
        for row in ordered:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    ok = sum(1 for r in ordered if r["extraction_status"] == "ok")
    fb = sum(1 for r in ordered if r["extraction_status"] == "fallback")
    avg_conf = sum(r["confidence"] for r in ordered if r["extraction_status"] == "ok") / max(ok, 1)
    print(f"\nWrote {CACHE_OUT}  ({len(ordered)} rows)  ok={ok} fallback={fb}  mean_conf={avg_conf:.3f}")


if __name__ == "__main__":
    main()
