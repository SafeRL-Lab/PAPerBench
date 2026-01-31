#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, json, argparse
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

USE_OPENAI = os.getenv("USE_OPENAI", "0") == "1"

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
# VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "https://litellm-991596698159.us-west1.run.app")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")

def client() -> OpenAI:
    if USE_OPENAI:
        return OpenAI()
    return OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

CLI = client()
#  and adds missing constraints/structure
# - Does NOT invent facts not supported by the context.
# - Not overly verbose or poorly structured.

#Output must be exactly one character from {A,B,C,D,E}. No punctuation. No whitespace. No explanation.

SYSTEM = """You are evaluating personalization-aware prompt expansion.
Given context, an initial query, and five candidate rewritten prompts (A–E),
choose the SINGLE best rewritten prompt.

Criteria:
- Best uses the context and personalization targets (persona/preferences/format/domain).
- Removes ambiguity from the initial query.

Return ONLY one letter: A, B, C, D, or E.


"""

LETTER_RE = re.compile(r"\b([ABCDE])\b", re.IGNORECASE)

def extract_letter(text: str) -> str:
    if not text: return ""
    m = LETTER_RE.search(text.strip())
    return m.group(1).upper() if m else ""

def opt_text(options: Dict[str, Any], k: str) -> str:
    v = options.get(k, {})
    if isinstance(v, dict):
        return (v.get("text","") or "").strip()
    return str(v).strip()

# Personalization targets (for reference):
# {json.dumps(targets, ensure_ascii=False)}

def build_user(sample: Dict[str, Any]) -> str:
    context = sample.get("specific_generated_context","") or ""
    initial = sample.get("generated_query","") or ""
    # targets = sample.get("personalization_targets", [])
    mcq = sample["mcq"]
    opts = mcq["options"]

    return f"""Context:
{context}

Initial query (old prompt):
{initial}

Candidate rewritten prompts:
A) {opt_text(opts,'A')}
B) {opt_text(opts,'B')}
C) {opt_text(opts,'C')}
D) {opt_text(opts,'D')}
E) {opt_text(opts,'E')}
"""

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    summary_rows = []

    for model in models:
        records = []
        out_jsonl = outdir / f"pred_{model.replace('/','_')}.jsonl"

        with input_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
            for i, line in enumerate(tqdm(fin, desc=f"eval {model}")):
                if args.limit > 0 and i >= args.limit:
                    break
                line=line.strip()
                if not line: continue
                sample=json.loads(line)
                if "mcq" not in sample: 
                    continue

                gold = (sample["mcq"].get("correct_option") or "").upper()
                user = build_user(sample)
                # print("user----------------: ", user)
                # break

                pred_raw = ""
                pred = ""
                try:
                    # r = CLI.chat.completions.create(
                    #     model=model,
                    #     messages=[{"role":"system","content":SYSTEM},{"role":"user","content":user}],
                    #     temperature=args.temperature,
                    #     max_tokens=args.max_tokens,
                    # )

                    kwargs = dict(
                        model=model,
                        messages=[
                            {"role": "system", "content": SYSTEM},
                            {"role": "user", "content": user},
                        ],
                        temperature=args.temperature,
                    )

                    if USE_OPENAI:
                        # new OpenAI models（gpt-5.x, gpt-4.1）
                        kwargs["max_completion_tokens"] = args.max_tokens
                    else:
                        # vLLM 
                        kwargs["max_tokens"] = args.max_tokens
                    r = CLI.chat.completions.create(**kwargs)

                    pred_raw = r.choices[0].message.content or ""
                    pred = extract_letter(pred_raw)
                except Exception as e:
                    pred_raw = f"ERROR: {e}"
                    pred = ""

                rec = {
                    "id": sample.get("id"),
                    "gold": gold,
                    "pred": pred,
                    "correct": int(pred == gold),
                    "pred_raw": pred_raw,
                }
                records.append(rec)
                fout.write(json.dumps({"id": rec["id"], "gold": gold, "pred": pred, "pred_raw": pred_raw}, ensure_ascii=False) + "\n")

        df = pd.DataFrame(records)
        acc = float(df["correct"].mean()) if len(df) else 0.0
        pred_dist = df["pred"].value_counts(dropna=False).to_dict()

        # personalization diagnostic: choosing ignore_context option
        ignore_cnt = 0
        hallu_cnt = 0

        #
        pred_type_list: List[str] = []
        with input_path.open("r", encoding="utf-8") as fin:
            sample_map = {}
            for line in fin:
                line=line.strip()
                if not line: continue
                s=json.loads(line)
                if "mcq" in s:
                    sample_map[s.get("id")] = s["mcq"]["options"]

        for rec in records:
            opts = sample_map.get(rec["id"])
            if not opts or rec["pred"] not in opts:
                pred_type_list.append("")
                continue
            t = (opts[rec["pred"]].get("type") if isinstance(opts[rec["pred"]], dict) else "")
            pred_type_list.append(t)
            if t == "ignore_context":
                ignore_cnt += 1
            if t == "hallucination":
                hallu_cnt += 1

        ignore_rate = ignore_cnt / max(len(records), 1)
        hallu_rate = hallu_cnt / max(len(records), 1)

        summary_rows.append({
            "model": model,
            "n": int(len(records)),
            "accuracy": acc,
            "ignore_context_rate": ignore_rate,
            "hallucination_trap_rate": hallu_rate,
            "pred_dist": json.dumps(pred_dist, ensure_ascii=False),
        })

        # confusion matrix by letters
        if len(df):
            cm = pd.crosstab(df["gold"], df["pred"], dropna=False)
            cm.to_csv(outdir / f"confusion_letters_{model.replace('/','_')}.csv")

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
    print(f"[DONE] Wrote: {outdir/'summary.csv'}")

if __name__ == "__main__":
    main()
