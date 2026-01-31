#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust evaluator for personalization prompt-expansion MCQ (A–E).

Supports OpenAI-compatible Chat Completions via:
  1) Official OpenAI API (USE_OPENAI=1, OPENAI_API_KEY set)
  2) vLLM OpenAI-compatible server (USE_OPENAI=0, VLLM_BASE_URL, VLLM_API_KEY)
  3) LiteLLM proxy (Gemini/Claude/etc.) (USE_OPENAI=0, LITELLM_BASE_URL, LITELLM_API_KEY)

Key robustness:
  - Handles multiple MCQ JSON formats (mcq.options, mcq.A-E, mcq.choices list, etc.)
  - Robustly extracts model text from message.content, thinking_blocks, or choice.text
  - Gemini-3 forced-thinking: retries with exponential token budget until visible text appears
  - Writes per-sample JSONL predictions + summary.csv + confusion matrix
"""

import os
import re
import json
import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import pandas as pd
from tqdm import tqdm
from openai import OpenAI

import time

# -------------------------
# Backend configuration
# -------------------------

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip() == "1"

USE_OPENAI = _env_bool("USE_OPENAI", "0")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1").strip()
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY").strip()

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL", "").strip()
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY", "").strip()

def make_client(use_openai: bool, base_url: Optional[str] = None, api_key: Optional[str] = None) -> Tuple[OpenAI, bool]:
    """
    Returns (client, is_official_openai).
    """
    if use_openai:
        return OpenAI(), True

    if base_url:
        return OpenAI(base_url=base_url, api_key=api_key or "EMPTY"), False

    if LITELLM_BASE_URL:
        return OpenAI(base_url=LITELLM_BASE_URL, api_key=(api_key or LITELLM_API_KEY or "EMPTY")), False

    return OpenAI(base_url=VLLM_BASE_URL, api_key=(api_key or VLLM_API_KEY or "EMPTY")), False

# -------------------------
# Prompt / Parsing helpers
# -------------------------

# SYSTEM = """You are evaluating personalization-aware prompt expansion.
# Given a context, an initial query, and five candidate rewritten prompts (A–E),
# choose the SINGLE best rewritten prompt.

# Criteria:
# - Best uses the context and personalization targets (persona/preferences/format/domain/constraints).
# - Removes ambiguity from the initial query.
# - Does NOT invent facts not supported by the context.

# Return ONLY one letter: A, B, C, D, or E.
# """

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
    if not text:
        return ""
    m = LETTER_RE.search(text.strip())
    return m.group(1).upper() if m else ""

def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    if isinstance(x, dict):
        # common shapes: {"text": "..."} or {"prompt": "..."} or {"content": "..."}
        for k in ("text", "prompt", "content", "value"):
            if k in x and isinstance(x[k], str):
                return x[k].strip()
        return json.dumps(x, ensure_ascii=False)
    return str(x).strip()

def normalize_options_to_AE(opts_any: Any) -> Optional[Dict[str, Any]]:
    """
    Normalize various option representations to dict with keys A-E.

    Accepts:
      - dict {"A": {...}, "B": "...", ...}
      - list [{"label":"A","text":"..."}, ...]
      - list [{"option":"A", "prompt":"..."}, ...]
    """
    if opts_any is None:
        return None

    # If already dict with letters
    if isinstance(opts_any, dict):
        if all(k in opts_any for k in ("A", "B", "C", "D", "E")):
            return {k: opts_any[k] for k in ("A", "B", "C", "D", "E")}
        # Sometimes nested dict: {"options": {...}}
        if "options" in opts_any:
            return normalize_options_to_AE(opts_any.get("options"))
        # Sometimes keys are lowercase
        if all(k.lower() in opts_any for k in ("a", "b", "c", "d", "e")):
            return {k.upper(): opts_any[k.lower()] for k in ("a", "b", "c", "d", "e")}

    # If list of option dicts
    if isinstance(opts_any, list):
        out: Dict[str, Any] = {}
        for item in opts_any:
            if isinstance(item, dict):
                lab = item.get("label") or item.get("option") or item.get("key") or item.get("id")
                if isinstance(lab, str):
                    lab = lab.strip().upper()
                if lab in ("A", "B", "C", "D", "E"):
                    out[lab] = item
        if all(k in out for k in ("A", "B", "C", "D", "E")):
            return {k: out[k] for k in ("A", "B", "C", "D", "E")}

    return None

def extract_options(sample: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Try hard to find A-E options from sample.
    """
    # Common canonical: sample["mcq"]["options"]
    mcq = sample.get("mcq")
    if isinstance(mcq, dict):
        # mcq.options
        if "options" in mcq:
            norm = normalize_options_to_AE(mcq.get("options"))
            if norm:
                return norm
        # mcq has direct A-E
        if all(k in mcq for k in ("A", "B", "C", "D", "E")):
            return {k: mcq[k] for k in ("A", "B", "C", "D", "E")}
        # mcq.choices list
        for key in ("choices", "candidates", "items"):
            if key in mcq:
                norm = normalize_options_to_AE(mcq.get(key))
                if norm:
                    return norm

    # Sometimes options are placed at top level
    for key in ("options", "candidate_prompts", "candidates"):
        if key in sample:
            norm = normalize_options_to_AE(sample.get(key))
            if norm:
                return norm

    # Sometimes stored in near_miss_meta
    nmm = sample.get("near_miss_meta")
    if isinstance(nmm, dict):
        for key in ("options", "mcq_options", "choices", "candidates"):
            if key in nmm:
                norm = normalize_options_to_AE(nmm.get(key))
                if norm:
                    return norm

    return None

def extract_gold(sample: Dict[str, Any]) -> str:
    """
    Extract gold correct option letter if present.
    """
    for k in ("gold", "answer", "label", "correct"):
        v = sample.get(k)
        if isinstance(v, str) and v.strip().upper() in ("A","B","C","D","E"):
            return v.strip().upper()

    mcq = sample.get("mcq")
    if isinstance(mcq, dict):
        for k in ("correct_option", "correct", "answer", "gold"):
            v = mcq.get(k)
            if isinstance(v, str) and v.strip().upper() in ("A","B","C","D","E"):
                return v.strip().upper()
        # Sometimes numeric index 0-4
        for k in ("correct_index", "answer_index"):
            v = mcq.get(k)
            if isinstance(v, int) and 0 <= v <= 4:
                return "ABCDE"[v]

    return ""

def opt_text(opts: Dict[str, Any], letter: str) -> str:
    v = opts.get(letter)
    # Common: dict with "text"
    if isinstance(v, dict):
        return _as_text(v.get("text") or v.get("prompt") or v.get("content") or v)
    return _as_text(v)

def build_user(sample: Dict[str, Any]) -> str:
    """
    Build the user prompt. Uses available fields robustly.
    """
    context = (sample.get("specific_generated_context")
               or sample.get("generated_context")
               or sample.get("context")
               or "")
    initial = (sample.get("generated_query")
               or sample.get("query")
               or sample.get("initial_query")
               or "")

    constraints = sample.get("constraints")
    constraints_txt = ""
    if constraints:
        constraints_txt = _as_text(constraints)

    opts = extract_options(sample)
    if not opts:
        raise ValueError(f"Missing options A-E in sample keys={list(sample.keys())}")

    return f"""Context:
{context}

Initial query (old prompt):
{initial}

Candidate rewritten prompts (choose best):
A) {opt_text(opts,'A')}
B) {opt_text(opts,'B')}
C) {opt_text(opts,'C')}
D) {opt_text(opts,'D')}
E) {opt_text(opts,'E')}

Remember: Output MUST be exactly one character from {{A,B,C,D,E}}. No punctuation. No whitespace. No explanation.
"""
# Constraints (if any):
# {constraints_txt}


# -------------------------
# Response extraction
# -------------------------

def message_to_text(msg: Any) -> str:
    """
    Extract visible text from a ChatCompletionMessage-like object.
    Handles Gemini via LiteLLM where content may be null but thinking_blocks exists.
    """
    if msg is None:
        return ""

    content = getattr(msg, "content", None)
    if isinstance(content, str) and content.strip():
        return content.strip()

    tb = getattr(msg, "thinking_blocks", None)
    if tb:
        chunks: List[str] = []
        for b in tb:
            if isinstance(b, dict):
                # Some providers embed JSON string at b["thinking"]["text"]
                if "thinking" in b and isinstance(b["thinking"], dict):
                    t = b["thinking"].get("text")
                    if isinstance(t, str) and t.strip():
                        chunks.append(t.strip())
                if "text" in b and isinstance(b["text"], str) and b["text"].strip():
                    chunks.append(b["text"].strip())
        if chunks:
            return "\n".join(chunks).strip()

    return ""

def completion_to_debug_json(r: Any) -> str:
    try:
        if hasattr(r, "model_dump"):
            return json.dumps(r.model_dump(), ensure_ascii=False)
        if hasattr(r, "to_dict"):
            return json.dumps(r.to_dict(), ensure_ascii=False)
        return str(r)
    except Exception:
        return str(r)

def get_usage_tokens(r: Any) -> Tuple[int,int,int]:
    """
    returns (text_tokens, reasoning_tokens, completion_tokens) if available else (0,0,0)
    """
    try:
        usage = getattr(r, "usage", None)
        if usage is None:
            return (0,0,0)
        # openai python types: usage.completion_tokens_details.text_tokens etc
        det = getattr(usage, "completion_tokens_details", None)
        if det is None:
            # maybe dict
            if isinstance(usage, dict):
                det = usage.get("completion_tokens_details") or {}
            else:
                return (0,0, getattr(usage, "completion_tokens", 0) or 0)
        if isinstance(det, dict):
            text_t = det.get("text_tokens") or 0
            reason_t = det.get("reasoning_tokens") or 0
        else:
            text_t = getattr(det, "text_tokens", 0) or 0
            reason_t = getattr(det, "reasoning_tokens", 0) or 0
        comp_t = getattr(usage, "completion_tokens", 0) if not isinstance(usage, dict) else (usage.get("completion_tokens") or 0)
        return (int(text_t), int(reason_t), int(comp_t))
    except Exception:
        return (0,0,0)

def get_finish_reason(r: Any) -> str:
    try:
        ch = r.choices[0]
        fr = getattr(ch, "finish_reason", None)
        if isinstance(fr, str):
            return fr
    except Exception:
        pass
    return ""

# -------------------------
# Model call with retry (Gemini forced-thinking)
# -------------------------

def is_gemini_3(model_name: str) -> bool:
    m = (model_name or "").lower()
    return ("gemini-3" in m) or ("gemini3" in m)

def clamp_gemini_temperature(temp: float) -> float:
    # LiteLLM recommends temperature=1.0 for Gemini-3; keep user temp for others.
    return 1.0 if temp < 1.0 else temp

def normalize_reasoning_effort_for_gemini_3(req: str) -> str:
    # Gemini-3 pro cannot fully disable thinking; map disable/none -> low
    if not req:
        return ""
    r = req.strip().lower()
    if r in ("disable", "none", "off", "0"):
        return "low"
    return req

def chat_once(
    CLI: OpenAI,
    model: str,
    messages: List[Dict[str,str]],
    temperature: float,
    max_completion_tokens: int,
    reasoning_effort: str,
    is_official_openai: bool,
) -> Any:
    kwargs: Dict[str, Any] = dict(
        model=model,
        messages=messages,
        temperature=temperature,
    )

    # reasoning_effort is non-standard but supported by LiteLLM for several providers.
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort

    # Token arg compatibility:
    # - Official OpenAI newer models prefer max_completion_tokens.
    # - vLLM/LiteLLM proxies often accept max_tokens; some also accept max_completion_tokens.
    if is_official_openai:
        kwargs["max_completion_tokens"] = max_completion_tokens
    else:
        # Prefer max_completion_tokens when we suspect Gemini forced-thinking, else max_tokens.
        # Many proxies accept both; if one fails it will be caught by caller.
        kwargs["max_completion_tokens"] = max_completion_tokens
        kwargs["max_tokens"] = max_completion_tokens

    return CLI.chat.completions.create(**kwargs)

def call_with_retry(
    CLI: OpenAI,
    model: str,
    sys_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    reasoning_effort: str,
    is_official_openai: bool,
    max_retries: int,
    max_retry_tokens: int,
) -> Tuple[str, str]:
    """
    Returns (pred_letter, pred_raw_text_or_debug_json)
    """
    # Build messages
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # Gemini 3 quirks
    if is_gemini_3(model):
        temperature = clamp_gemini_temperature(temperature)
        reasoning_effort = normalize_reasoning_effort_for_gemini_3(reasoning_effort)

    budgets = [max_tokens]
    # exponential backoff budgets
    b = max_tokens
    for _ in range(max_retries):
        b = min(max_retry_tokens, max(b * 2, b + 1))
        if b == budgets[-1]:
            break
        budgets.append(b)

    last_debug = ""
    for bi, budget in enumerate(budgets):
        # Add a stronger suffix on retries
        time.sleep(0.5)
        if bi == 0:
            up = user_prompt
        else:
            up = user_prompt + "\n\nFINAL ANSWER: Output ONLY one letter (A/B/C/D/E)."
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": up},
            ]
        try:
            r = chat_once(
                CLI=CLI,
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=budget,
                reasoning_effort=reasoning_effort,
                is_official_openai=is_official_openai,
            )
        except Exception as e:
            last_debug = f"ERROR: request_failed: {e}"
            continue

        # Try extract visible text
        choice0 = r.choices[0] if getattr(r, "choices", None) else None
        msg = getattr(choice0, "message", None)
        text = message_to_text(msg)

        # legacy choice.text
        if (not text) and choice0 is not None and hasattr(choice0, "text"):
            t = getattr(choice0, "text", None)
            if isinstance(t, str) and t.strip():
                text = t.strip()

        pred = extract_letter(text)

        # If got a letter, return
        if pred:
            return pred, text

        # If no visible text/letter, decide whether retry
        fr = get_finish_reason(r)
        text_tokens, reasoning_tokens, completion_tokens = get_usage_tokens(r)
        last_debug = completion_to_debug_json(r)

        # Retry condition: Gemini forced-thinking ate budget (text_tokens==0) and finished by length
        if fr == "length" and text_tokens == 0 and reasoning_tokens > 0 and budget < max_retry_tokens:
            continue

        # Otherwise break (likely safety/refusal or bad output)
        break

    # Fall back: try parse letter from debug json
    pred2 = extract_letter(last_debug)
    return pred2, last_debug

# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--models", required=True, help="comma-separated model ids")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=-1)

    ap.add_argument("--reasoning_effort", type=str, default="", help="e.g., disable/low/medium/high (proxy-dependent)")

    # retry controls (important for Gemini-3)
    ap.add_argument("--max_retries", type=int, default=5)
    ap.add_argument("--max_retry_tokens", type=int, default=8192)

    # Optional overrides
    ap.add_argument("--base-url", default="", help="Override base_url for OpenAI-compatible endpoint (e.g., LiteLLM)")
    ap.add_argument("--api-key", default="", help="Override API key for base_url endpoint")
    ap.add_argument("--use-openai", action="store_true", help="Force official OpenAI endpoint (ignores --base-url)")

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    # Build client
    use_openai = True if args.use_openai else USE_OPENAI
    base_url = args.base_url.strip() or None
    api_key = args.api_key.strip() or None
    CLI, is_official_openai = make_client(use_openai=use_openai, base_url=base_url, api_key=api_key)

    # Build id -> options map once for type diagnostics
    sample_map: Dict[Any, Dict[str, Any]] = {}
    gold_map: Dict[Any, str] = {}
    with input_path.open("r", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            sid = s.get("id")
            opts = extract_options(s)
            if opts:
                sample_map[sid] = opts
            g = extract_gold(s)
            if g:
                gold_map[sid] = g

    summary_rows = []

    for model in models:
        records = []
        out_jsonl = outdir / f"pred_{model.replace('/','_')}.jsonl"

        with input_path.open("r", encoding="utf-8") as fin, out_jsonl.open("w", encoding="utf-8") as fout:
            for i, line in enumerate(tqdm(fin, desc=f"eval {model}")):
                if args.limit > 0 and i >= args.limit:
                    break
                line = line.strip()
                if not line:
                    continue
                sample = json.loads(line)
                if "mcq" not in sample:
                    continue

                sid = sample.get("id", i)
                gold = extract_gold(sample)

                pred_raw = ""
                pred = ""
                try:
                    user = build_user(sample)
                    pred, pred_raw = call_with_retry(
                        CLI=CLI,
                        model=model,
                        sys_prompt=SYSTEM,
                        user_prompt=user,
                        temperature=args.temperature,
                        max_tokens=args.max_tokens,
                        reasoning_effort=args.reasoning_effort,
                        is_official_openai=is_official_openai,
                        max_retries=args.max_retries,
                        max_retry_tokens=args.max_retry_tokens,
                    )
                except Exception as e:
                    pred_raw = f"ERROR: build_prompt: {e}"
                    pred = ""

                rec = {
                    "id": sid,
                    "gold": gold,
                    "pred": pred,
                    "correct": int(bool(gold) and pred == gold),
                    "pred_raw": pred_raw,
                }
                records.append(rec)
                fout.write(json.dumps({"id": sid, "gold": gold, "pred": pred, "pred_raw": pred_raw}, ensure_ascii=False) + "\n")

        df = pd.DataFrame(records)
        acc = float(df["correct"].mean()) if len(df) else 0.0
        pred_dist = df["pred"].value_counts(dropna=False).to_dict()

        # diagnostics by option type (if option objects have "type")
        ignore_cnt = 0
        hallu_cnt = 0
        for rec in records:
            opts = sample_map.get(rec["id"])
            if not opts or rec["pred"] not in opts:
                continue
            chosen = opts[rec["pred"]]
            t = (chosen.get("type") if isinstance(chosen, dict) else "") or ""
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
