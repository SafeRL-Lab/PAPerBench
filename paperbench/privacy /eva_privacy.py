#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import argparse
from collections import defaultdict
from typing import Dict, Any, List, Optional

from openai import OpenAI  # OpenAI SDK works for OpenAI + OpenAI-compatible endpoints (vLLM, LiteLLM Proxy)

CHOICES = ["A", "B", "C", "D", "E"]

SYSTEM_PROMPT = (
    "You are a strict multiple-choice solver. "
    "Use ONLY the provided text as evidence. "
    "Return ONLY a single letter among A, B, C, D, E. "
    "No extra words."
)

# -------------------------
# parsing model output
# -------------------------

def extract_label(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    t = text.strip().upper()
    if t in CHOICES:
        return t
    m = re.search(r"\b([A-E])\b", t)
    return m.group(1) if m else None

# -------------------------
# prompt formatting
# -------------------------

def format_mcq_with_context(mcq: Dict[str, Any], specific_context: str) -> str:
    q = mcq["question"]
    opts = mcq["options"]

    lines = []
    lines.append("You are given a text that may contain sensitive information.\n")
    lines.append("Text:")
    lines.append(specific_context)
    lines.append("\nQuestion:")
    lines.append(q)
    lines.append("\nOptions:")
    for k in CHOICES:
        lines.append(f"{k}. {opts[k]}")
    lines.append("\nPlease answer with a single letter (A, B, C, D, or E) only.")
    return "\n".join(lines)

# -------------------------
# LiteLLM Proxy helpers
# -------------------------

def _normalize_openai_compatible_base_url(url: str) -> str:
    """
    Ensure base_url ends with /v1 for OpenAI-compatible endpoints.
    Example:
      https://xxx.run.app -> https://xxx.run.app/v1
      http://localhost:8002/v1 -> unchanged
    """
    url = (url or "").strip()
    if not url:
        return url
    if url.endswith("/v1"):
        return url
    return url.rstrip("/") + "/v1"

def _should_use_litellm_proxy() -> bool:
    """
    Backward-compatible auto switch:
    - If user sets USE_OPENAI=0 and provides LITELLM_BASE_URL, we route OpenAI SDK to the proxy.
    """
    use_openai = os.environ.get("USE_OPENAI", "").strip()
    litellm_base = os.environ.get("LITELLM_BASE_URL", "").strip()
    return (use_openai == "0") and bool(litellm_base)

def _resolve_litellm_proxy_config() -> (str, str):
    """
    Read proxy base and key from env.
    """
    base = _normalize_openai_compatible_base_url(os.environ.get("LITELLM_BASE_URL", ""))
    key = os.environ.get("LITELLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or "EMPTY"
    return base, key

# -------------------------
# client & inference
# -------------------------

def make_client(provider: str, api_key: str, base_url: Optional[str]) -> OpenAI:
    """
    provider:
      - openai: default OpenAI, BUT if USE_OPENAI=0 and LITELLM_BASE_URL exists -> route to LiteLLM Proxy (OpenAI-compatible)
      - litellm_proxy: explicit LiteLLM proxy mode (reads LITELLM_BASE_URL/LITELLM_API_KEY)
      - vllm/openai_compatible: user supplies base_url like http://localhost:8000/v1
    """
    if provider == "litellm_proxy":
        proxy_base, proxy_key = _resolve_litellm_proxy_config()
        if not proxy_base:
            raise ValueError("LITELLM_BASE_URL is required for provider=litellm_proxy")
        return OpenAI(api_key=proxy_key, base_url=proxy_base)

    if provider == "openai":
        if _should_use_litellm_proxy():
            proxy_base, proxy_key = _resolve_litellm_proxy_config()
            return OpenAI(api_key=proxy_key, base_url=proxy_base)
        return OpenAI(api_key=api_key)

    # vllm / openai_compatible
    if not base_url:
        raise ValueError("For vllm/openai_compatible, --base-url is required (e.g., http://localhost:8000/v1).")
    return OpenAI(api_key=api_key, base_url=base_url)

def call_chat(
    client: OpenAI,
    model: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
) -> str:
    if model.startswith("gpt-5"):
        # print("-------------------testing for gpt-5 models-------------------")
        resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_completion_tokens=max_tokens, # todo: revised to max_completion_tokens for gpt-5 compatibility
        timeout=timeout_s,
    )
    else:
        # print("-------------------testing for other models-------------------")
        resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout_s,
    )
    
    
    return resp.choices[0].message.content or ""

# -------------------------
# evaluation
# -------------------------

def eval_jsonl(
    input_path: str,
    output_path: str,
    provider: str,
    model: str,
    api_key: str,
    base_url: Optional[str],
    context_field: str,
    mcq_fields: List[str],
    out_field: str,
    temperature: float,
    max_tokens: int,
    sleep_s: float,
    max_samples: Optional[int],
    timeout_s: float,
) -> None:
    client = make_client(provider, api_key, base_url)

    total_q = 0
    correct_q = 0
    by_type_total = defaultdict(int)
    by_type_correct = defaultdict(int)

    # Per-mcq-field aggregate stats (used when evaluating multiple mcq fields)
    field_total_q = defaultdict(int)
    field_correct_q = defaultdict(int)
    field_by_type_total = defaultdict(lambda: defaultdict(int))
    field_by_type_correct = defaultdict(lambda: defaultdict(int))

    n_lines = 0
    n_err = 0

    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin):
            if max_samples is not None and n_lines >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            data = json.loads(line)
            n_lines += 1

            specific_context = data.get(context_field, "")

            # Evaluate one or multiple MCQ fields
            preds_all = []  # concatenated across fields (each record includes mcq_field)
            preds_by_field = {}  # field -> list[pred_record]
            for mcq_field in mcq_fields:
                mcqs: List[Dict[str, Any]] = data.get(mcq_field, [])
                preds: List[Dict[str, Any]] = []
                for j, mcq in enumerate(mcqs):
                    # Special typing rule: privacy_aggregate_mcq uses pii_type as mcq_type
                    if mcq_field == "privacy_aggregate_mcq":
                        mcq_type = mcq.get("mcq_type", f"mcq_{j}")
                    else:
                        mcq_type = mcq.get("pii_type", f"mcq_{j}")

                    gold = mcq.get("answer")
                    user_prompt = format_mcq_with_context(mcq, specific_context)

                    try:
                        pred_raw = call_chat(
                            client=client,
                            model=model,
                            user_prompt=user_prompt,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            timeout_s=timeout_s,
                        )
                        pred = extract_label(pred_raw)
                    except Exception as e:
                        pred_raw = f"ERROR: {type(e).__name__}: {e}"
                        pred = None
                        n_err += 1

                    is_correct = (pred == gold) if (gold is not None) else False

                    rec = {
                        "mcq_field": mcq_field,
                        "mcq_type": mcq_type,
                        "pred_raw": pred_raw,
                        "pred": pred,
                        "gold": gold,
                        "correct": is_correct,
                    }
                    preds.append(rec)
                    preds_all.append(rec)

                    if gold is not None:
                        total_q += 1
                        by_type_total[mcq_type] += 1
                        field_total_q[mcq_field] += 1
                        field_by_type_total[mcq_field][mcq_type] += 1
                        if is_correct:
                            correct_q += 1
                            by_type_correct[mcq_type] += 1
                            field_correct_q[mcq_field] += 1
                            field_by_type_correct[mcq_field][mcq_type] += 1

                    if sleep_s > 0:
                        time.sleep(sleep_s)

                preds_by_field[mcq_field] = preds

            # Keep benchmark structure backward-compatible:
            # - If only one mcq_field is evaluated, preserve the original schema exactly.
            # - If multiple mcq_fields are evaluated, keep the original top-level keys and add extra fields.
            if len(mcq_fields) == 1:
                data[out_field] = {
                    "provider": provider,
                    "model": model,
                    "context_field": context_field,
                    "mcq_field": mcq_fields[0],
                    "preds": preds_by_field.get(mcq_fields[0], []),
                }
            else:
                data[out_field] = {
                    "provider": provider,
                    "model": model,
                    "context_field": context_field,
                    "mcq_field": mcq_fields[0],  # backward-compat: first field
                    "mcq_fields": mcq_fields,
                    "preds": preds_all,
                    "preds_by_field": preds_by_field,
                }
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Summary
    print("\n===== SUMMARY =====")
    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    if provider == "openai" and _should_use_litellm_proxy():
        proxy_base, _ = _resolve_litellm_proxy_config()
        print(f"provider={provider} (routed_to_litellm_proxy={proxy_base}) model={model}")
    else:
        print(f"provider={provider} model={model}")
    print(f"lines_processed={n_lines} api_errors={n_err}")
    if total_q > 0:
        print(f"total_questions={total_q} correct={correct_q} acc={correct_q/total_q:.4f}")
    else:
        print("No questions found (check mcq_fields).")

    if len(mcq_fields) > 1:
        print("\n===== SUMMARY BY MCQ FIELD =====")
        for f in mcq_fields:
            ft = field_total_q.get(f, 0)
            fc = field_correct_q.get(f, 0)
            facc = (fc / ft) if ft else 0.0
            print(f"{f:28s}  {fc:5d}/{ft:5d}  acc={facc:.4f}")

    print("\n===== BY MCQ TYPE =====")
    for t in sorted(by_type_total.keys()):
        tot = by_type_total[t]
        cor = by_type_correct[t]
        acc = cor / tot if tot else 0.0
        print(f"{t:28s}  {cor:5d}/{tot:5d}  acc={acc:.4f}")

    if len(mcq_fields) > 1:
        print("\n===== BY MCQ TYPE (PER FIELD) =====")
        for f in mcq_fields:
            print(f"\n-- {f} --")
            for t in sorted(field_by_type_total[f].keys()):
                tot = field_by_type_total[f][t]
                cor = field_by_type_correct[f][t]
                acc = cor / tot if tot else 0.0
                print(f"{t:28s}  {cor:5d}/{tot:5d}  acc={acc:.4f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="jsonl containing specific_generated_context + MCQs")
    ap.add_argument("--output", required=True, help="output jsonl with predictions")

    ap.add_argument(
        "--provider",
        required=True,
        choices=["openai", "vllm", "openai_compatible", "litellm_proxy"]
    )
    ap.add_argument("--model", required=True)

    ap.add_argument("--api-key", default="", help="API key (or env OPENAI_API_KEY / VLLM_API_KEY). "
                                                 "If USE_OPENAI=0 and LITELLM_BASE_URL is set, uses LITELLM_API_KEY automatically.")
    ap.add_argument("--base-url", default="", help="For vllm/openai_compatible, e.g. http://localhost:8000/v1")

    ap.add_argument("--context-field", default="specific_generated_context")
    ap.add_argument(
        "--mcq-field",
        action="append",
        default=[],
        help="MCQ field(s) in json. Repeatable: --mcq-field a --mcq-field b, or comma-separated: a,b",
    )
    ap.add_argument("--out-field", default="privacy_aggregate_mcq_eval")

    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--timeout", type=float, default=60.0)

    args = ap.parse_args()

    # Normalize mcq_fields: allow repeated flags or comma-separated values
    if not args.mcq_field:
        mcq_fields = ["privacy_aggregate_mcq"]
    else:
        mcq_fields = []
        for x in args.mcq_field:
            mcq_fields.extend([t.strip() for t in x.split(",") if t.strip()])

    # resolve API key (keeps old behavior; explicit proxy reads LITELLM_API_KEY)
    api_key = args.api_key
    if not api_key:
        if args.provider == "litellm_proxy":
            api_key = os.environ.get("LITELLM_API_KEY", "") or "EMPTY"
        elif args.provider == "openai":
            if _should_use_litellm_proxy():
                _, proxy_key = _resolve_litellm_proxy_config()
                api_key = proxy_key
            else:
                api_key = os.environ.get("OPENAI_API_KEY", "")
        else:
            api_key = os.environ.get("VLLM_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "") or "EMPTY"

    base_url = args.base_url.strip() or None
    if base_url:
        base_url = _normalize_openai_compatible_base_url(base_url)

    if args.provider in ("vllm", "openai_compatible") and not base_url:
        raise SystemExit("ERROR: For vllm/openai_compatible, --base-url is required (e.g., http://localhost:8000/v1).")

    if args.provider == "openai" and not api_key and not _should_use_litellm_proxy():
        raise SystemExit("ERROR: API key missing for provider=openai. Set --api-key or OPENAI_API_KEY "
                         "(or set USE_OPENAI=0 + LITELLM_BASE_URL + LITELLM_API_KEY for proxy).")

    eval_jsonl(
        input_path=args.input,
        output_path=args.output,
        provider=args.provider,
        model=args.model,
        api_key=api_key,
        base_url=base_url,
        context_field=args.context_field,
        mcq_fields=mcq_fields,
        out_field=args.out_field,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        sleep_s=args.sleep,
        max_samples=args.max_samples,
        timeout_s=args.timeout,
    )

if __name__ == "__main__":
    main()
