"""
Main script: run structured extraction across all models for all job postings.
Saves raw results to results/extraction_results.json
"""

import json
import os
import time
import re
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from config import MODELS, SYSTEM_PROMPT, USER_PROMPT_TEMPLATE

load_dotenv()

client = OpenAI(
    base_url="https://inference.baseten.co/v1",
    api_key=os.environ["BASETEN_API_KEY"],
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


THINKING_BLOCK_PATTERN = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def serialize_message_content(content: Any) -> str:
    """Normalize message content into a single string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part for part in parts if part)
    return str(content)


def strip_reasoning_preamble(text: str) -> str:
    """Remove common reasoning-model preambles before JSON output."""
    if "</think>" in text:
        return text.rsplit("</think>", 1)[-1].strip()

    return THINKING_BLOCK_PATTERN.sub("", text).strip()


def extract_first_json_object(text: str) -> dict | None:
    """Decode the first valid JSON object found in a string."""
    decoder = json.JSONDecoder()

    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def extract_json_from_response(text: str) -> dict | None:
    """Attempt to parse JSON from model response, handling common issues."""
    text = strip_reasoning_preamble(text)

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    cleaned = re.sub(r"```(?:json)?\s*", "", text)
    cleaned = cleaned.strip().rstrip("`")
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    return extract_first_json_object(cleaned)


def run_extraction(model_config: dict, raw_text: str) -> dict:
    """Run a single extraction and return result with metadata."""
    model_id = model_config["id"]
    user_prompt = USER_PROMPT_TEMPLATE.format(raw_text=raw_text)

    start_time = time.perf_counter()
    error = None
    extracted = None
    json_valid = False
    raw_response = ""
    input_tokens = 0
    output_tokens = 0

    try:
        response = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=2000,
        )
        raw_response = serialize_message_content(response.choices[0].message.content)
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0

        extracted = extract_json_from_response(raw_response)
        json_valid = extracted is not None

    except Exception as e:
        error = str(e)

    elapsed = time.perf_counter() - start_time

    cost = (
        (input_tokens / 1_000_000) * model_config["input_cost_per_m"]
        + (output_tokens / 1_000_000) * model_config["output_cost_per_m"]
    )

    return {
        "model": model_config["label"],
        "model_id": model_id,
        "tier": model_config["tier"],
        "raw_response": raw_response,
        "extracted": extracted,
        "json_valid": json_valid,
        "latency_seconds": round(elapsed, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(cost, 6),
        "error": error,
    }


def main():
    with open("data/job_postings.json", encoding="utf-8") as f:
        postings = json.load(f)

    print(f"Loaded {len(postings)} job postings")
    print(f"Testing {len(MODELS)} models: {', '.join(m['label'] for m in MODELS.values())}")
    print("=" * 70)

    all_results = []

    for posting in postings:
        posting_id = posting["id"]
        raw_text = posting["raw_text"]
        print(f"\n--- Posting {posting_id}/{len(postings)} ---")
        print(f"    Preview: {raw_text[:80]}...")

        posting_results = {
            "posting_id": posting_id,
            "raw_text": raw_text,
            "ground_truth": posting["ground_truth"],
            "model_results": {},
        }

        for tier_key, model_config in MODELS.items():
            print(f"    [{model_config['label']}] ", end="", flush=True)
            result = run_extraction(model_config, raw_text)
            posting_results["model_results"][tier_key] = result

            status = "OK" if result["json_valid"] else "FAIL"
            if result["error"]:
                status = f"ERROR: {result['error'][:50]}"
            print(f"{status} ({result['latency_seconds']}s, ${result['cost_usd']:.4f})")

        all_results.append(posting_results)

    output_path = RESULTS_DIR / "extraction_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"Done! Results saved to {output_path}")
    print(f"Run 'python score_results.py' to evaluate accuracy.")


if __name__ == "__main__":
    main()
