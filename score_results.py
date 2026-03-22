"""
Score extraction results against ground truth.
Produces scored_results.json with per-field and aggregate accuracy.
"""

import json
from pathlib import Path


def normalize(value):
    """Normalize a value for comparison."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        return sorted([normalize(v) for v in value])
    return value


def score_scalar(extracted_val, truth_val) -> dict:
    """Score a scalar field (string, number, or null)."""
    e = normalize(extracted_val)
    t = normalize(truth_val)

    if t is None and e is None:
        return {"score": 1.0, "status": "correct_null"}
    if t is None and e is not None:
        return {"score": 0.0, "status": "hallucinated"}  # made up data
    if t is not None and e is None:
        return {"score": 0.0, "status": "missed"}

    if isinstance(t, (int, float)) and isinstance(e, (int, float)):
        if t == 0:
            return {"score": 1.0 if e == 0 else 0.0, "status": "correct" if e == 0 else "wrong"}
        ratio = abs(e - t) / abs(t)
        if ratio <= 0.05:
            return {"score": 1.0, "status": "correct"}
        if ratio <= 0.15:
            return {"score": 0.5, "status": "close"}
        return {"score": 0.0, "status": "wrong"}

    if isinstance(t, str) and isinstance(e, str):
        if e == t:
            return {"score": 1.0, "status": "correct"}
        if t in e or e in t:
            return {"score": 0.75, "status": "partial"}
        return {"score": 0.0, "status": "wrong"}

    return {"score": 0.0, "status": "type_mismatch"}


def score_array(extracted_list, truth_list) -> dict:
    """Score an array field using set overlap."""
    if not isinstance(extracted_list, list):
        extracted_list = []
    if not isinstance(truth_list, list):
        truth_list = []

    if len(truth_list) == 0 and len(extracted_list) == 0:
        return {"score": 1.0, "matched": 0, "total": 0, "extra": 0, "status": "correct_empty"}

    if len(truth_list) == 0 and len(extracted_list) > 0:
        return {"score": 0.0, "matched": 0, "total": 0, "extra": len(extracted_list), "status": "hallucinated"}

    matched = 0
    truth_normalized = [normalize(t) for t in truth_list]
    extracted_normalized = [normalize(e) for e in extracted_list]

    used_indexes = set()
    for t_item in truth_normalized:
        for index, e_item in enumerate(extracted_normalized):
            if index in used_indexes:
                continue
            if t_item is None or e_item is None:
                continue
            if t_item == e_item:
                matched += 1
                used_indexes.add(index)
                break
            if isinstance(t_item, str) and isinstance(e_item, str):
                t_words = set(t_item.split())
                e_words = set(e_item.split())
                overlap = len(t_words & e_words) / max(len(t_words), 1)
                if overlap >= 0.5:
                    matched += 1
                    used_indexes.add(index)
                    break

    recall = matched / len(truth_list) if truth_list else 1.0
    extra = max(0, len(extracted_list) - len(truth_list))
    precision = matched / len(extracted_list) if extracted_list else 1.0
    score = min(recall, precision)

    return {
        "score": round(score, 2),
        "matched": matched,
        "total": len(truth_list),
        "extra": extra,
        "status": "full_match" if score == 1.0 else "partial_match" if score > 0 else "no_match",
    }


def score_extraction(extracted: dict, ground_truth: dict) -> dict:
    """Score a single extraction against ground truth."""
    if extracted is None:
        return {
            "overall_score": 0.0,
            "fields": {},
            "error": "No valid JSON extracted",
        }

    scalar_fields = ["title", "company", "location", "work_model", "salary_min", "salary_max", "salary_currency"]
    array_fields = ["requirements", "nice_to_have", "benefits"]

    field_scores = {}
    for field in scalar_fields:
        field_scores[field] = score_scalar(extracted.get(field), ground_truth.get(field))

    for field in array_fields:
        field_scores[field] = score_array(
            extracted.get(field, []),
            ground_truth.get(field, []),
        )

    weights = {
        "title": 2.0,
        "company": 1.5,
        "location": 1.5,
        "work_model": 1.0,
        "salary_min": 1.5,
        "salary_max": 1.5,
        "salary_currency": 1.0,
        "requirements": 2.0,
        "nice_to_have": 1.0,
        "benefits": 1.0,
    }

    weighted_sum = sum(field_scores[f]["score"] * weights[f] for f in field_scores)
    max_possible = sum(weights.values())
    overall = round(weighted_sum / max_possible, 3)

    return {
        "overall_score": overall,
        "fields": field_scores,
    }


def main():
    results_path = Path("results/extraction_results.json")
    if not results_path.exists():
        print("No results found. Run 'python run_comparison.py' first.")
        return

    with open(results_path) as f:
        results = json.load(f)

    scored = []

    for posting_result in results:
        posting_scored = {
            "posting_id": posting_result["posting_id"],
            "model_scores": {},
        }

        for tier_key, model_result in posting_result["model_results"].items():
            extraction_score = score_extraction(
                model_result.get("extracted"),
                posting_result["ground_truth"],
            )
            posting_scored["model_scores"][tier_key] = {
                "model": model_result["model"],
                "tier": tier_key,
                "json_valid": model_result["json_valid"],
                "latency_seconds": model_result["latency_seconds"],
                "cost_usd": model_result["cost_usd"],
                "input_tokens": model_result["input_tokens"],
                "output_tokens": model_result["output_tokens"],
                **extraction_score,
            }

        scored.append(posting_scored)

    output_path = Path("results/scored_results.json")
    with open(output_path, "w") as f:
        json.dump(scored, f, indent=2)

    print(f"Scored results saved to {output_path}")
    print(f"Run 'python show_results.py' to see the comparison table.")


if __name__ == "__main__":
    main()
