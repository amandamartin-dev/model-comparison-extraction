"""
Score extraction results against ground truth.
Produces scored_results.json with per-field and aggregate accuracy.
"""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict, Union


ScalarValue = Optional[Union[str, int, float]]
NormalizedValue = Optional[Union[str, int, float, List["NormalizedValue"]]]
FieldName = str
ScalarScoreDict = Dict[str, Union[float, str]]
ArrayScoreDict = Dict[str, Union[float, int, str]]
FieldScoreDict = Dict[FieldName, Dict[str, Union[float, int, str]]]


class ExtractionData(TypedDict, total=False):
    title: ScalarValue
    company: ScalarValue
    location: ScalarValue
    work_model: ScalarValue
    salary_min: ScalarValue
    salary_max: ScalarValue
    salary_currency: ScalarValue
    requirements: list[ScalarValue]
    nice_to_have: list[ScalarValue]
    benefits: list[ScalarValue]


class ModelResult(TypedDict):
    model: str
    json_valid: bool
    latency_seconds: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    extracted: Optional[ExtractionData]


class PostingResult(TypedDict):
    posting_id: int
    ground_truth: ExtractionData
    model_results: dict[str, ModelResult]


@dataclass(frozen=True)
class ScalarScore:
    score: float
    status: str


@dataclass(frozen=True)
class ArrayScore:
    score: float
    matched: int
    total: int
    extra: int
    status: str


@dataclass(frozen=True)
class ExtractionScore:
    overall_score: float
    fields: FieldScoreDict
    error: Optional[str] = None


SCALAR_FIELDS = (
    "title",
    "company",
    "location",
    "work_model",
    "salary_min",
    "salary_max",
    "salary_currency",
)

ARRAY_FIELDS = ("requirements", "nice_to_have", "benefits")

WEIGHTS: Dict[FieldName, float] = {
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

RESULTS_PATH = Path("results/extraction_results.json")
OUTPUT_PATH = Path("results/scored_results.json")


def _normalized_sort_key(value: NormalizedValue) -> tuple[str, str]:
    """Provide a stable cross-type key so mixed-type arrays do not fail to sort."""
    return (type(value).__name__, repr(value))


def normalize(value: object) -> NormalizedValue:
    """Normalize a value for comparison."""
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower()
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, list):
        normalized_items = [normalize(item) for item in value]
        return sorted(normalized_items, key=_normalized_sort_key)
    return None


def score_scalar(extracted_val: object, truth_val: object) -> ScalarScoreDict:
    """Score a scalar field (string, number, or null)."""
    e = normalize(extracted_val)
    t = normalize(truth_val)

    if t is None and e is None:
        return asdict(ScalarScore(score=1.0, status="correct_null"))
    if t is None and e is not None:
        return asdict(ScalarScore(score=0.0, status="hallucinated"))
    if t is not None and e is None:
        return asdict(ScalarScore(score=0.0, status="missed"))

    if isinstance(t, (int, float)) and isinstance(e, (int, float)):
        if t == 0:
            return asdict(ScalarScore(score=1.0 if e == 0 else 0.0, status="correct" if e == 0 else "wrong"))
        ratio = abs(e - t) / abs(t)
        if ratio <= 0.05:
            return asdict(ScalarScore(score=1.0, status="correct"))
        if ratio <= 0.15:
            return asdict(ScalarScore(score=0.5, status="close"))
        return asdict(ScalarScore(score=0.0, status="wrong"))

    if isinstance(t, str) and isinstance(e, str):
        if e == t:
            return asdict(ScalarScore(score=1.0, status="correct"))
        return asdict(ScalarScore(score=0.0, status="wrong"))

    return asdict(ScalarScore(score=0.0, status="type_mismatch"))


def score_array(extracted_list: object, truth_list: object) -> ArrayScoreDict:
    """Score an array field using exact normalized matches and F1."""
    if not isinstance(extracted_list, list):
        extracted_list = []
    if not isinstance(truth_list, list):
        truth_list = []

    if len(truth_list) == 0 and len(extracted_list) == 0:
        return asdict(ArrayScore(score=1.0, matched=0, total=0, extra=0, status="correct_empty"))

    if len(truth_list) == 0 and len(extracted_list) > 0:
        return asdict(
            ArrayScore(
                score=0.0,
                matched=0,
                total=0,
                extra=len(extracted_list),
                status="hallucinated",
            )
        )

    matched = 0
    truth_normalized = [normalize(item) for item in truth_list]
    extracted_normalized = [normalize(item) for item in extracted_list]

    used_indexes: set[int] = set()
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

    recall = matched / len(truth_list) if truth_list else 1.0
    extra = max(0, len(extracted_list) - len(truth_list))
    precision = matched / len(extracted_list) if extracted_list else 1.0
    if precision + recall == 0:
        score = 0.0
    else:
        score = (2 * precision * recall) / (precision + recall)

    return asdict(
        ArrayScore(
            score=round(score, 2),
            matched=matched,
            total=len(truth_list),
            extra=extra,
            status="full_match" if score == 1.0 else "partial_match" if score > 0 else "no_match",
        )
    )


def score_extraction(extracted: Optional[ExtractionData], ground_truth: ExtractionData) -> Dict[str, object]:
    """Score a single extraction against ground truth."""
    if extracted is None:
        return asdict(ExtractionScore(overall_score=0.0, fields={}, error="No valid JSON extracted"))

    field_scores: FieldScoreDict = {}
    for field in SCALAR_FIELDS:
        field_scores[field] = score_scalar(extracted.get(field), ground_truth.get(field))

    for field in ARRAY_FIELDS:
        field_scores[field] = score_array(
            extracted.get(field, []),
            ground_truth.get(field, []),
        )

    weighted_sum = sum(float(field_scores[field]["score"]) * WEIGHTS[field] for field in field_scores)
    max_possible = sum(WEIGHTS.values())
    overall = round(weighted_sum / max_possible, 3)

    return asdict(ExtractionScore(overall_score=overall, fields=field_scores))


def parse_args() -> argparse.Namespace:
    """Parse optional CLI paths while preserving default behavior."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=RESULTS_PATH,
        help=f"Path to extraction results JSON. Default: {RESULTS_PATH}",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Path to scored results JSON. Default: {OUTPUT_PATH}",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = args.input
    output_path = args.output

    if not results_path.exists():
        print("No results found. Run 'python run_comparison.py' first.")
        return

    with results_path.open(encoding="utf-8") as file:
        results: List[PostingResult] = json.load(file)

    scored: List[Dict[str, object]] = []

    for posting_result in results:
        posting_scored: Dict[str, object] = {
            "posting_id": posting_result["posting_id"],
            "model_scores": {},
        }
        model_scores = posting_scored["model_scores"]
        if not isinstance(model_scores, dict):
            raise TypeError("model_scores must be a dictionary")

        for tier_key, model_result in posting_result["model_results"].items():
            extraction_score = score_extraction(
                model_result.get("extracted"),
                posting_result["ground_truth"],
            )
            model_scores[tier_key] = {
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

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(scored, file, indent=2)

    print(f"Scored results saved to {output_path}")
    print("Run 'python show_results.py' to see the comparison table.")


if __name__ == "__main__":
    main()
