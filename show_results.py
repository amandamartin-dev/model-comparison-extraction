"""
Display comparison results as formatted tables.
"""

import json
from pathlib import Path
from tabulate import tabulate


DISPLAY_ORDER = ["frontier", "mid", "budget"]
DISPLAY_NAMES = {
    "frontier": "Frontier",
    "mid": "Mid-Tier",
    "budget": "Budget",
}


def load_scored():
    path = Path("results/scored_results.json")
    if not path.exists():
        print("No scored results found. Run these commands first:")
        print("  python run_comparison.py")
        print("  python score_results.py")
        return None
    with open(path) as f:
        return json.load(f)


def aggregate_by_model(scored_results: list) -> dict:
    """Compute aggregate stats per model tier."""
    aggregates = {}

    for posting in scored_results:
        for tier_key, scores in posting["model_scores"].items():
            if tier_key not in aggregates:
                aggregates[tier_key] = {
                    "model": scores["model"],
                    "scores": [],
                    "latencies": [],
                    "costs": [],
                    "json_valid_count": 0,
                    "total": 0,
                    "field_scores": {},
                }
            agg = aggregates[tier_key]
            agg["scores"].append(scores["overall_score"])
            agg["latencies"].append(scores["latency_seconds"])
            agg["costs"].append(scores["cost_usd"])
            agg["total"] += 1
            if scores["json_valid"]:
                agg["json_valid_count"] += 1

            # Track per-field scores
            for field, field_data in scores.get("fields", {}).items():
                if field not in agg["field_scores"]:
                    agg["field_scores"][field] = []
                agg["field_scores"][field].append(field_data["score"])

    return aggregates


def get_present_tiers(aggregates: dict) -> list[str]:
    """Return tiers in display order, followed by any unexpected tiers."""
    ordered = [tier for tier in DISPLAY_ORDER if tier in aggregates]
    extras = sorted(tier for tier in aggregates if tier not in DISPLAY_ORDER)
    return ordered + extras


def percentile(values: list[float], pct: float) -> float:
    """Return a simple percentile using linear interpolation."""
    if not values:
        raise ValueError("percentile() requires at least one value")

    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * pct
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)

    lower_value = sorted_values[lower_index]
    upper_value = sorted_values[upper_index]
    weight = position - lower_index
    return lower_value + (upper_value - lower_value) * weight


def print_summary_table(aggregates: dict):
    """Print the main comparison table."""
    print("\n" + "=" * 80)
    print("MODEL COMPARISON: Structured Job Posting Extraction")
    print("=" * 80)

    tiers = get_present_tiers(aggregates)
    if not tiers:
        print("No aggregate data available.")
        return

    headers = ["Metric"] + [DISPLAY_NAMES.get(tier, tier.title()) for tier in tiers]
    rows = []

    # Model names
    rows.append(["Model"] + [aggregates[t]["model"] for t in tiers])

    # Accuracy
    rows.append(["Avg Accuracy"] + [
        f"{sum(aggregates[t]['scores'])/len(aggregates[t]['scores'])*100:.1f}%"
        for t in tiers
    ])

    # JSON validity rate
    rows.append(["JSON Valid Rate"] + [
        f"{aggregates[t]['json_valid_count']}/{aggregates[t]['total']} ({aggregates[t]['json_valid_count']/aggregates[t]['total']*100:.0f}%)"
        for t in tiers
    ])

    # Latency
    rows.append(["Avg Latency"] + [
        f"{sum(aggregates[t]['latencies'])/len(aggregates[t]['latencies']):.1f}s"
        for t in tiers
    ])

    rows.append(["P95 Latency"] + [
        f"{percentile(aggregates[t]['latencies'], 0.95):.1f}s"
        for t in tiers
    ])

    # Cost
    rows.append(["Avg Cost/Posting"] + [
        f"${sum(aggregates[t]['costs'])/len(aggregates[t]['costs']):.4f}"
        for t in tiers
    ])

    sample_size = max(aggregates[t]["total"] for t in tiers)
    rows.append([f"Total Cost ({sample_size})"] + [
        f"${sum(aggregates[t]['costs']):.4f}"
        for t in tiers
    ])

    # Projected cost at scale
    avg_costs = {t: sum(aggregates[t]["costs"]) / len(aggregates[t]["costs"]) for t in tiers}
    rows.append(["Est. Cost/100K Posts"] + [
        f"${avg_costs[t] * 100_000:.2f}"
        for t in tiers
    ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_field_breakdown(aggregates: dict):
    """Print per-field accuracy breakdown."""
    print("\n" + "=" * 80)
    print("PER-FIELD ACCURACY BREAKDOWN")
    print("=" * 80)

    tiers = get_present_tiers(aggregates)
    if not tiers:
        print("No aggregate data available.")
        return

    fields = ["title", "company", "location", "work_model",
              "salary_min", "salary_max", "salary_currency",
              "requirements", "nice_to_have", "benefits"]

    headers = ["Field"] + [DISPLAY_NAMES.get(tier, tier.title()) for tier in tiers]
    rows = []

    for field in fields:
        row = [field]
        for t in tiers:
            scores = aggregates[t]["field_scores"].get(field, [])
            if scores:
                avg = sum(scores) / len(scores) * 100
                row.append(f"{avg:.0f}%")
            else:
                row.append("N/A")
        rows.append(row)

    print(tabulate(rows, headers=headers, tablefmt="grid"))


def print_cost_quality_analysis(aggregates: dict):
    """Print the cost-quality tradeoff analysis."""
    print("\n" + "=" * 80)
    print("COST-QUALITY TRADEOFF ANALYSIS")
    print("=" * 80)

    tiers = get_present_tiers(aggregates)
    if not tiers:
        print("No aggregate data available.")
        return

    baseline_tier = "frontier" if "frontier" in aggregates else tiers[0]
    baseline_acc = sum(aggregates[baseline_tier]["scores"]) / len(aggregates[baseline_tier]["scores"])
    baseline_cost = sum(aggregates[baseline_tier]["costs"]) / len(aggregates[baseline_tier]["costs"])

    for t in tiers:
        acc = sum(aggregates[t]["scores"]) / len(aggregates[t]["scores"])
        cost = sum(aggregates[t]["costs"]) / len(aggregates[t]["costs"])
        acc_diff = (acc - baseline_acc) / baseline_acc * 100 if baseline_acc > 0 else 0
        cost_diff = (1 - cost / baseline_cost) * 100 if baseline_cost > 0 else 0

        print(f"\n{aggregates[t]['model']}:")
        print(f"  Accuracy: {acc*100:.1f}%  ({acc_diff:+.1f}% vs {baseline_tier})")
        print(f"  Cost/post: ${cost:.4f}  ({cost_diff:.0f}% savings vs {baseline_tier})")

        if t != baseline_tier and acc > 0 and baseline_acc > 0:
            quality_ratio = acc / baseline_acc
            cost_ratio = cost / baseline_cost if baseline_cost > 0 else 0
            print(f"  Quality retention: {quality_ratio*100:.1f}% of {baseline_tier} quality at {cost_ratio*100:.1f}% of the cost")


def print_hard_cases(scored_results: list):
    """Show which postings were hardest for each model."""
    print("\n" + "=" * 80)
    print("HARDEST POSTINGS (lowest scores per model)")
    print("=" * 80)

    if not scored_results:
        print("No scored results available.")
        return

    tiers = get_present_tiers(aggregate_by_model(scored_results))
    for t in tiers:
        posting_scores = []
        for posting in scored_results:
            if t in posting["model_scores"]:
                posting_scores.append(
                    (posting["posting_id"], posting["model_scores"][t]["overall_score"])
                )
        posting_scores.sort(key=lambda x: x[1])

        model_name = scored_results[0]["model_scores"].get(t, {}).get("model", t)
        print(f"\n{model_name} — Bottom 5:")
        for pid, score in posting_scores[:5]:
            print(f"  Posting {pid}: {score*100:.0f}%")


def main():
    scored = load_scored()
    if not scored:
        return

    aggregates = aggregate_by_model(scored)
    print_summary_table(aggregates)
    print_field_breakdown(aggregates)
    print_cost_quality_analysis(aggregates)
    print_hard_cases(scored)


if __name__ == "__main__":
    main()
