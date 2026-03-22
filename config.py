"""
Configuration for the model comparison.

After running list_models.py to see available model IDs on your Baseten account,
update the model IDs below if needed.
"""

# Models to compare
# Pricing from Baseten Model APIs as of March 2026 — verify at baseten.co/pricing
MODELS = {
    "frontier": {
        "id": "deepseek-ai/DeepSeek-V3.1",
        "label": "DeepSeek V3.1 (671B/37B active)",
        "tier": "frontier",
        "input_cost_per_m": 0.50,   # $ per 1M input tokens
        "output_cost_per_m": 1.50,  # $ per 1M output tokens
    },
    "mid": {
        "id": "nvidia/Nemotron-120B-A12B",
        "label": "Nvidia Nemotron 3 Super (120B/12B active)",
        "tier": "mid",
        "input_cost_per_m": 0.30,
        "output_cost_per_m": 0.75,
    },
    "budget": {
        "id": "openai/gpt-oss-120b",
        "label": "OpenAI GPT-OSS-120B (117B/5.1B active)",
        "tier": "budget",
        "input_cost_per_m": 0.10,
        "output_cost_per_m": 0.50,
    },
}

# System prompt for extraction
SYSTEM_PROMPT = """You are a structured data extraction system for a job board aggregator.
Given a raw job posting, extract the following fields into valid JSON.
Be precise — only extract information explicitly stated in the text.
If a field is not mentioned, use null (for strings/numbers) or an empty array (for arrays).
Do not infer or guess. Do not add information not present in the source text.

Required output schema:
{
  "title": "string — the job title",
  "company": "string or null — company name",
  "location": "string or null — location as stated",
  "work_model": "string or null — one of: Remote, Hybrid, On-site",
  "salary_min": "integer or null — minimum annual salary (convert hourly/monthly if needed)",
  "salary_max": "integer or null — maximum annual salary",
  "salary_currency": "string or null — ISO currency code",
  "requirements": ["array of strings — hard requirements"],
  "nice_to_have": ["array of strings — preferred/bonus qualifications"],
  "benefits": ["array of strings — benefits and perks"]
}

Return ONLY valid JSON. No markdown, no explanation, no code fences."""

# User prompt template
USER_PROMPT_TEMPLATE = """Extract structured data from the following job posting:

---
{raw_text}
---

Return only valid JSON matching the required schema."""
