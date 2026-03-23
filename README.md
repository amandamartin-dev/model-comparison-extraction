# model-comparison-extraction

This repo is a companion to [this blog](https://dev.to/amandamartindev/choosing-a-model-means-measuring-cost-vs-quality-on-your-data-1e58).  It is intended for education and exploration and is not considered production ready. 

A minimal Python project for comparing Baseten-hosted LLMs on structured job posting extraction.

The repository is intentionally trimmed down so someone can clone it, install dependencies, add a [Baseten](https://www.baseten.co/) API key, and run the comparison themselves without carrying around generated results or writing artifacts.

## What It Does

1. Runs a set of sample job postings through multiple configured models on Baseten.
2. Extracts structured JSON for each posting.
3. Scores each extraction against ground truth.
4. Prints summary tables for accuracy, latency, JSON validity, and cost.

## Setup

This project targets Python 3.11+.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your Baseten API key
```
## Models

The models are set in `config.py`.  To choose different models use `list_models.py` which will pull the ids and avaialble models from Baseten

## Run It

```bash
# Optional: inspect model IDs available on your Baseten account
python list_models.py

# Run the extraction comparison
python run_comparison.py

# Score the generated outputs
python score_results.py

# Print summary tables
python show_results.py
```

Generated files are written to `results/`, which is gitignored and not committed.

## Configure Models

Update `config.py` if you want to change which models are tested or adjust pricing assumptions.

## Project Structure

```text
├── README.md
├── requirements.txt
├── .env.example
├── config.py
├── list_models.py
├── run_comparison.py
├── score_results.py
├── show_results.py
└── data/
    └── job_postings.json
```
