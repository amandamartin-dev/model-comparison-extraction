"""
List all available models on your Baseten account.
Run this first to confirm your API key works and see exact model IDs.
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    base_url="https://inference.baseten.co/v1",
    api_key=os.environ["BASETEN_API_KEY"],
)


def main():
    print("Querying Baseten for available models...\n")

    models = client.models.list()

    for model in sorted(models.data, key=lambda m: m.id):
        print(f"  {model.id}")

    print(f"\nTotal: {len(models.data)} models available")
    print("\nUse these model IDs in config.py to configure which models to compare.")


if __name__ == "__main__":
    main()
