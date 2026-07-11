from __future__ import annotations

import json
import os
from pathlib import Path

import requests
from dotenv import load_dotenv


CATAPI_BREEDS_URL = "https://api.thecatapi.com/v1/breeds"
OUTPUT_PATH = Path("data/raw/catapi_breeds.json")
TIMEOUT_SECONDS = 30


def fetch_breeds() -> list[dict] | None:
    load_dotenv()

    headers = {}
    api_key = os.getenv("CAT_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    try:
        response = requests.get(
            CATAPI_BREEDS_URL,
            headers=headers,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"Could not fetch CatAPI breeds: {exc}")
        return None

    try:
        data = response.json()
    except ValueError:
        print("Could not parse CatAPI response as JSON.")
        return None

    if not isinstance(data, list):
        print("Unexpected CatAPI response format: expected a list of breeds.")
        return None

    return data


def save_raw_breeds(breeds: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(breeds, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    breeds = fetch_breeds()
    if breeds is None:
        return 1

    save_raw_breeds(breeds)
    print(f"Saved {len(breeds)} CatAPI breeds to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
