from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cat_breed_assistant.evaluation.retrieval.io import read_jsonl


def write_jsonl_atomic(records: list[dict], path: Path) -> None:
    temp_path = path.with_name(f".{path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as output_file:
            for record in records:
                output_file.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temp_path, path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review retrieval evaluation candidates.")
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("data/evaluation/retrieval/v1/pilot_candidates.jsonl"),
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=Path("data/processed/knowledge_chunks.jsonl"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = read_jsonl(args.candidates)
    chunks = {record["chunk_id"]: record for record in read_jsonl(args.chunks)}
    changed = False

    for index, candidate in enumerate(candidates, start=1):
        if candidate.get("status") != "pending_review":
            continue
        chunk = chunks.get(candidate["chunk_id"], {})
        print("\n" + "=" * 80)
        print(f"{index}/{len(candidates)}  {candidate['query_id']}")
        print(f"Status: {candidate['status']}")
        print(f"Generator: {candidate['generator_provider']}:{candidate['generator_model']}")
        print(f"Validator: {candidate['validator_provider']}:{candidate['validator_model']}")
        print(f"Query: {candidate['query']}")
        print(f"Answer: {candidate['answer']}")
        print(f"Evidence: {candidate['evidence_quote']}")
        print(f"Validation: {json.dumps(candidate['validation'], ensure_ascii=False)}")
        print("\nSource chunk:")
        print(chunk.get("text", "")[:2500])
        decision = input("\na approve / r reject / s skip / q quit: ").strip().casefold()
        if decision == "q":
            break
        if decision == "s":
            continue
        if decision == "a":
            candidate["status"] = "approved"
            changed = True
        elif decision == "r":
            candidate["status"] = "rejected"
            changed = True
        else:
            print("Unknown command, skipped.")

        if changed:
            write_jsonl_atomic(candidates, args.candidates)

    if changed:
        write_jsonl_atomic(candidates, args.candidates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
