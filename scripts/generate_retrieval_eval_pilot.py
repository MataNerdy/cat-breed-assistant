from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cat_breed_assistant.evaluation.retrieval.generators import EvaluationProviderError
from src.cat_breed_assistant.evaluation.retrieval.pipeline import (
    apply_cli_overrides,
    dry_run,
    dry_run_unused_breeds,
    load_config,
    run_pipeline,
    run_unused_breeds_pipeline,
)


def parse_breed_ids(value: str | None) -> set[str] | None:
    if value is None:
        return None
    breed_ids = {item.strip() for item in value.split(",") if item.strip()}
    if not breed_ids:
        raise ValueError("--breed-ids must contain at least one breed id")
    return breed_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pilot retrieval evaluation candidates with Gemini/Mistral."
    )
    parser.add_argument("--config", type=Path, default=Path("configs/retrieval_eval_pilot.yaml"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--target-count", type=int)
    parser.add_argument("--breed-limit", type=int)
    parser.add_argument("--query-language")
    parser.add_argument("--answer-language")
    parser.add_argument("--unused-breeds", action="store_true")
    parser.add_argument(
        "--breed-ids",
        help="Comma-separated CatAPI breed ids to process in unused-breeds mode.",
    )
    parser.add_argument(
        "--allow-reused-chunks",
        action="store_true",
        help="Allow generating additional distinct questions from chunks that already have candidates.",
    )
    parser.add_argument("--questions-per-breed", type=int, default=3)
    parser.add_argument("--max-new-candidates", type=int)
    parser.add_argument(
        "--api-call-delay-seconds",
        type=float,
        default=0.0,
        help="Optional delay before each LLM API call, useful for free-tier rate limits.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        config = apply_cli_overrides(
            load_config(args.config),
            seed=args.seed,
            target_count=args.target_count,
            breed_limit=args.breed_limit,
            query_language=args.query_language,
            answer_language=args.answer_language,
        )
        breed_ids = parse_breed_ids(args.breed_ids)
        if args.dry_run:
            if args.unused_breeds:
                result = dry_run_unused_breeds(
                    config,
                    questions_per_breed=args.questions_per_breed,
                    resume=args.resume,
                    breed_ids=breed_ids,
                    allow_reused_chunks=args.allow_reused_chunks,
                )
            else:
                result = dry_run(config)
            print(json.dumps(result.model_dump(), ensure_ascii=False, indent=2))
            return 0

        if args.unused_breeds:
            manifest = run_unused_breeds_pipeline(
                config,
                questions_per_breed=args.questions_per_breed,
                resume=args.resume,
                api_call_delay_seconds=args.api_call_delay_seconds,
                max_new_candidates=args.max_new_candidates,
                breed_ids=breed_ids,
                allow_reused_chunks=args.allow_reused_chunks,
            )
        else:
            manifest = run_pipeline(
                config,
                resume=args.resume,
                api_call_delay_seconds=args.api_call_delay_seconds,
            )
        print(json.dumps(manifest.model_dump(mode="json"), ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, EvaluationProviderError) as exc:
        print(f"Could not generate retrieval evaluation pilot: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
