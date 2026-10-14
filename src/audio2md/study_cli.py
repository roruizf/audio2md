from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .evidence import load_segments, prepare_evidence, validate_citations
from .exceptions import Audio2MdError
from .legacy import migrate_legacy_bundle
from .opencode_adapter import DEFAULT_MODEL as OPENCODE_MODEL, OpenCodeAdapter
from .openrouter import DEFAULT_MODEL as OPENROUTER_MODEL, OpenRouterAdapter
from .study import extract_and_render


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio2md-study", description="Prepare and validate grounded study artifacts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Build deterministic evidence packets")
    prepare.add_argument("bundle", type=Path)
    prepare.add_argument("--max-words", type=int, default=1200)
    prepare.add_argument("--overlap-segments", type=int, default=2)
    prepare.add_argument("--force", action="store_true")

    validate = subparsers.add_parser("validate", help="Check segment citations in Markdown")
    validate.add_argument("bundle", type=Path)
    validate.add_argument("artifact", type=Path)
    validate.add_argument("--allow-uncited", action="store_true")

    migrate = subparsers.add_parser("migrate-legacy", help="Convert a legacy bundle without ASR")
    migrate.add_argument("legacy_bundle", type=Path)
    migrate.add_argument("output_bundle", type=Path)

    extract = subparsers.add_parser("extract", help="Run authorized hosted claim extraction")
    extract.add_argument("bundle", type=Path)
    extract.add_argument("--provider", choices=("opencode-go", "openrouter"), default="opencode-go")
    extract.add_argument("--model")
    extract.add_argument("--execute-hosted", action="store_true", required=True)
    extract.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "prepare":
            result = prepare_evidence(
                args.bundle,
                max_words=args.max_words,
                overlap_segments=args.overlap_segments,
                force=args.force,
            )
        elif args.command == "validate":
            segments = load_segments(args.bundle)
            markdown = args.artifact.expanduser().read_text(encoding="utf-8")
            issues = validate_citations(
                markdown,
                {str(segment["id"]) for segment in segments},
                require_citations=not args.allow_uncited,
            )
            result = {
                "status": "pass" if not issues else "failed",
                "issue_count": len(issues),
                "issues": [issue.as_dict() for issue in issues],
            }
            print(json.dumps(result, indent=2, ensure_ascii=False))
            return 0 if not issues else 2
        elif args.command == "migrate-legacy":
            result = migrate_legacy_bundle(args.legacy_bundle, args.output_bundle)
        else:
            if args.provider == "opencode-go":
                adapter = OpenCodeAdapter(model=args.model or OPENCODE_MODEL)
            else:
                adapter = OpenRouterAdapter(model=args.model or OPENROUTER_MODEL)
            result = extract_and_render(args.bundle, adapter, force=args.force)
    except (Audio2MdError, OSError, ValueError) as error:
        print(f"audio2md-study: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
