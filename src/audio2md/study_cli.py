from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .evidence import load_segments, prepare_evidence, validate_citations
from .exceptions import Audio2MdError


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
        else:
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
    except (Audio2MdError, OSError, ValueError) as error:
        print(f"audio2md-study: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
