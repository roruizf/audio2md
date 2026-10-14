from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path
from typing import Sequence

from . import __version__
from .models import TranscriptionConfig
from .pipeline import run_transcription
from .exceptions import Audio2MdError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="audio2md", description="Turn local audio and video into auditable Markdown"
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("source", metavar="INPUT", type=Path)
    parser.add_argument("-o", "--output-root", metavar="OUTPUT_ROOT", type=Path, required=True)
    parser.add_argument("--collection-id")
    parser.add_argument("--item-id")
    parser.add_argument("--title")
    parser.add_argument("--language", default="en", help="ISO language code; use 'auto' to detect")
    parser.add_argument("--vocabulary-file", type=Path)
    parser.add_argument("--source-offset-seconds", type=float, default=0.0)
    parser.add_argument("--model", default="small.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--chunk-seconds", type=float, default=600.0)
    parser.add_argument("--chunk-overlap-seconds", type=float, default=30.0)
    parser.add_argument("--no-vad", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-z0-9]+", "-", normalized.lower()).strip("-")
    if not slug:
        raise ValueError(f"Cannot infer an item-id from {value!r}; pass --item-id")
    return slug


def title_from_stem(stem: str) -> str:
    title = re.sub(r"[_-]+", " ", stem).strip()
    return re.sub(r"\s+", " ", title) or stem


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        initial_prompt = None
        if args.vocabulary_file:
            initial_prompt = args.vocabulary_file.expanduser().read_text(encoding="utf-8").strip()
            if not initial_prompt:
                raise ValueError("vocabulary-file is empty")
        language = None if args.language.lower() == "auto" else args.language
        config = TranscriptionConfig(
            model=args.model,
            device=args.device,
            compute_type=args.compute_type,
            cpu_threads=args.cpu_threads,
            beam_size=args.beam_size,
            vad_filter=not args.no_vad,
            language=language,
            initial_prompt=initial_prompt,
            hotwords=initial_prompt,
        )
        result = run_transcription(
            source=args.source,
            output_root=args.output_root,
            collection_id=args.collection_id,
            item_id=args.item_id or slugify(args.source.stem),
            title=args.title or title_from_stem(args.source.stem),
            config=config,
            source_offset_seconds=args.source_offset_seconds,
            force=args.force,
            chunk_seconds=args.chunk_seconds,
            chunk_overlap_seconds=args.chunk_overlap_seconds,
        )
    except (Audio2MdError, FileNotFoundError, OSError, ValueError) as error:
        print(f"audio2md: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 2 if result.get("quality_warnings") else 0


if __name__ == "__main__":
    raise SystemExit(main())
