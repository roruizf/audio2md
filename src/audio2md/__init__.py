"""Turn local audio and video into auditable Markdown."""

__version__ = "0.1.0"

from .models import Segment, Transcript, TranscriptionConfig
from .pipeline import run_transcription

__all__ = ["Segment", "Transcript", "TranscriptionConfig", "run_transcription"]
