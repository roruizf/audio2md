"""Domain errors exposed by the audio2md CLI and public API."""


class Audio2MdError(Exception):
    """Base class for expected audio2md failures."""


class InvalidTranscriptError(Audio2MdError):
    """Raised when ASR output cannot be represented safely."""
