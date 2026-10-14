"""Domain errors exposed by the audio2md CLI and public API."""


class Audio2MdError(Exception):
    """Base class for expected audio2md failures."""


class InvalidTranscriptError(Audio2MdError):
    """Raised when ASR output cannot be represented safely."""


class InvalidBundleError(Audio2MdError):
    """Raised when an audio2md bundle is missing or structurally invalid."""
