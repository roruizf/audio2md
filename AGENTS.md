# audio2md — Agent Rules

- Use Python 3.12 explicitly through `uv`; never use the host Anaconda Python 3.8.
- Keep model weights, media, transcripts, caches, credentials and `.venv` out of Git.
- Preserve the local-first boundary. Network use is limited to dependency/model download unless a connector is
  explicitly approved.
- Do not implement DRM circumvention or automated access to a course platform without explicit authorization.
- Treat transcripts as primary evidence and all summaries/manuals as regenerable derived artifacts.
- Preserve source hash, parameters and timestamps whenever output formats change.
- Run `uv run pytest` and an end-to-end fixture before declaring pipeline changes complete.
- Do not create a remote, publish, push or upload artifacts without Roberto's explicit approval.
