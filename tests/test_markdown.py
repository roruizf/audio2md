from audio2md.markdown import format_timestamp, render_transcript_markdown
from audio2md.models import Segment, Transcript


def test_format_timestamp_supports_hours() -> None:
    assert format_timestamp(3661.234) == "01:01:01.234"


def test_render_uses_original_timeline_offset() -> None:
    transcript = Transcript(
        language="en",
        language_probability=1.0,
        duration_seconds=10.0,
        segments=(Segment(start=1.5, end=3.25, text="Appendix G."),),
    )
    rendered = render_transcript_markdown(
        collection_id="collection-one",
        item_id="item-one",
        title="Lesson One",
        source_name="clip.mp3",
        source_sha256="abc123",
        source_offset_seconds=120.0,
        transcript=transcript,
    )
    assert "[S000001 | 00:02:01.500 → 00:02:03.250] Appendix G." in rendered
    assert 'collection_id: "collection-one"' in rendered
