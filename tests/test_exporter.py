from transcribator.exporter import (
    build_text_blocks,
    render_text_transcript,
    speaker_label_template_for_language,
)


def test_build_text_blocks_merges_short_neighbor_segments():
    segments = [
        {"start": 0.0, "end": 0.7, "text": "hello", "speaker": 0},
        {"start": 0.8, "end": 1.4, "text": "world", "speaker": 0},
        {"start": 2.8, "end": 3.5, "text": "reply", "speaker": 1},
    ]

    blocks = build_text_blocks(segments)

    assert blocks == [
        {"start": 0.0, "end": 1.4, "speaker": 0, "text": "hello world"},
        {"start": 2.8, "end": 3.5, "speaker": 1, "text": "reply"},
    ]


def test_render_text_transcript_creates_paragraphs_per_speaker():
    segments = [
        {"start": 0.0, "end": 0.7, "text": "hello", "speaker": 0},
        {"start": 0.8, "end": 1.4, "text": "world", "speaker": 0},
        {"start": 2.8, "end": 3.5, "text": "reply", "speaker": 1},
        {"start": 3.6, "end": 4.2, "text": "again", "speaker": 1},
    ]

    rendered = render_text_transcript(
        segments,
        include_timestamps=False,
        include_speakers=True,
        speaker_label_template="[Speaker {index}]",
    )

    assert rendered == "[Speaker 1]\nhello world\n\n[Speaker 2]\nreply again"


def test_render_text_transcript_keeps_timestamps_on_merged_blocks():
    segments = [
        {"start": 0.0, "end": 0.7, "text": "hello", "speaker": 0},
        {"start": 0.8, "end": 1.4, "text": "world", "speaker": 0},
    ]

    rendered = render_text_transcript(
        segments,
        include_timestamps=True,
        include_speakers=False,
    )

    assert rendered == "[00:00] hello world"


def test_build_text_blocks_normalizes_common_continuation_word_case():
    segments = [
        {"start": 0.0, "end": 0.7, "text": "Да, отлично", "speaker": 0},
        {"start": 0.8, "end": 1.4, "text": "И видно", "speaker": 0},
        {"start": 1.5, "end": 2.2, "text": "Теймур", "speaker": 0},
    ]

    blocks = build_text_blocks(segments)

    assert blocks == [
        {"start": 0.0, "end": 2.2, "speaker": 0, "text": "Да, отлично и видно Теймур"},
    ]


def test_speaker_label_template_can_switch_language():
    assert speaker_label_template_for_language("en") == "[Speaker {index}]"
    assert speaker_label_template_for_language("ru") == "[Спикер {index}]"
