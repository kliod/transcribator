"""Export transcription results to text and subtitle formats."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Iterable, List


SENTENCE_ENDINGS = (".", "!", "?", "…")
LEADING_WORD_PATTERN = re.compile(r"^([\"'“”„«»([{]*)?([A-Za-zА-ЯЁ][A-Za-zА-Яа-яЁё'-]*)(.*)$")
LOWERCASE_CONTINUATION_WORDS = {
    "a", "an", "and", "as", "at", "because", "but", "by", "for", "from", "he", "her", "hers",
    "him", "his", "i", "if", "in", "is", "it", "its", "me", "my", "no", "not", "of", "on", "or",
    "our", "she", "so", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "we", "well", "with", "you", "your",
    "а", "без", "бы", "был", "была", "были", "было", "в", "во", "вот", "вроде", "вы", "да", "для",
    "до", "если", "еще", "ещё", "же", "за", "и", "из", "или", "именно", "их", "к", "как", "какая",
    "какие", "какой", "ко", "когда", "которые", "который", "ли", "либо", "меня", "мне", "мы", "на",
    "над", "не", "него", "нет", "но", "ну", "о", "об", "однако", "он", "она", "они", "оно", "от",
    "очень", "по", "пока", "потому", "потом", "почему", "при", "про", "просто", "с", "со", "сейчас",
    "скажем", "собственно", "так", "такая", "такие", "такой", "там", "то", "тоже", "только", "тут",
    "ты", "у", "уж", "уже", "хорошо", "хотя", "что", "чтобы", "эта", "это", "этот", "я",
}


def format_timestamp(seconds: float) -> str:
    """Format seconds as an SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds as a VTT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def format_timestamp_readable(seconds: float) -> str:
    """Format seconds as a readable TXT timestamp."""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"[{minutes:02d}:{secs:02d}]"


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _word_count(text: str) -> int:
    return len([word for word in text.split() if word])


def speaker_label_template_for_language(ui_language: str) -> str:
    return "[Спикер {index}]" if (ui_language or "").lower() == "ru" else "[Speaker {index}]"


def _normalize_continuation_text(previous_text: str, text: str) -> str:
    normalized = _normalize_text(text)
    if not normalized:
        return normalized

    previous_text = _normalize_text(previous_text)
    if not previous_text or previous_text.endswith(SENTENCE_ENDINGS):
        return normalized

    match = LEADING_WORD_PATTERN.match(normalized)
    if not match:
        return normalized

    prefix = match.group(1) or ""
    word = match.group(2)
    suffix = match.group(3) or ""
    normalized_word = word.strip("\"'“”„«»()[]{}.,!?;:-").lower()
    if not normalized_word or normalized_word not in LOWERCASE_CONTINUATION_WORDS:
        return normalized
    if word.isupper() and len(word) > 1:
        return normalized

    lowered_word = word[:1].lower() + word[1:]
    return f"{prefix}{lowered_word}{suffix}"


def build_text_blocks(
    segments: Iterable[Dict],
    *,
    max_pause: float = 1.0,
    max_words: int = 48,
    max_duration: float = 35.0,
) -> List[Dict]:
    """Merge raw ASR segments into readable text blocks."""
    blocks: List[Dict] = []

    for segment in segments:
        text = _normalize_text(str(segment.get("text") or ""))
        if not text:
            continue

        start = float(segment["start"])
        end = float(segment["end"])
        speaker = segment.get("speaker")

        if not blocks:
            blocks.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            })
            continue

        current = blocks[-1]
        pause = max(0.0, start - float(current["end"]))
        continuation_text = _normalize_continuation_text(str(current["text"]), text)
        merged_text = f"{current['text']} {continuation_text}".strip()
        should_merge = (
            current.get("speaker") == speaker
            and pause <= max_pause
            and (end - float(current["start"])) <= max_duration
            and _word_count(merged_text) <= max_words
        )

        if should_merge:
            current["end"] = end
            current["text"] = merged_text
        else:
            blocks.append({
                "start": start,
                "end": end,
                "speaker": speaker,
                "text": text,
            })

    return blocks


def render_text_transcript(
    segments: Iterable[Dict],
    *,
    include_timestamps: bool,
    include_speakers: bool,
    speaker_label_template: str = "[Speaker {index}]",
) -> str:
    """Render a readable transcript from diarized segments."""
    blocks = build_text_blocks(segments)
    rendered_blocks: List[str] = []

    for block in blocks:
        parts: List[str] = []
        speaker = block.get("speaker")
        if include_speakers and speaker is not None:
            parts.append(speaker_label_template.format(index=int(speaker) + 1))

        line = block["text"]
        if include_timestamps:
            line = f"{format_timestamp_readable(float(block['start']))} {line}"
        parts.append(line)
        rendered_blocks.append("\n".join(parts))

    return "\n\n".join(rendered_blocks).strip()


def export_txt(
    segments: List[Dict],
    output_path: str,
    include_timestamps: bool = True,
    include_speakers: bool = False,
    speaker_label_template: str = "[Speaker {index}]",
):
    """Export a readable TXT transcript."""
    content = render_text_transcript(
        segments,
        include_timestamps=include_timestamps,
        include_speakers=include_speakers,
        speaker_label_template=speaker_label_template,
    )

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write(content)
        if content:
            handle.write("\n")

    print(f"Text saved: {output_path}")


def export_srt(segments: List[Dict], output_path: str):
    """Export an SRT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as handle:
        for index, segment in enumerate(segments, start=1):
            start_time = format_timestamp(float(segment["start"]))
            end_time = format_timestamp(float(segment["end"]))
            text = str(segment["text"])

            handle.write(f"{index}\n")
            handle.write(f"{start_time} --> {end_time}\n")
            handle.write(f"{text}\n\n")

    print(f"SRT saved: {output_path}")


def export_vtt(segments: List[Dict], output_path: str):
    """Export a WebVTT subtitle file."""
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("WEBVTT\n\n")

        for segment in segments:
            start_time = format_timestamp_vtt(float(segment["start"]))
            end_time = format_timestamp_vtt(float(segment["end"]))
            text = str(segment["text"])

            handle.write(f"{start_time} --> {end_time}\n")
            handle.write(f"{text}\n\n")

    print(f"VTT saved: {output_path}")


def export_transcription(
    segments: List[Dict],
    output_base_path: str,
    formats: List[str],
    include_timestamps_in_txt: bool = True,
    include_speakers: bool = False,
    speaker_label_template: str = "[Speaker {index}]",
):
    """Export a transcript to the requested formats."""
    if not segments:
        raise ValueError("No segments available for export.")

    for index, segment in enumerate(segments):
        if "start" not in segment or "end" not in segment or "text" not in segment:
            raise ValueError(
                f"Segment {index} is missing one of the required fields: start, end, text."
            )

    output_path = Path(output_base_path)

    if "txt" in formats:
        export_txt(
            segments,
            str(output_path.with_suffix(".txt")),
            include_timestamps=include_timestamps_in_txt,
            include_speakers=include_speakers,
            speaker_label_template=speaker_label_template,
        )

    if "srt" in formats:
        export_srt(segments, str(output_path.with_suffix(".srt")))

    if "vtt" in formats:
        export_vtt(segments, str(output_path.with_suffix(".vtt")))
