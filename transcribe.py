#!/usr/bin/env python3
"""
Session Transcriber
Turns hours of tabletop session audio into a searchable, timestamped transcript.

Pipeline:
    load audio -> find cut points -> export chunks -> Whisper each chunk
    -> stitch segments back onto the original timeline -> write .txt / .json

Two functions are left for you to write (search for "YOUR PART"):
    find_cut_points()  - where to split the recording
    stitch_segments()  - put chunk-relative timestamps back on the session timeline
Everything else is plumbing. `pytest test_transcribe.py` checks your two parts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from pydub import AudioSegment
from pydub.silence import detect_silence  # noqa: F401  (you'll want this)

# Whisper rejects uploads over 25 MB. We aim well under it.
MAX_UPLOAD_BYTES = 24 * 1024 * 1024

# Chunks are exported as mono 16 kHz MP3 at 64 kbps: ~0.48 MB per minute.
# Whisper resamples to 16 kHz internally, so this loses nothing it would use.
SAMPLE_RATE = 16_000
EXPORT_FORMAT = "mp3"
EXPORT_BITRATE = "64k"

# Whisper's prompt only looks at roughly the last 224 tokens.
PROMPT_CHAR_BUDGET = 800


# --------------------------------------------------------------------------- #
# Data types
# --------------------------------------------------------------------------- #
@dataclass
class Segment:
    """One line of transcript. Times are in seconds."""
    start: float
    end: float
    text: str


@dataclass
class Chunk:
    """A slice of the original recording. Times are in milliseconds."""
    index: int
    start_ms: int
    end_ms: int

    @property
    def duration_ms(self) -> int:
        return self.end_ms - self.start_ms


# --------------------------------------------------------------------------- #
# YOUR PART 1: where to cut
# --------------------------------------------------------------------------- #
def find_cut_points(
    audio: AudioSegment,
    max_chunk_ms: int,
    silence_thresh_dbfs: float,
    min_silence_ms: int = 700,
    search_window_ms: int = 60_000,
) -> list[int]:
    """
    Decide where to split `audio` so every chunk fits under Whisper's limit,
    cutting at natural pauses wherever possible.

    Returns the interior cut positions in milliseconds, ascending.
    Don't include 0 or len(audio). Audio that already fits returns [].

    Contract (the tests check all of this):
      * Every chunk -- between 0, each cut, and len(audio) -- is > 0 ms and
        <= max_chunk_ms long.
      * Work forward from a cursor (starting at 0). Each new chunk may end
        anywhere up to cursor + max_chunk_ms.
      * Look for silence only in the last `search_window_ms` before that
        limit: [cursor + max_chunk_ms - search_window_ms, cursor + max_chunk_ms].
        A silence only counts if it is at least `min_silence_ms` long and
        quieter than `silence_thresh_dbfs`.
      * If you find one, cut somewhere inside it (the middle is a good choice).
      * If you don't, fall back to a hard cut at exactly cursor + max_chunk_ms.
      * The next chunk starts from wherever you actually cut, not from a
        fixed grid.
      * Stop once the remaining audio fits in one chunk.

    Hints:
      * pydub.silence.detect_silence(segment, min_silence_len, silence_thresh)
        returns [[start_ms, end_ms], ...] relative to the segment you pass in.
        Slicing first (audio[a:b]) means you have to add `a` back.
      * Don't run detect_silence over the whole recording. On a 4-hour file
        it's very slow; you only ever need the window.
      * Several silences in the window? Your call: the latest gives longer
        chunks, the longest is the safest place to cut.
    """
    raise NotImplementedError("find_cut_points: your part")


# --------------------------------------------------------------------------- #
# YOUR PART 2: stitching timestamps back together
# --------------------------------------------------------------------------- #
def stitch_segments(
    chunks: list[Chunk],
    results: list[list[Segment]],
) -> list[Segment]:
    """
    Whisper returns times relative to the start of each chunk. Map them back
    onto the original recording and flatten into one list.

    `results[i]` holds the segments Whisper returned for `chunks[i]`.

    Contract (the tests check all of this):
      * Shift each segment's start and end by its chunk's start (ms -> seconds).
      * Strip whitespace from text; drop segments whose text is then empty.
      * Whisper sometimes reports times past the end of the audio it was
        given. Clamp each segment's end to its chunk's duration, and drop a
        segment entirely if it *starts* at or after the chunk's duration.
      * Return segments in chronological order by start time.
      * Return new Segment objects; don't mutate the inputs.
    """
    raise NotImplementedError("stitch_segments: your part")


# --------------------------------------------------------------------------- #
# Plumbing: audio
# --------------------------------------------------------------------------- #
def load_audio(path: Path) -> AudioSegment:
    """Decode any ffmpeg-readable file straight to mono 16 kHz.

    Asking ffmpeg to downmix during decode keeps memory sane: a 4-hour session
    is ~460 MB this way versus ~2.5 GB as stereo 44.1 kHz.
    """
    audio = AudioSegment.from_file(
        str(path), parameters=["-ac", "1", "-ar", str(SAMPLE_RATE)]
    )
    # Belt and braces in case the parameters were ignored for this format.
    if audio.channels != 1 or audio.frame_rate != SAMPLE_RATE:
        audio = audio.set_channels(1).set_frame_rate(SAMPLE_RATE)
    return audio


def build_chunks(total_ms: int, cuts: list[int]) -> list[Chunk]:
    bounds = [0, *cuts, total_ms]
    return [Chunk(i, a, b) for i, (a, b) in enumerate(zip(bounds, bounds[1:]))]


def check_cuts(cuts: list[int], total_ms: int, max_chunk_ms: int) -> None:
    """Catch contract violations from find_cut_points with a readable error."""
    if any(not isinstance(c, int) for c in cuts):
        raise ValueError(f"cut points must be ints (ms), got {cuts!r}")
    bounds = [0, *cuts, total_ms]
    for a, b in zip(bounds, bounds[1:]):
        if b <= a:
            raise ValueError(f"cut points must be strictly increasing inside (0, {total_ms}); got {cuts}")
        if b - a > max_chunk_ms:
            raise ValueError(
                f"chunk {fmt_ts(a / 1000)}-{fmt_ts(b / 1000)} is {(b - a) / 1000:.1f}s, "
                f"over the {max_chunk_ms / 1000:.0f}s maximum"
            )


def export_chunk(audio: AudioSegment, chunk: Chunk, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"chunk_{chunk.index:03d}_{fmt_ts(chunk.start_ms / 1000).replace(':', '-')}.{EXPORT_FORMAT}"
    audio[chunk.start_ms:chunk.end_ms].export(str(path), format=EXPORT_FORMAT, bitrate=EXPORT_BITRATE)
    size = path.stat().st_size
    if size > MAX_UPLOAD_BYTES:
        raise RuntimeError(
            f"{path.name} is {size / 1e6:.1f} MB, over Whisper's limit. "
            f"Lower --max-chunk-min."
        )
    return path


# --------------------------------------------------------------------------- #
# Plumbing: Whisper
# --------------------------------------------------------------------------- #
def make_client():
    try:
        from openai import OpenAI
    except ImportError:
        sys.exit("The openai package isn't installed: pip install -r requirements.txt")
    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("Set OPENAI_API_KEY first (or use --dry-run to test cutting without the API).")
    # The SDK retries rate limits and transient errors itself.
    return OpenAI(max_retries=4, timeout=600)


def build_prompt(vocab: str, previous_text: str) -> str | None:
    """Whisper's prompt is a spelling and style hint, not an instruction.

    Proper nouns (character and place names) help a lot with fantasy names.
    The tail of the previous chunk helps it carry context across the cut.
    """
    parts = []
    if vocab:
        parts.append(vocab.strip())
    if previous_text:
        room = PROMPT_CHAR_BUDGET - sum(len(p) for p in parts) - 1
        if room > 0:
            parts.append(previous_text[-room:].strip())
    prompt = " ".join(p for p in parts if p)
    return prompt[-PROMPT_CHAR_BUDGET:] or None


def _get(obj, key):
    return obj.get(key) if isinstance(obj, dict) else getattr(obj, key)


def transcribe_chunk(client, path: Path, model: str, language: str | None, prompt: str | None) -> list[Segment]:
    kwargs = dict(
        model=model,
        response_format="verbose_json",
        timestamp_granularities=["segment"],
    )
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["prompt"] = prompt
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(file=f, **kwargs)
    segments = _get(resp, "segments") or []
    return [Segment(float(_get(s, "start")), float(_get(s, "end")), _get(s, "text")) for s in segments]


# --------------------------------------------------------------------------- #
# Plumbing: cache
# A 4-hour session is ~12 API calls. If call 9 fails, don't pay for 1-8 again.
# Cache entries are keyed by chunk boundaries, so changing your cut logic
# invalidates exactly the chunks that moved.
# --------------------------------------------------------------------------- #
def cache_path(cache_dir: Path, chunk: Chunk, model: str, language: str | None) -> Path:
    return cache_dir / f"{model}_{language or 'auto'}_{chunk.start_ms}-{chunk.end_ms}.json"


def read_cache(path: Path) -> list[Segment] | None:
    try:
        return [Segment(**s) for s in json.loads(path.read_text(encoding="utf-8"))]
    except (OSError, ValueError, TypeError):
        return None


def write_cache(path: Path, segments: list[Segment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([asdict(s) for s in segments]), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Plumbing: output
# --------------------------------------------------------------------------- #
def fmt_ts(seconds: float) -> str:
    total = int(seconds)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def write_txt(segments: list[Segment], path: Path) -> None:
    lines = [f"[{fmt_ts(s.start)}] {s.text}" for s in segments]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(segments: list[Segment], path: Path, source: Path, model: str, duration_s: float) -> None:
    data = {
        "source": source.name,
        "model": model,
        "duration_seconds": round(duration_s, 3),
        "segments": [
            {"start": round(s.start, 3), "end": round(s.end, 3), "timestamp": fmt_ts(s.start), "text": s.text}
            for s in segments
        ],
    }
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def print_plan(chunks: list[Chunk]) -> None:
    print(f"{len(chunks)} chunk(s):")
    for c in chunks:
        print(f"  #{c.index:03d}  {fmt_ts(c.start_ms / 1000)} -> {fmt_ts(c.end_ms / 1000)}  ({c.duration_ms / 60000:.1f} min)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe a tabletop session recording with Whisper.")
    p.add_argument("audio", type=Path, help="session recording (anything ffmpeg can read)")
    p.add_argument("-o", "--output", type=Path, help="output path without extension (default: next to the input)")
    p.add_argument("-f", "--format", choices=["txt", "json", "both"], default="txt")
    p.add_argument("--model", default="whisper-1", help="must support verbose_json segments (default: whisper-1)")
    p.add_argument("--language", default="en", help="ISO-639-1 code, or 'auto' to let Whisper detect it (default: en)")
    p.add_argument("--vocab", type=Path, help="text file of names/terms to bias spelling (e.g. characters, places)")

    g = p.add_argument_group("cutting")
    g.add_argument("--max-chunk-min", type=float, default=20, help="longest chunk in minutes (default: 20)")
    g.add_argument("--search-window-s", type=float, default=60, help="how far back from the limit to look for a pause (default: 60)")
    g.add_argument("--min-silence-ms", type=int, default=700, help="shortest pause that counts (default: 700)")
    g.add_argument("--silence-offset-db", type=float, default=16,
                   help="silence = this many dB below the recording's average loudness (default: 16)")

    d = p.add_argument_group("debugging")
    d.add_argument("--dry-run", action="store_true", help="plan the chunks and stop; no API calls")
    d.add_argument("--export-chunks", type=Path, metavar="DIR", help="also save the chunk MP3s here so you can listen to the cuts")
    d.add_argument("--no-cache", action="store_true", help="ignore cached chunk transcripts")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    src: Path = args.audio
    if not src.exists():
        sys.exit(f"No such file: {src}")

    max_chunk_ms = int(args.max_chunk_min * 60_000)
    window_ms = int(args.search_window_s * 1000)
    if not 0 < window_ms < max_chunk_ms:
        sys.exit("--search-window-s must be positive and shorter than --max-chunk-min")
    language = None if args.language == "auto" else args.language
    vocab = ""
    if args.vocab:
        terms = [line.strip() for line in args.vocab.read_text(encoding="utf-8").splitlines()]
        vocab = ", ".join(t for t in terms if t and not t.startswith("#")) + "."

    print(f"Loading {src.name} ...", flush=True)
    audio = load_audio(src)
    total_ms = len(audio)
    thresh = audio.dBFS - args.silence_offset_db
    print(f"  {fmt_ts(total_ms / 1000)} long, average {audio.dBFS:.1f} dBFS, silence below {thresh:.1f} dBFS")

    cuts = find_cut_points(audio, max_chunk_ms, thresh, args.min_silence_ms, window_ms)
    check_cuts(cuts, total_ms, max_chunk_ms)
    chunks = build_chunks(total_ms, cuts)
    print_plan(chunks)

    if args.export_chunks:
        for c in chunks:
            export_chunk(audio, c, args.export_chunks)
        print(f"Chunk audio saved to {args.export_chunks}/")
    if args.dry_run:
        return 0

    client = make_client()
    out_base = args.output or src.with_suffix("")
    cache_dir = out_base.parent / ".transcribe_cache" / src.stem
    tmp_dir = cache_dir / "audio"

    results: list[list[Segment]] = []
    previous_text = ""
    for c in chunks:
        label = f"  chunk {c.index + 1}/{len(chunks)}"
        cpath = cache_path(cache_dir, c, args.model, language)
        segs = None if args.no_cache else read_cache(cpath)
        if segs is not None:
            print(f"{label}: cached")
        else:
            print(f"{label}: transcribing ...", flush=True)
            mp3 = export_chunk(audio, c, tmp_dir)
            try:
                segs = transcribe_chunk(client, mp3, args.model, language, build_prompt(vocab, previous_text))
            finally:
                mp3.unlink(missing_ok=True)
            write_cache(cpath, segs)
        results.append(segs)
        # Carry context into the next chunk, skipping anything Whisper
        # reported past the end of this chunk (usually hallucinated).
        previous_text = " ".join(s.text.strip() for s in segs if s.start < c.duration_ms / 1000)

    segments = stitch_segments(chunks, results)

    written = []
    if args.format in ("txt", "both"):
        write_txt(segments, out_base.with_suffix(".txt"))
        written.append(out_base.with_suffix(".txt"))
    if args.format in ("json", "both"):
        write_json(segments, out_base.with_suffix(".json"), src, args.model, total_ms / 1000)
        written.append(out_base.with_suffix(".json"))
    print(f"Done: {len(segments)} segments -> " + ", ".join(str(w) for w in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
