# Session Transcriber

Turns hours of tabletop session audio into a searchable, timestamped transcript.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

You also need **ffmpeg** on your PATH (`brew install ffmpeg`, `winget install ffmpeg`, or `apt install ffmpeg`).

```bash
export OPENAI_API_KEY=sk-...
```

## Usage

```bash
# Check where it would cut, no API calls
python transcribe.py session.m4a --dry-run

# Save the chunks so you can listen to the cut points
python transcribe.py session.m4a --dry-run --export-chunks chunks/

# Transcribe
python transcribe.py session.m4a                     # -> session.txt
python transcribe.py session.m4a -f both             # -> session.txt + session.json
python transcribe.py session.m4a --vocab names.txt   # better spelling of fantasy names
```

`names.txt` is one name or term per line (characters, places, factions, spells). Whisper uses it as a spelling hint.

Run `python transcribe.py -h` for the cutting knobs (`--max-chunk-min`, `--search-window-s`, `--min-silence-ms`, `--silence-offset-db`).

## How it works

Whisper's API caps uploads at 25 MB. The script decodes the recording to mono 16 kHz, splits it into chunks of up to 20 minutes (about 10 MB as 64 kbps MP3), preferring natural pauses near each limit and falling back to a hard cut during crosstalk. Each chunk is transcribed with the tail of the previous chunk as context, and the segments are shifted back onto the session timeline.

Finished chunks are cached in `.transcribe_cache/`, so if a run dies partway through a four-hour session you only pay for what's left.

## Tests

```bash
pytest -q
```
