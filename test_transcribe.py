"""
Tests for your two functions. Run with:  pytest -q

They use short synthetic audio (a tone for "talking", digital silence for
"pauses") with small limits so they run in a second or two:
    max chunk 10 s, search window 4 s, min silence 500 ms
"""
import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from transcribe import Chunk, Segment, find_cut_points, stitch_segments

RATE = 16_000
MAX_MS = 10_000
WINDOW_MS = 4_000
MIN_SIL_MS = 500
THRESH = -40.0  # tone is -10 dBFS; silence is -inf


def talk(ms: int) -> AudioSegment:
    return Sine(440, sample_rate=RATE).to_audio_segment(duration=ms, volume=-10).set_channels(1)


def pause(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms, frame_rate=RATE)


def cuts_for(audio: AudioSegment) -> list[int]:
    return find_cut_points(audio, MAX_MS, THRESH, MIN_SIL_MS, WINDOW_MS)


def assert_valid(cuts: list[int], total: int) -> None:
    bounds = [0, *cuts, total]
    for a, b in zip(bounds, bounds[1:]):
        assert b > a, f"cuts not strictly increasing: {cuts}"
        assert b - a <= MAX_MS, f"chunk {a}-{b} is longer than {MAX_MS} ms: {cuts}"


# --------------------------------------------------------------------------- #
# find_cut_points
# --------------------------------------------------------------------------- #
class TestFindCutPoints:
    def test_short_audio_needs_no_cuts(self):
        assert cuts_for(talk(6_000)) == []

    def test_exactly_max_length_needs_no_cuts(self):
        assert cuts_for(talk(MAX_MS)) == []

    def test_no_silence_falls_back_to_hard_cuts(self):
        audio = talk(25_000)
        assert cuts_for(audio) == [10_000, 20_000]

    def test_cuts_inside_a_pause_in_the_window(self):
        # pause at 8.0-9.0 s, inside the 6-10 s window
        audio = talk(8_000) + pause(1_000) + talk(8_000)
        cuts = cuts_for(audio)
        assert len(cuts) == 1
        assert 8_000 <= cuts[0] <= 9_000, f"expected a cut inside the pause (8000-9000), got {cuts}"
        assert_valid(cuts, len(audio))

    def test_ignores_pause_before_the_window(self):
        # pause at 3.0-4.0 s is too early (window is 6-10 s); cutting there
        # would make a needlessly tiny chunk
        audio = talk(3_000) + pause(1_000) + talk(13_000)
        assert cuts_for(audio) == [10_000]

    def test_ignores_pauses_shorter_than_min_silence(self):
        # a 200 ms gap between words isn't a safe cut
        audio = talk(8_000) + pause(200) + talk(8_000)
        assert cuts_for(audio) == [10_000]

    def test_next_chunk_starts_from_the_actual_cut(self):
        # First pause 7.0-8.0 s -> cut ~7.5 s.
        # The next window is then ~[13.5, 17.5] s, which contains the pause at
        # 15.0-16.0 s. A fixed 10 s grid would look at [16, 20] s and miss it.
        audio = talk(7_000) + pause(1_000) + talk(7_000) + pause(1_000) + talk(9_000)
        cuts = cuts_for(audio)
        assert len(cuts) == 2, f"expected 2 cuts, got {cuts}"
        assert 7_000 <= cuts[0] <= 8_000
        assert 15_000 <= cuts[1] <= 16_000
        assert_valid(cuts, len(audio))

    def test_mixed_session_obeys_the_contract(self):
        # A longer mix of pauses and crosstalk; just check the invariants.
        audio = (
            talk(9_000) + pause(800)        # pause in window 1
            + talk(14_000)                  # long crosstalk, no pause
            + talk(5_000) + pause(600)      # short-ish pause
            + talk(20_000) + pause(1_500)
            + talk(3_000)
        )
        cuts = cuts_for(audio)
        assert cuts == sorted(cuts)
        assert all(isinstance(c, int) for c in cuts), "cut points should be ints (ms)"
        assert all(0 < c < len(audio) for c in cuts)
        assert_valid(cuts, len(audio))

    def test_returns_ints(self):
        audio = talk(8_000) + pause(1_000) + talk(8_000)
        assert all(isinstance(c, int) for c in cuts_for(audio))


# --------------------------------------------------------------------------- #
# stitch_segments
# --------------------------------------------------------------------------- #
class TestStitchSegments:
    def test_empty(self):
        assert stitch_segments([], []) == []

    def test_offsets_each_chunk_by_its_start(self):
        chunks = [Chunk(0, 0, 10_000), Chunk(1, 10_000, 25_000)]
        results = [
            [Segment(0.0, 4.0, "a"), Segment(4.0, 9.5, "b")],
            [Segment(0.0, 3.0, "c"), Segment(3.0, 7.25, "d")],
        ]
        out = stitch_segments(chunks, results)
        assert [(s.start, s.end, s.text) for s in out] == [
            (pytest.approx(0.0), pytest.approx(4.0), "a"),
            (pytest.approx(4.0), pytest.approx(9.5), "b"),
            (pytest.approx(10.0), pytest.approx(13.0), "c"),
            (pytest.approx(13.0), pytest.approx(17.25), "d"),
        ]

    def test_uses_real_chunk_starts_not_index_times_length(self):
        # Silence-based cuts make uneven chunks.
        chunks = [Chunk(0, 0, 7_500), Chunk(1, 7_500, 15_250), Chunk(2, 15_250, 20_000)]
        results = [[Segment(1.0, 2.0, "x")], [Segment(1.0, 2.0, "y")], [Segment(1.0, 2.0, "z")]]
        out = stitch_segments(chunks, results)
        assert [s.start for s in out] == [pytest.approx(1.0), pytest.approx(8.5), pytest.approx(16.25)]

    def test_strips_text_and_drops_empty(self):
        chunks = [Chunk(0, 0, 10_000)]
        results = [[Segment(0.0, 1.0, "  Roll initiative.  "), Segment(1.0, 2.0, "   "), Segment(2.0, 3.0, "")]]
        out = stitch_segments(chunks, results)
        assert [s.text for s in out] == ["Roll initiative."]

    def test_clamps_end_to_chunk_duration(self):
        chunks = [Chunk(0, 0, 10_000), Chunk(1, 10_000, 20_000)]
        results = [[Segment(8.0, 12.0, "overruns")], [Segment(0.0, 1.0, "next")]]
        out = stitch_segments(chunks, results)
        assert out[0].end == pytest.approx(10.0)
        assert out[1].start == pytest.approx(10.0)

    def test_drops_segments_starting_past_the_chunk(self):
        chunks = [Chunk(0, 0, 10_000)]
        results = [[Segment(5.0, 6.0, "real"), Segment(10.0, 11.0, "hallucinated"), Segment(12.0, 13.0, "also")]]
        out = stitch_segments(chunks, results)
        assert [s.text for s in out] == ["real"]

    def test_output_is_chronological(self):
        chunks = [Chunk(0, 0, 10_000), Chunk(1, 10_000, 20_000)]
        results = [[Segment(6.0, 7.0, "b"), Segment(1.0, 2.0, "a")], [Segment(0.5, 1.0, "c")]]
        out = stitch_segments(chunks, results)
        assert [s.text for s in out] == ["a", "b", "c"]

    def test_does_not_mutate_inputs(self):
        seg = Segment(1.0, 2.0, " hi ")
        chunks = [Chunk(0, 0, 10_000), Chunk(1, 10_000, 20_000)]
        stitch_segments(chunks, [[], [seg]])
        assert (seg.start, seg.end, seg.text) == (1.0, 2.0, " hi ")
