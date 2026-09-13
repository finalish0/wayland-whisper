from wayland_whisper.vad import PhraseCollector


def test_collector_waits_for_voice_then_finishes_after_silence() -> None:
    collector = PhraseCollector(activation_frames=2, silence_frames=2, lead_frames=2)
    frame = b"x" * 1024

    assert collector.add(frame, 0.9) is None
    assert collector.add(frame, 0.9) is None
    assert collector.add(frame, 0.1) is None
    assert collector.add(frame, 0.1) == frame * 4


def test_collector_discards_short_noise() -> None:
    collector = PhraseCollector(activation_frames=2)
    assert collector.add(b"x" * 1024, 0.9) is None
    assert collector.add(b"x" * 1024, 0.1) is None
    assert collector.finish() is None
