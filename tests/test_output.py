from wayland_whisper.output import continued_text, wtype_arguments


def test_leading_space_is_a_keypress() -> None:
    assert wtype_arguments(" hello") == [
        "wtype",
        "-d",
        "8",
        "-s",
        "40",
        "-M",
        "shift",
        "-m",
        "shift",
        "-s",
        "12",
        "-k",
        "space",
        "--",
        "hello",
    ]


def test_only_space_has_no_text_argument() -> None:
    assert wtype_arguments(" ")[-4:] == ["-s", "12", "-k", "space"]


def test_continuation_gets_a_separator_space() -> None:
    assert continued_text("also das scheint", True) == " also das scheint"


def test_first_phrase_has_no_leading_space() -> None:
    assert continued_text("hallo welt", False) == "hallo welt"


def test_existing_leading_space_is_not_doubled() -> None:
    assert continued_text(" hallo", True) == " hallo"


def test_empty_text_stays_empty() -> None:
    assert continued_text("", True) == ""
