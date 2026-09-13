from wayland_whisper.output import wtype_arguments


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
