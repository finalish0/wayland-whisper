"""Send finished dictation text to the focused Wayland client."""

from __future__ import annotations

import subprocess


_SPACE_PREFIX = " \t\u00a0"


def continued_text(text: str, continuing: bool) -> str:
    """Separate consecutive phrases of one dictation with a space.

    Every phrase is transcribed and stripped on its own, so the junction
    between two phrases would otherwise be typed without a space
    ("hat.Also"). The leading space added here is turned into a real
    space keypress by wtype_arguments.
    """
    if continuing and text and not text[0].isspace():
        return " " + text
    return text


def wtype_arguments(text: str) -> list[str]:
    """Build an argv that keeps initial spaces reliable in Wayland clients."""
    command = ["wtype", "-d", "8", "-s", "40", "-M", "shift", "-m", "shift"]
    position = 0
    while position < len(text) and text[position] in _SPACE_PREFIX:
        command.extend(("-s", "12", "-k", "space"))
        position += 1
    if position != len(text):
        command.extend(("--", text[position:]))
    return command


def type_text(text: str) -> None:
    """Type text through wtype, raising a useful error on failure."""
    if not text:
        return
    try:
        completed = subprocess.run(wtype_arguments(text), check=False)
    except OSError as error:
        raise RuntimeError("could not start wtype") from error
    if completed.returncode:
        raise RuntimeError("wtype exited with status %d" % completed.returncode)
