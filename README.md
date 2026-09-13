# wayland-whisper

`wayland-whisper` turns local microphone speech into Wayland keyboard input.
It captures PCM with `parec`, uses a local Silero VAD ONNX model to decide
when speech has ended, transcribes the speech with `whisper.cpp`, and sends the
result through `wtype`.

The project is a focused local speech-to-text utility with a small, dedicated
command-line interface.

## Requirements

- Python 3.10 or newer
- `parec` from PulseAudio or PipeWire's PulseAudio compatibility service
- `wtype`
- a local Whisper GGML/GGUF model accepted by `pywhispercpp`
- a local `silero_vad.onnx` model

## Install

```bash
python -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

## Run

```bash
.venv/bin/wayland-whisper run \
  --whisper-model /path/to/ggml-small.bin \
  --vad-model /path/to/silero_vad.onnx \
  --language de
```

Speech begins after six voiced 32 ms frames and is committed after about 700
ms of silence. `Ctrl+C` stops capture cleanly. Use `--output stdout` while
testing to avoid typing into the focused application.

## License Notes

This repository's code is MIT licensed. It runs external programs and Python
packages; their licenses remain their own. At the time this project was
created, pywhispercpp/whisper.cpp, ONNX Runtime, and Silero VAD publish MIT
licenses. Model weights can have additional terms, so verify the exact model
file before redistribution.
