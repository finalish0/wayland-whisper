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

Speech begins after six voiced 32 ms frames and is committed after a run of
silence, 22 frames (~700 ms) by default. `--silence-frames N` overrides the
endpointing delay (32 ms per frame); the desktop unit ships 20 (~640 ms) as a
compromise between perceived latency and not splitting words at breath pauses.
Consecutive phrases of one dictation are separated by a single
space: each phrase is transcribed on its own, so the junction is re-inserted
before the next phrase is typed. `Ctrl+C` stops capture cleanly. Use
`--output stdout` while testing to avoid typing into the focused application.

## Engines

`run` supports two local backends:

- `--engine whispercpp` (default) — GGML models through `pywhispercpp`.
- `--engine fasterwhisper` — CTranslate2 models through the optional `faster`
  extra (`pip install -e '.[faster]'`), for example:

```bash
.venv/bin/wayland-whisper run \
  --engine fasterwhisper \
  --faster-whisper-model ~/.local/share/nerd-dictation/models/faster-whisper-small \
  --vad-model ~/.local/share/nerd-dictation/models/silero-vad.onnx \
  --language de --faster-whisper-threads 4
```

In the 2026-10-10 CPU benchmark on this machine, faster-whisper small INT8
with 4 threads was ~1.4x faster than the whisper.cpp small default over 12
full German sentences (up to ~3x on a short-clip tuning set) with the same
word-error count. The client pins `OPENBLAS_NUM_THREADS=1` — the fw decode
output was measured to depend on it, while the whisper.cpp path is
unaffected — so manual runs should keep that environment. The daemon logs one
`inference engine=... elapsed=... audio=... chars=...` line per phrase;
`--phrase-dump DIR` additionally saves every finished phrase as a 16 kHz mono
WAV for offline A/B tests. `contrib/desktop/wayland-whisper-engine` switches
the user service between the two engines via a systemd drop-in (`show`,
`cpp`, `fw`, `reset`).

## Desktop integration (Sway)

`contrib/desktop/install.sh` installs the user service and the toggle used on
this machine:

- `wayland-whisper.service` — warm daemon, started with `--suspend-on-start`:
  the model stays loaded while the microphone stays closed.
- `wayland-whisper-toggle` — `master|on|off` binds/releases `Ctrl+Space` and
  the AhaKey pad Enter key in Sway and enables/disables the service; `mic`
  (also the default action when Sway runs the toggle) opens or closes the
  microphone only; `send` (bound to the pad Enter key while master is on)
  finishes the phrase being dictated, closes the microphone and presses
  Enter; `status` prints master, service, microphone state and PID.
- `Super+n` runs `wayland-whisper-toggle master`; `Ctrl+Space` runs the toggle
  with no argument. The microphone LED on the AhaKey pad breathes while the
  microphone is open (`ahakey.sh pulse breathing`) and is off otherwise;
  notifications are reserved for the master switch, never for the microphone.

`run --suspend-on-start --cookie <file>` writes its PID to `<file>` and waits;
`wayland-whisper resume --cookie <file>` sends SIGCONT to open the microphone;
SIGUSR1 closes it again while the process, the loaded model, and Whisper stay
warm — a phrase still being collected or transcribed is finished and typed
first, then the microphone closes.

## License Notes

This repository's code is MIT licensed. It runs external programs and Python
packages; their licenses remain their own. At the time this project was
created, pywhispercpp/whisper.cpp, ONNX Runtime, and Silero VAD publish MIT
licenses. Model weights can have additional terms, so verify the exact model
file before redistribution.
