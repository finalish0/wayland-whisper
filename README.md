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

`run` supports three backends:

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
the user service between the engines via a systemd drop-in (`show`, `cpp`,
`fw`, `remote`, `reset`).

### Remote engine (the GPU on `an`)

`--engine remote` sends the recorded audio to the **admission gateway** on the
GPU host and gets the transcript back. The transcription runs on the
RX 7600 XT there, using `whisper-large-v3-turbo` (German, q5_0).

```bash
.venv/bin/wayland-whisper run \
  --engine remote \
  --remote-url http://192.168.188.51:8888/v1/audio/transcriptions \
  --remote-model whisper-large-v3-turbo \
  --vad-model ~/.local/share/nerd-dictation/models/silero-vad.onnx \
  --language de
```

The bearer token is read from the file named by
`WHISPER_REMOTE_API_KEY_FILE` (default
`~/.config/wayland-whisper/remote-key`, mode 0600). It is the gateway's own
token — the same one llama-swap uses. On the machine that runs this client,
the token is provisioned by Ansible (`wayland_whisper_remote` role in
`phcx-Server-Setup`), not copied by hand.

**Always go through the gateway, never straight to llama-swap.** The gateway
carries the VRAM reservation: the admission lock, the eviction of competing
models, the preflight and the per-model start budget. A direct request to
llama-swap on port 8889 skips all of it, and the speech child and the TTS
model then evict each other. Port 8888 is the only public entry.

### Automatic fallback

`--engine remote` with fallback options keeps the dictation alive when the
GPU host is unreachable:

```bash
--fallback-model ~/.local/share/nerd-dictation/models/faster-whisper-small \
--fallback-compute int8 --fallback-threads 4
```

The rules:

- The remote engine is tried for **every** phrase. The host may come back,
  so the client must not get stuck on the local path.
- The local engine is built **lazily**, on the first remote failure only.
  Loading faster-whisper costs about 3 GB of RAM and seconds of start time; a
  healthy remote path must not pay for it.
- If both engines fail, the error names both, so a lost phrase is visible in
  the journal instead of looking like silence.

The journal line names the engine that actually did the work
(`inference engine=remote ...` or `engine=local ...`).

### Measured behaviour (2026-10-22)

- Remote: 0.77 % word error rate over 12 FLEURS German clips (261 words),
  median 2.07 s; live dictation phrases took 0.31–0.57 s.
- Local fallback (faster-whisper small, INT8, 4 threads): 6.90 % WER, 1.73 s
  warm; 5.72 s on the first fallback including the model load.
- A healthy remote path costs about 114–120 MB of resident memory. The local
  model is not loaded at all.

The whisper child on the GPU host stays resident, so the first phrase after a
long pause does not pay a model load — except right after a restart of the
scheduler, where the cold start (about 30 s) falls into the first request.

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
