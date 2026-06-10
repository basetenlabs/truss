# Voice clone reference audio

Drop reference audio here to register cloned voices with the Qwen3-TTS service.

## Naming convention

Each audio file becomes a server-side voice whose name is the file's basename.
A sibling `.txt` file (same basename) is uploaded as the reference transcript.

```
data/voices/
├── dan.wav         # voice name: "dan"
├── dan.txt         # optional: transcript for "dan"
├── alice.wav       # voice name: "alice"
└── alice.txt       # optional: transcript for "alice"
```

The active voice in the running pipeline is whichever name is set as
`TTS_KWARGS["voice"]` in `model/model.py`. Other voices in this directory
are still uploaded so they can be selected at runtime — for example, a
client could swap voices via `session.update`.

Supported audio extensions: `wav`, `mp3`, `flac`, `ogg`. Sample rate is
unconstrained — the server resamples internally. 5–30s of clean reference
speech per file is a good starting point.

## How the upload works

On the first TTS warmup (per process) the service:

1. Opens an out-of-band control WebSocket to the Baseten deployment.
2. Sends `voice.list` once to enumerate already-registered voices.
3. For every audio file in this directory whose basename is **not** already
   registered server-side, base64-encodes the bytes and sends `voice.add`
   with `consent: "user_consent"` (override via `TTS_KWARGS["consent"]`).
4. Closes the control socket. Subsequent `session.config` messages reference
   each voice purely by name.

The upload is one-shot per process. To force a re-upload (e.g. after editing
or replacing reference audio), restart the agent. The server-side voice will
be reused by name on the next launch.

## Falling back to built-in voices

If you remove all files from `data/voices/` (or set `voice_audio_dir=None`
in `TTS_KWARGS`), the service skips bulk upload and forwards the configured
voice name straight to `session.config`, letting the server pick the matching
built-in voice (Vivian, Chelsie, etc.).

## Privacy

Files in this directory are **not** committed to source control by default
because of the `.gitignore` here. Replace it if you intentionally want to
ship a reference voice with the repo.
