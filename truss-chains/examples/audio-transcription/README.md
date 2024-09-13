# Audio Transcription Chain

This chain can transcribe any media files that are supported by the
[ffmpeg](https://ffmpeg.org/) library and that are hosted under a URL that
supports range downloads. Very large files are supported in near-constant time
using chunking.

More details are described in the
[guide](https://docs.baseten.co/chains/examples/audio-transcription) (note that the docs
will be moved soon and the link might need to be updated).

To separate the development of the chain business logic and the low level
whisper transcription model (which has slower deployment times), they are in
the current setup deployed separately:

```bash
truss chains push whisper_chainlet.py
```

Insert the predict URL for the Whisper Chainlet (printed by above push
command or can be found on the status page) as a value for
`WHISPER_PREDICT_URL` in `transcribe.py`. Then push the transcribe chain.

```bash
truss chains push transcribe.py
```

An example local invocation of the chain is given in the main-section of
`transcribe.py`.

The corresponding HTTP request (e.g. for invocation with cURL) has a top-level
json payload that combines to two arguments of `transcribe_job.run_remote`,
`media_url` and `params` (the JSON representation
of the pydantic model), together into one dictionary, i.e.:

```json
{
  "media_url": "http://commondatastorage.googleapis.com/gtv-videos-bucket/sample/TearsOfSteel.mp4",
  "params": {
    "wav_sampling_rate_hz": 16000,
    "macro_chunk_size_sec": 300,
    "macro_chunk_overlap_sec": 0,
    "micro_chunk_size_sec": 25,
    "silence_detection_smoothing_num_samples": 1600
  }
}
```

NOTE: `adapter.py` is a use case specific extension, that is not relevant for
the general chain functionality.
