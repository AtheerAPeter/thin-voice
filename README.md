# thin-voice

A fully local voice assistant built to stay fast on a MacBook Air M1 with 8 GB of RAM.
You talk, it answers out loud in about half a second, and you can interrupt it.

```
mic (browser, echo cancelled)
  -> Silero VAD v6            is someone talking?                    ~1 ms per 32 ms frame
  -> Smart Turn v3.2          was that pause the end of the turn?     ~25 ms, once per pause
  -> Moonshine v2 streaming   transcribes while you talk              ~75 ms after you stop
  -> LFM2.5-1.2B (llama.cpp)  streams the answer                      first clause ~180 ms
  -> Pocket TTS               streams audio while it generates        first audio ~70 ms
speaker (browser)
```

## Run it

Needs `uv` and `brew install llama.cpp`. Models download on first start (about 2.5 GB).

```sh
uv run main.py                            # then open http://localhost:8765
uv run main.py --llm qwen3.5-2b           # smarter, ~180 ms slower, +600 MB
uv run main.py --voice-engine moonshine   # Kokoro voice (no PyTorch, but slower to start talking)
uv run main.py --language de --llm qwen3.5-2b
uv run bench.py                           # measure the whole pipeline without a microphone
```

German uses Pocket TTS's native German voice `juergen`. To clone another voice, accept the terms at
https://huggingface.co/kyutai/pocket-tts, run `uvx hf auth login`, and pass a clean 10-20 s recording, for example
`--voice voices/de-thorsten.wav` (Thorsten Müller, CC0, from Thorsten-Voice/TV-44kHz-Full).

Use headphones, or untick "Let me interrupt", if it keeps interrupting itself: that is the speaker leaking into the mic.
The page asks the browser for echo cancellation, but how well that covers the page's own playback has not been tested yet.

## Measured (M1 Pro, 4 threads, `bench.py`, median of 4 spoken questions)

Time from the end of your speech to the first audio, not counting the browser's ~20 ms playout:

| Setup | Reply after you stop | Memory (app + llm) |
| --- | --- | --- |
| **Pocket TTS + LFM2.5-1.2B** (default) | **548 ms** (max 599) | 2.4 GB |
| Pocket TTS + Qwen3.5-2B | 727 ms | 3.0 GB |
| Supertonic 3 + LFM2.5-1.2B | 1154 ms | 2.1 GB |
| Kokoro (via Moonshine) + LFM2.5-1.2B | 1258 ms | 2.0 GB |
| German: Pocket TTS + LFM2.5-1.2B, Moonshine small | 566 ms | 2.5 GB |

About 245 ms of every reply is the deliberate wait: 200 ms of silence before Smart Turn is asked.
Kokoro and Supertonic must render a whole sentence before playing anything; Pocket TTS streams, which is the whole difference.

**On an M1 Air (estimate, not measured):** same CPU cores, so VAD, STT and TTS stay about the same,
but the GPU has half the cores and a third of the memory bandwidth, so the LLM's first clause takes
roughly 2-3x longer. Expect about **0.8-0.9 s** with LFM2.5-1.2B. Memory fits: ~2.4 GB of 8 GB.

## Files

- `listen.py`: VAD, Smart Turn (with a numpy port of Whisper's log-mel features), Moonshine on its own thread, and the turn-taking state machine.
- `think.py`: streams from llama-server and cuts the reply into pieces: the first clause as soon as it has 4 words, then whole sentences.
- `speak.py`: the three voices behind one `stream(text, stop)` interface.
- `conversation.py`: wires it together, handles barge-in, and joins a turn the user wasn't done with to what they say next.
- `main.py`: starts llama-server, loads the models, serves `web/index.html`, the avatars and the `/talk` websocket.
- `web/index.html`: microphone capture, playback, the face and lip-sync, transcript and per-reply timing.
- `bench.py`: feeds questions spoken by a different voice through the real pipeline at real-time pace.

## Things learned building it

- Moonshine's transcription passes take up to 480 ms; running them on the VAD thread delayed end-of-turn detection by the same amount. They now run on their own thread.
- Pocket TTS's int8 mode uses more memory on Apple Silicon and is not faster. It only helps on x86.
- llama-server closes the connection after a streamed reply, so the HTTP client must not reuse it.
- llama-server defaults to 4 parallel slots, which quarters the context; `--parallel 1` keeps one slot with the full context and its prompt cache.
- Telling a 1.2B model to "keep the first sentence short" made it slower, not faster.
- Supertonic was archived in September 2026: it still works, but gets no fixes.
- Moonshine's German model is under a non-commercial licence.
- A funny, chatty persona costs accuracy on these small models: Qwen3.5-2B jokes better but gets facts wrong more often, and its German is weak; LFM2.5-1.2B stays factual but is barely funny.

## The face

Pick it in the dropdown under the face; the choice is kept in the page URL (`?face=...`).

- **Photo-real head (LAM):** a 3D Gaussian-splat head reconstructed from one photo by Alibaba's
  [LAM](https://github.com/aigc3d/LAM), rendered in WebGL by
  [gaussian-splat-renderer-for-lam](https://github.com/aigc3d/LAM_WebRender) (MIT). It moves on its own while
  idle, listening, thinking and speaking, blinks, and its mouth follows the reply audio. Ships with `james`
  (from [OpenAvatarChat](https://github.com/HumanAIGC-Engineering/OpenAvatarChat)'s samples, Apache-2.0).
  **Make your own:** upload a front-facing photo to the
  [LAM demo on ModelScope](https://www.modelscope.cn/studios/Damo_XR_Lab/LAM_Large_Avatar_Model), export the
  avatar for OpenAvatarChat, and drop the zip into `web/avatars/lam/`; it shows up in the dropdown by its file name.
  Only use a face you have the right to use. Head and shoulders only.
- **Animated picture:** any still picture of a cartoon-like character, brought to life by redrawing its eyes and mouth:
  the pupils look at you and drift up while it thinks, the lids blink, the mouth opens with the speech, and it breathes.
  Costs next to nothing to draw. Ships with `blue` (`web/avatars/pictures/blue.png`).
  **Add your own:** put the picture in `web/avatars/pictures/` with a `<name>.json` rig next to it, like `blue.json`:
  the picture's pixel positions of each eye white and pupil (as rotated ellipses), the mouth slit, a patch of plain skin
  above each eye for its eyelid, the colours of the eye white and the dark "ink", and which part to frame (`focusY`, `frameHeight`).
  Works for flat cartoon features, not for real faces; use a LAM head for those.
- **3D character (TalkingHead):** full upper body with hand gestures, from [TalkingHead](https://github.com/met4citizen/TalkingHead)
  (three.js, MIT). Two photo-based avatars ship in `web/avatars/`, Avaturn (female) and MetaPerson (male),
  both from TalkingHead's examples and **free for non-commercial use only**.

Lips are driven by [HeadAudio](https://github.com/met4citizen/HeadAudio) (MIT), which reads mouth shapes from
the sound itself, so it works with every voice and needs no word timings. For LAM heads the 15 mouth shapes are
mapped onto ARKit face controls in `VISEME_TO_ARKIT`. The audio is held back 80 ms so lips and sound line up.

Without a `?face=` choice, the page picks a face matching the voice: the server measures the voice's pitch at
startup (below ~160 Hz male, above female), because voice names lie: Pocket's "alba" is a male voice.
Male voices get the photo-real `james`, female voices the Avaturn character until a photo-real female head is added.
Female Pocket voices: anna, vera, fantine, eponine, azelma, mary, jane, eve, cosette. Male: alba, javert, jean, charles, paul, george, michael.

All face libraries load from cdn.jsdelivr.net. Pick "No face" to save the GPU work on a weak machine.
More realistic than this (photoreal neural video like MuseTalk, Ditto, LivePortrait) does not fit in 8 GB next to everything else.

## Next: meetings

Show the page through OBS Virtual Camera and route its audio through BlackHole as the meeting mic,
or have a Chrome extension swap `getUserMedia` for the page's canvas and audio.
An AI participant must say it is one (EU AI Act Art. 50, since August 2026), and Meet and Zoom now gate bots.

[miniface](https://github.com/minifaceorg/miniface-facial-motion-capture) tracks a human face from a webcam,
so it can't drive an AI's face by itself, but its three.js + ARKit avatar code is reusable.
