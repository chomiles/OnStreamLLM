# OnStreamLLM

OnStreamLLM is a Windows desktop tool for live audio and screen translation. It can translate input audio, loopback/output audio, and selected screen regions, then show captions locally, in a popup, or through an OBS browser overlay.

## Recommended Setup

For gaming or streaming, use **Preset 1: SenseVoice CPU / Hy-MT2 GPU**.

- STT: SenseVoice 2024 on CPU
- Translation: Hy-MT2 on NVIDIA GPU
- STT CPU threads: start with 2, increase to 3 or 4 if the PC has headroom
- This split keeps speech recognition away from the GPU while the translation model uses GPU acceleration.

Qwen presets are heavier:

- Qwen 4B: heavy, recommended for 12GB+ VRAM when not gaming
- Qwen 8B: very heavy, recommended for 16GB+ VRAM when not gaming

## First Launch

Portable build:

```bat
OnStreamLLM.exe
```

Source run:

```bat
run.bat
```

The app downloads required runtime libraries and models from inside the UI. They are not bundled into the release.

## Required Libraries

Open **Model Management** and install the libraries from the top **Required Library Download** section.

Required order:

```text
Torch -> Qwen ASR -> llama -> SenseVoice
```

After the required libraries are installed, restart the app once.

PaddleOCR is optional. It is installed separately when screen translation is first used.

## Main Menus

- **Translator**: start/stop engine, input detection, output detection, screen translation, one-line translator.
- **Per-Channel Translation / Device Settings**: source/target languages for input audio, output audio, and screen translation. Input/output devices are selected here.
- **Model Management**: required libraries, model downloads, presets, CPU/GPU devices, CPU thread assignment.
- **Settings**: caption style, popup, OBS overlay, server/remote connection.
- **Info**: app information.

## Model Notes

- SenseVoice model: `csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17`
- SenseVoice is treated as CPU-only to avoid GPU runtime conflicts.
- Hy-MT2 is recommended on GPU.
- Large Qwen models can take time to load. Wait for the status bar to report that model loading is complete.

## Screen Translation

Select the subtitle/chat region before enabling screen translation. Switching OCR languages such as Korean and Japanese reloads the OCR engine, so wait for the transition countdown before enabling screen translation again.

If PaddleOCR is not installed, the app asks whether to install it when you enable screen translation.

## Manuals

The full Korean, English, and Japanese manual is included in [Readme.txt](Readme.txt).
