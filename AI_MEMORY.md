# AI ARCHITECTURE MEMORY & RULEBOOK
**Project:** Single-Speaker Local Vietnamese TTS Framework (Extensible CLI-Driven)
**Goal:** A modular, offline-first TTS engine tailored for Vietnamese, supporting plug-and-play engines (Piper TTS, XTTSv2, Edge-TTS) and controllable via CLI, following Clean Architecture principles.

## 1. Core Principles (DO NOT VIOLATE)
* **Virtual Environment:** The project MUST use a Python virtual environment (`.venv`).
* **Offline First:** Models must run locally. Do NOT use Hugging Face auto-downloads during inference.
* **Memory Efficiency:** Use PyTorch >= 2.1.1 SDPA natively. DO NOT include `flash-attn` in `requirements.txt`.
* **Chunking is Mandatory:** Long text must be chunked by sentences/punctuation before passing to any engine to prevent OOM errors on 6GB VRAM.
* **Dynamic Naming:** Output audio files MUST take the base name of the input `.md` file.
* **Clean Architecture Layout:** All logic goes in `src/`, config files in `configs/`, models in `models/`, user files in `data/`, and test assets in `tests/test_data/`.

## 2. Global System Architecture
The system follows an Object-Oriented, Plugin-style architecture under `src/`.
* `src/engines/base.py`: Contains the abstract class `BaseTTSEngine` with the method `generate(self, text: str, output_path: str, **kwargs) -> bool`.
* `main.py`: The CLI controller (using `argparse`). Routes to the correct engine class.

## 3. The Engines
* **PiperEngine (`src/engines/piper.py`):** For fast, lightweight Vietnamese reading (e.g., `vi_VN-vais1000-medium` or `vi_VN-vivos-x_low`).
* **CloneEngine (`src/engines/clone.py`):** For zero-shot voice cloning using XTTSv2 (Coqui TTS). Requires `--ref_audio`.
* **EdgeEngine (`src/engines/edge.py`):** Microsoft Edge online TTS for high-quality cloud-generated Vietnamese voices.

## 4. Helper Modules
* `src/utils/text.py`: Strips markdown and splits text into chunks.
* `src/utils/audio.py`: Handles audio array concatenation, linear fade-in/fade-out, and LUFS normalization.
* `src/utils/phoneme.py`: Converts Vietnamese text to IPA phonemes using `viphoneme` library.

## 5. CLI Arguments (Customization Interface)
The `main.py` exposes these arguments via `argparse`:
* `--input`: Path to the input `.md` or `.txt` file.
* `--input_dir`: Path to input directory for batch processing.
* `--engine`: Selection of the engine (`piper`, `clone`, `edge`).
* `--model`: Local path to specific model weights (e.g., `models/piper/vi_VN-vais1000-medium.onnx`).
* `--speed`: Float value to control speech rate (e.g., `1.0`, `1.2`).
* `--voice`: Name of the voice (e.g., `vi-VN-NamMinhNeural` or lang code `vi`, `en`).
* `--ref_audio`: Path to the reference `.wav` file (for CloneEngine, defaults to `data/voices/ref_voice.wav`).
* `--config`: Path to customization configuration file (defaults to `configs/default.json`).
* `--output_dir`: Path to base output folder (defaults to `data/outputs`).
* `--output_name`: Custom filename for output audio file.
* `--phonemize` / `--no-phonemize`: Enable/disable IPA phoneme translation.
* `--normalize` / `--no-normalize`: Enable/disable LUFS volume normalization.
* `--target_lufs`: Target LUFS value (defaults to `-14.0`).
* `--fade_in`: Duration of linear fade-in (defaults to `0.1`s).
* `--fade_out`: Duration of linear fade-out (defaults to `0.1`s).
* `--silence_duration`: Silence gap between chunks (defaults to `0.3`s).