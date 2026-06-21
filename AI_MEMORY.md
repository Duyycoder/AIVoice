# AI ARCHITECTURE MEMORY & RULEBOOK
**Project:** Single-Speaker Local Vietnamese TTS Framework (Extensible CLI & Web UI)
**Target Hardware:** Optimized for RTX 5060 (8GB VRAM) and fully compatible down to RTX 3060 (6GB VRAM) / Fallback CPU.

## 1. Core Principles (DO NOT VIOLATE)
* **Virtual Environment:** The project MUST use a Python virtual environment (`.venv`).
* **Offline First:** Models must run locally. Do NOT use Hugging Face auto-downloads during inference.
* **FP16 Mixed Precision (Autocast):** Coqui XTTSv2 uses `torch.amp.autocast` on GPU. Output is safely cast back to `float32` before writing to `soundfile`.
* **Zero-Reload In-Memory Fallback:** If FP16 inference fails or yields NaNs, the system automatically falls back to FP32 in-memory for that chunk only (no disk reloading).
* **TF32 Matmul Acceleration:** Enable `matmul.allow_tf32 = True` and `cudnn.allow_tf32 = True` globally once at startup if CUDA is active.
* **VRAM Serialization (Locking):** Flask backend must serialize GPU inference calls via a global `threading.Lock()` to prevent concurrent VRAM OOM.
* **Path Security & Sanitization:** Absolute external paths are allowed, but writing to critical system directories (`C:\Windows`, `C:\Program Files`) and project code/weight folders (`src/`, `models/`, `.venv/`, `.git/`, `configs/`, `tests/`) is strictly blocked.
* **Chunking Limit:** Text chunking limit (`max_words`) is set to `50` to improve prosody on modern GPUs while keeping VRAM safe.
* **Launchers:** Location-independent `.bat` files (`chay_giao_dien.bat`, `kiem_tra_gpu.bat`, `chay_kiem_thu.bat`) resolve the project root (`F:\programfiles\AIVoice`) if moved or run from the Desktop.

## 2. Global System Layout
All logic goes in `src/`, config files in `configs/`, models in `models/`, user files in `data/`, templates in `templates/`, and test assets in `tests/test_data/`.
* `src/engines/base.py`: Declares abstract TTS engine plugin interfaces.
* `main.py`: Command Line Interface controller (with argparse).
* `web_ui.py`: Flask Web application backend.

## 3. The Engines
* **PiperEngine (`src/engines/piper.py`):** Fast local ONNX synthesis. Supports CPU/GPU via onnxruntime execution providers.
* **CloneEngine (`src/engines/clone.py`):** Coqui XTTSv2 voice cloning. Integrates SDPA context, FP16 autocast, and zero-reload FP32 fallback.
* **EdgeEngine (`src/engines/edge.py`):** Microsoft Edge cloud neural TTS wrapper.
* **RVCEngine (`src/engines/rvc_engine.py`):** Voice-to-Voice RVC post-processing. Accepts custom `device` argument.

## 4. Helper Modules
* `src/utils/text.py`: Clean markdown, chunk paragraphs into 50-word punctuation-bound segments.
* `src/utils/audio.py`: Concatenate arrays, apply linear fade-in/out, target LUFS volume normalization.
* `src/utils/phoneme.py`: IPA phoneme translation using local `viphoneme`.
* `src/utils/local_ai_spice.py`: Rewrite text styles using Qwen GGUF.

## 5. Web UI Features & Endpoints
* `/`: Serves modern glassmorphism SPA.
* `/api/generate`: Runs background thread generation with serialization lock and logging queues.
* `/api/diagnose`: Runs GPU check health tool.
* `/api/models` / `/api/voices`: Discovery APIs for local models and cloud voices.
* `/api/audio`: Serves generated wav files securely.