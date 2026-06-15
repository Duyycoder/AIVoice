# Project Plan & Architectural Specification - AIVoice Clean Architecture

This document serves as the implementation plan, status register, and future roadmap of the **AIVoice** project, which has been restructured under the **Clean Architecture** principles.

---

## 1. Project Architecture Blueprint

The project is structured to separate concern and responsibility into distinct modular layers:

```text
AIVoice/
├── configs/               # System default options and setups (JSON)
├── data/                  # User inputs, outputs, and voice templates
├── models/                # Local model checkpoints (ONNX, PyTorch)
├── src/                   # Source core logics
│   ├── engines/           # TTS adapter plugins (Base, Edge, Piper, XTTSv2)
│   └── utils/             # Helper libraries (audio processing, text normalization)
├── tests/                 # Isolated integration tests and test data
└── main.py                # Single-point control wizard
```

### Component Details
* **`src/engines/base.py`**: Declares abstract `BaseTTSEngine` with a unified `generate()` method signature.
* **`src/engines/edge.py`**: Leverages Microsoft Edge Neural TTS online web API.
* **`src/engines/piper.py`**: Fast local CPU/GPU speech synthesis using Piper ONNX. Housed with both Python in-process synthesis and subprocess fallback.
* **`src/engines/clone.py`**: Local zero-shot voice cloning using Coqui XTTSv2, optimized via PyTorch SDPA kernel context.
* **`src/utils/text.py`**: Strips markdown formatting and divides long paragraphs into sentence chunks based on punctuation terminals.
* **`src/utils/audio.py`**: Concatenates audio segments with adjustable silence spaces, normalizes volume to professional broadcast standard (e.g. -14 LUFS using BS.1770 meters), and applies linear fade-in/out to prevent pop noise.
* **`src/utils/phoneme.py`**: Bypasses OS-compatibility issues on Windows to convert raw Vietnamese text into IPA phonemes via `viphoneme`.

---

## 2. Dev Roadmap & Implementation Progress

- [x] **Phase 1: Foundation & Adapters**
  - [x] Create modular base classes and implement Microsoft Edge TTS wrapper.
  - [x] Create Piper engine adapter supporting custom model paths and speeds.
  - [x] Create XTTSv2 Voice Cloning engine adapter with GPU support.
- [x] **Phase 2: Post-Processing & Text Processing**
  - [x] Write regex-based markdown cleaner and sentence segmenter.
  - [x] Write audio composer with padding silence, LUFS normalization, and linear fades.
  - [x] Integrate optional Vietnamese phonemizer fallback.
- [x] **Phase 3: Interactive Wizard & Configuration Defaults**
  - [x] Implement config loader supporting override options.
  - [x] Develop terminal interactive setup wizard.
- [x] **Phase 4: Clean Architecture Restructuring**
  - [x] Group python source files inside `src/`.
  - [x] Create global `data/` folder and `configs/` folder.
  - [x] Update imports, local relative checks, and test runner configurations.
- [ ] **Phase 5: Future Enhancements (Roadmap)**
  - [ ] Add Multi-Speaker Markdown script parser (splitting text into dialogues by speaker names).
  - [ ] Implement support for batch processing multi-speaker scripts.
  - [ ] Provide web GUI interface (Gradio or Next.js local server).

---

## 3. Verification & CI/CD Strategy

* **Local Integration Tests:** Run the test suite using:
  ```powershell
  .venv\Scripts\python.exe tests/run_tests.py
  ```
  The test suite executes 10 separate scenarios (Edge online, Piper offline, XTTSv2 cloning, batch processing, config override, etc.) using isolated files in `tests/test_data/`.
* **Zero Regression Rule:** Any new change must pass all 10 test suite scenarios before merging/publishing.
