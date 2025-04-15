# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2024-07-28
### Added
- Introduced `ModelManager` for handling multiple ASR pipelines (e.g., `sensevoice`, `paraformer`) and their components (VAD, ASR, Punctuation).
- Added command-line option `--pipeline` to select the ASR core pipeline at startup.
- Added API endpoints for hotword management: `GET /asr/hotwords` to retrieve and `PUT /asr/hotwords` to update the list.
- Implemented hotword support for the `paraformer` pipeline, configurable via `--hotword-file` option or the hotword API endpoints.
- Added new performance testing script `src/example/performance_test.py` using `requests`, `numpy`, and `rich`.
- Added documentation for selecting pipelines, using hotwords, and running the new performance test in `README.md`.
- Added utility function `utils.read_hotwords` to load hotwords from a file.
- Added comprehensive unit and integration tests for `ModelManager`, `ASREngine`, API endpoints (including hotwords), and utilities.
- Added support for VAD and Punctuation Restoration as part of ASR pipelines managed by `ModelManager`.

### Changed
- **Breaking:** Switched CLI framework from `click` to `typer`. Command-line invocation might differ slightly.
- **Breaking:** `/asr/recognize` endpoint now returns JSON `{"text": "..."}` instead of plain text for successful recognitions.
- Refactored `ASREngine` to delegate model loading, path management, and device handling to `ModelManager`.
- Refactored API (`main.py`) to use a global state dictionary instead of global variables and initialize `ModelManager` and `ASREngine` during startup.
- Updated `utils.process_audio` to save processed audio to a temporary file before passing to the ASR engine.
- Merged performance testing dependencies (`requests`, `numpy`, `rich`) into main project dependencies, removing the `perf-test` optional group.
- Updated all tests (`test_api.py`, `test_asr_engine.py`, `test_model_manager.py`, `test_utils.py`) to reflect refactoring and new features, including extensive use of mocking and dependency overrides.

### Removed
- Removed old performance testing scripts and related files (`run_performance_test.sh`, `visualize_results.py`, `asr_request.py`, etc.).
- Removed unused `ASREngineError` exception.
- Removed direct `funasr.AutoModel` usage from `ASREngine`.

### Fixed
- Ensured `pyproject.toml` and `README.md` accurately reflect current dependencies, features, and usage instructions.

## [0.1.0] - 2024-04-04
### Added
- Initial release of easy-asr-server.
- Speech recognition API (`/asr/recognize`, `/asr/health`) with FastAPI based on FunASR `AutoModel`.
- Command-line interface using Click for basic configuration (host, port, workers, device).
- High-concurrency support with multiple Uvicorn workers.
- Basic unit tests for core functionality.
- Apache License 2.0.

[Unreleased]: https://github.com/3rd-Musketeer/easy-asr-server/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/3rd-Musketeer/easy-asr-server/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/3rd-Musketeer/easy-asr-server/releases/tag/v0.1.0 