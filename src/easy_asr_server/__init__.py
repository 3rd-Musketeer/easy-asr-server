"""
easy_asr_server - A simple high-concurrency speech recognition service based on FunASR

This package provides both server-based and direct Python API for speech recognition.
"""

__version__ = "0.1.0"

# Import high-level API (recommended for most users)
from .api import (
    EasyASR,
    create_asr_engine,
    recognize,
    get_available_pipelines,
    get_default_pipeline
)

# Import core components (for advanced users)
from .model_manager import ModelManager, MODEL_CONFIGS, DEFAULT_PIPELINE
from .asr_engine import ASREngine
from .utils import (
    process_audio, 
    save_audio_to_file,
    get_audio_duration,
    AudioProcessingError,
    setup_logging,
    read_audio_bytes,
    resolve_device_string
)

# Public API to be exported
__all__ = [
    # High-level API (recommended)
    "EasyASR",
    "create_asr_engine", 
    "recognize",
    "get_available_pipelines",
    "get_default_pipeline",
    
    # Core components (advanced usage)
    "ModelManager",
    "ASREngine", 
    "MODEL_CONFIGS",
    "DEFAULT_PIPELINE",
    
    # Utilities
    "process_audio",
    "save_audio_to_file",
    "get_audio_duration",
    "AudioProcessingError",
    "setup_logging",
    "read_audio_bytes",
    "resolve_device_string",
]
