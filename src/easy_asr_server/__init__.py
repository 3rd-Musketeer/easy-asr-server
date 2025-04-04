"""
easy_asr_server - A simple high-concurrency speech recognition service based on FunASR
"""

__version__ = "0.1.0"

# Import key components
from .model_manager import ModelManager
from .utils import (
    process_audio, 
    save_audio_to_file,
    get_audio_duration,
    AudioProcessingError,
    setup_logging
)
from .asr_engine import ASREngine, ASREngineError

# Public API to be exported
__all__ = [
    "ModelManager",
    "process_audio",
    "save_audio_to_file",
    "get_audio_duration",
    "AudioProcessingError",
    "setup_logging",
    "ASREngine",
    "ASREngineError",
]
