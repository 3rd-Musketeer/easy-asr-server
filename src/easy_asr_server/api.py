"""
API Module for easy_asr_server

This module provides both backward compatibility and a high-level API
for easy integration and direct Python usage.
"""

import logging
from typing import Optional, Union
import numpy as np

# Re-export server functionality for backward compatibility
from .server import (
    app,
    app_state,
    lifespan,
    get_asr_engine,
    recognize_audio,
    health_check,
    get_hotwords,
    update_hotwords
)

# Re-export CLI functionality for backward compatibility
from .cli import (
    app_cli,
    run,
    download,
    main
)

# Import core components
from .model_manager import ModelManager, MODEL_CONFIGS, DEFAULT_PIPELINE
from .asr_engine import ASREngine
from .utils import setup_logging, resolve_device_string

logger = logging.getLogger(__name__)


class EasyASR:
    """
    High-level API for easy_asr_server that encapsulates all initialization complexity.
    
    This class provides a simple interface for speech recognition without requiring
    users to understand the internal ModelManager and ASREngine architecture.
    
    Example:
        >>> # Simple usage
        >>> asr = EasyASR()
        >>> result = asr.recognize("path/to/audio.wav")
        >>> print(result)
        
        >>> # With configuration
        >>> asr = EasyASR(pipeline="paraformer", device="cuda", hotwords="你好 世界")
        >>> result = asr.recognize(audio_array)
        >>> print(result)
    """
    
    def __init__(
        self, 
        pipeline: str = DEFAULT_PIPELINE,
        device: str = "auto", 
        hotwords: str = "",
        log_level: str = "WARNING",
        auto_init: bool = True
    ):
        """
        Initialize the EasyASR instance.
        
        Args:
            pipeline: ASR pipeline type ('sensevoice' or 'paraformer')
            device: Device for inference ('auto', 'cpu', 'cuda', etc.)
            hotwords: Space-separated hotwords string for better recognition
            log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
            auto_init: Whether to automatically initialize the ASR engine
            
        Raises:
            ValueError: If pipeline or device configuration is invalid
            RuntimeError: If ASR engine initialization fails
        """
        self.pipeline = pipeline
        self.device = device
        self.hotwords = hotwords
        self.log_level = log_level
        
        # Validate pipeline
        if pipeline not in MODEL_CONFIGS:
            raise ValueError(f"Invalid pipeline '{pipeline}'. Available: {list(MODEL_CONFIGS.keys())}")
        
        # Setup logging
        setup_logging(level=log_level)
        
        # Internal components
        self._model_manager = None
        self._asr_engine = None
        self._resolved_device = None
        self._initialized = False
        
        if auto_init:
            self.initialize()
    
    def initialize(self) -> bool:
        """
        Initialize the ASR engine and load the model.
        
        Returns:
            bool: True if initialization successful, False otherwise
            
        Raises:
            ValueError: If device configuration is invalid
            RuntimeError: If model loading fails
        """
        if self._initialized:
            logger.info("ASR engine already initialized")
            return True
            
        try:
            logger.info(f"Initializing EasyASR with pipeline='{self.pipeline}', device='{self.device}'")
            
            # Resolve device
            self._resolved_device = resolve_device_string(self.device)
            logger.info(f"Using resolved device: {self._resolved_device}")
            
            # Initialize ModelManager
            self._model_manager = ModelManager()
            
            # Load pipeline
            logger.info(f"Loading pipeline: {self.pipeline}")
            self._model_manager.load_pipeline(pipeline_type=self.pipeline, device=self._resolved_device)
            
            # Initialize ASREngine
            self._asr_engine = ASREngine(model_manager=self._model_manager)
            
            # Perform health check
            logger.info("Performing health check...")
            if not self._asr_engine.test_health():
                raise RuntimeError("ASR engine health check failed")
                
            self._initialized = True
            logger.info("EasyASR initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize EasyASR: {e}")
            self._cleanup()
            raise RuntimeError(f"ASR initialization failed: {e}") from e
    
    def recognize(
        self, 
        audio_input: Union[str, np.ndarray], 
        hotwords: Optional[str] = None
    ) -> str:
        """
        Recognize speech from audio input.
        
        Args:
            audio_input: Either a file path (str) or audio array (np.ndarray)
                        If np.ndarray, must be 1D float32 array at 16kHz
            hotwords: Optional hotwords string (overrides instance hotwords)
            
        Returns:
            str: The recognized text transcription
            
        Raises:
            RuntimeError: If ASR engine not initialized or recognition fails
            ValueError: If audio_input format is invalid
        """
        if not self._initialized:
            raise RuntimeError("ASR engine not initialized. Call initialize() first or set auto_init=True")
        
        # Use provided hotwords or fall back to instance hotwords
        final_hotwords = hotwords if hotwords is not None else self.hotwords
        
        try:
            result = self._asr_engine.recognize(audio_input, hotword=final_hotwords)
            return result
        except Exception as e:
            logger.error(f"Recognition failed: {e}")
            raise RuntimeError(f"Speech recognition failed: {e}") from e
    
    def is_healthy(self) -> bool:
        """
        Check if the ASR engine is healthy and ready for recognition.
        
        Returns:
            bool: True if healthy, False otherwise
        """
        if not self._initialized or self._asr_engine is None:
            return False
        return self._asr_engine.test_health()
    
    def get_info(self) -> dict:
        """
        Get information about the current ASR configuration.
        
        Returns:
            dict: Configuration information
        """
        return {
            "pipeline": self.pipeline,
            "device": self.device,
            "resolved_device": self._resolved_device,
            "hotwords": self.hotwords,
            "initialized": self._initialized,
            "healthy": self.is_healthy() if self._initialized else False
        }
    
    def _cleanup(self):
        """Clean up resources."""
        self._asr_engine = None
        self._model_manager = None
        self._initialized = False
    
    def __enter__(self):
        """Context manager entry."""
        if not self._initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()
    
    def __del__(self):
        """Destructor to clean up resources."""
        self._cleanup()


# Convenience functions for functional-style API
def create_asr_engine(
    pipeline: str = DEFAULT_PIPELINE,
    device: str = "auto",
    hotwords: str = "",
    log_level: str = "WARNING"
) -> EasyASR:
    """
    Create and initialize an ASR engine with the specified configuration.
    
    Args:
        pipeline: ASR pipeline type ('sensevoice' or 'paraformer')
        device: Device for inference ('auto', 'cpu', 'cuda', etc.)
        hotwords: Space-separated hotwords string
        log_level: Logging level
        
    Returns:
        EasyASR: Initialized ASR engine instance
        
    Raises:
        ValueError: If configuration is invalid
        RuntimeError: If initialization fails
    """
    return EasyASR(
        pipeline=pipeline,
        device=device, 
        hotwords=hotwords,
        log_level=log_level,
        auto_init=True
    )


def recognize(
    audio_input: Union[str, np.ndarray],
    pipeline: str = DEFAULT_PIPELINE,
    device: str = "auto",
    hotwords: str = "",
    asr_engine: Optional[EasyASR] = None
) -> str:
    """
    One-shot speech recognition function.
    
    Args:
        audio_input: Either a file path (str) or audio array (np.ndarray)
        pipeline: ASR pipeline type (ignored if asr_engine provided)
        device: Device for inference (ignored if asr_engine provided)
        hotwords: Space-separated hotwords string
        asr_engine: Optional pre-initialized ASR engine for reuse
        
    Returns:
        str: The recognized text transcription
        
    Raises:
        ValueError: If configuration or audio input is invalid
        RuntimeError: If recognition fails
    """
    if asr_engine is not None:
        return asr_engine.recognize(audio_input, hotwords=hotwords)
    else:
        # Create temporary engine for one-shot usage
        with EasyASR(pipeline=pipeline, device=device, hotwords=hotwords) as asr:
            return asr.recognize(audio_input)


# Export available pipelines and configurations
def get_available_pipelines() -> dict:
    """
    Get information about available ASR pipelines.
    
    Returns:
        dict: Pipeline configurations
    """
    return MODEL_CONFIGS.copy()


def get_default_pipeline() -> str:
    """
    Get the default ASR pipeline name.
    
    Returns:
        str: Default pipeline name
    """
    return DEFAULT_PIPELINE


# Re-export for backward compatibility
__all__ = [
    # High-level API
    "EasyASR",
    "create_asr_engine", 
    "recognize",
    "get_available_pipelines",
    "get_default_pipeline",
    
    # Server components (backward compatibility)
    "app",
    "app_state", 
    "lifespan",
    "get_asr_engine",
    "recognize_audio",
    "health_check",
    "get_hotwords", 
    "update_hotwords",
    
    # CLI components (backward compatibility)
    "app_cli",
    "run", 
    "download",
    "main",
    
    # Constants
    "DEFAULT_PIPELINE",
    "MODEL_CONFIGS"
]


# Entry point for running as module (python -m easy_asr_server.api)
if __name__ == "__main__":
    main()
