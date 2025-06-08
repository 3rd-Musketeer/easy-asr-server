"""
ASR Engine Module for easy_asr_server

This module defines the ASREngine class which handles the core ASR processing
using a pre-configured pipeline provided by the ModelManager.
"""

import logging
import torch
import numpy as np
from typing import Any, Dict, List, Optional
import tempfile
import os
import torchaudio

# Removed AutoModel import
# from funasr import AutoModel 
from .model_manager import ModelManager # Added import

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ASREngine:
    """
    Handles Automatic Speech Recognition using a pipeline loaded by ModelManager.
    Receives audio data, passes it to the ModelManager's generate method,
    and returns the recognition results.
    """

    # Removed internal model loading and device handling
    # def __init__(self, asr_model_path: str, vad_model_path: str, punc_model_path: str, device: str):
    def __init__(self, model_manager: ModelManager):
        """
        Initialize the ASR Engine.
        Args:
            model_manager: An initialized instance of ModelManager with a loaded pipeline.
        """
        logger.info("Initializing ASREngine...")
        self.model_manager = model_manager
        # The actual pipeline instance is now managed within ModelManager
        # self.device = device
        # self.model = self._load_model(asr_model_path, vad_model_path, punc_model_path, device)
        logger.info("ASREngine initialized.")

    # Removed internal model loading method
    # def _load_model(self, asr_model_path, vad_model_path, punc_model_path, device):
    #     logger.info(f"Loading FunASR AutoModel...")
    #     logger.info(f"  ASR Model Path: {asr_model_path}")
    #     logger.info(f"  VAD Model Path: {vad_model_path}")
    #     logger.info(f"  PUNC Model Path: {punc_model_path}")
    #     logger.info(f"  Device: {device}")
    #     try:
    #         model = AutoModel(
    #             model=asr_model_path,
    #             vad_model=vad_model_path,
    #             punc_model=punc_model_path,
    #             device=device
    #             # Consider adding other relevant AutoModel params if needed
    #         )
    #         logger.info("FunASR AutoModel loaded successfully.")
    #         return model
    #     except Exception as e:
    #         logger.error(f"Failed to load FunASR AutoModel: {str(e)}", exc_info=True)
    #         raise RuntimeError(f"Could not load the ASR model pipeline.") from e

    def recognize(self, audio_input, hotword: str = "") -> str:
        """
        Recognize speech from audio input.

        Args:
            audio_input: Either a file path (str) or audio array (np.ndarray).
                        If np.ndarray, must be 1D float32 array at 16kHz.
            hotword: Optional space-separated string of hotwords.

        Returns:
            The recognized text transcription.
        Raises:
            ValueError: If audio_input format is invalid.
            RuntimeError: If the underlying model manager fails during generation.
        """
        logger.info("Received audio for recognition.")
        
        # Input validation and type checking
        if isinstance(audio_input, str):
            # File path input (existing functionality)
            input_description = f"file: {os.path.basename(audio_input)}"
            
        elif isinstance(audio_input, np.ndarray):
            # Numpy array input (new functionality)
            if audio_input.ndim != 1:
                raise ValueError(f"Audio array must be 1-dimensional, got shape: {audio_input.shape}")
            
            if audio_input.dtype != np.float32:
                raise ValueError(f"Audio array must be float32, got dtype: {audio_input.dtype}")
            
            if len(audio_input) < 512:
                raise ValueError(f"Audio array too short: {len(audio_input)} samples (minimum 512)")
            
            input_description = f"numpy array: {audio_input.shape} samples"
            
        else:
            raise TypeError(f"audio_input must be str or np.ndarray, got: {type(audio_input)}")
        
        try:
            # Delegate the actual generation to the ModelManager
            # Pass the hotword string along
            result = self.model_manager.generate(input_audio=audio_input, hotword=hotword)
            logger.info(f"Recognition successful for input: {input_description}")
            return result
        except Exception as e:
            logger.error(f"Error during ASR processing: {str(e)}", exc_info=True)
            # Re-raise as a runtime error to be caught by the API layer
            raise RuntimeError(f"ASR generation failed: {str(e)}") from e

    def test_health(self) -> bool:
        """
        Perform a quick health check of the ASR engine.
        Tries to process a short silent audio segment saved to a temp file.

        Returns:
            bool: True if the engine appears healthy, False otherwise.
        """
        temp_audio_path: Optional[str] = None
        try:
            logger.info("Performing ASREngine health check...")
            # Create a short silent audio signal
            sample_rate = 16000
            duration_ms = 100
            silent_audio_tensor = torch.zeros((1, int(sample_rate * duration_ms / 1000)), dtype=torch.float32)
            
            # Save the silent audio to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_f:
                temp_audio_path = tmp_f.name
            torchaudio.save(temp_audio_path, silent_audio_tensor, sample_rate)
            logger.debug(f"Saved silent audio for health check to: {temp_audio_path}")
            
            # Call recognize with the path to the silent audio file
            _ = self.recognize(audio_input=temp_audio_path)
            logger.info("ASREngine health check passed.")
            return True
        except Exception as e:
            logger.error(f"ASREngine health check failed: {str(e)}", exc_info=True)
            return False
        finally:
            # Clean up the temporary file created for the health check
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.unlink(temp_audio_path)
                    logger.debug(f"Deleted health check temp file: {temp_audio_path}")
                except OSError as unlink_error:
                     logger.warning(f"Could not delete health check temp file '{temp_audio_path}': {unlink_error}")

    # Removed detect_device static method (responsibility moved or handled at startup)
    # @staticmethod
    # def detect_device(preferred_device: str = "auto") -> str:
    #     # ... (previous device detection logic)
