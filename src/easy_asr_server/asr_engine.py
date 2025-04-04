"""
ASR Engine Module for easy_asr_server

This module encapsulates the functionality of FunASR's AutoModel for speech recognition.
It handles the initialization of the ASR and VAD models, device detection,
and provides methods for speech recognition.
"""

import os
import logging
import tempfile
import re
from typing import Dict, Optional, Tuple, Union, Any

import torch
import torchaudio
from funasr import AutoModel

from .model_manager import ModelManager, DEFAULT_ASR_MODEL_ID
from .utils import AudioProcessingError, save_audio_to_file

# Configure logging
logger = logging.getLogger(__name__)


class ASREngineError(Exception):
    """Exception raised for errors in the ASR engine."""
    pass


class ASREngine:
    """
    Encapsulates FunASR's AutoModel for speech recognition.
    Handles the initialization of models and provides recognition functionality.
    """
    
    def __init__(self, asr_model_path: str, vad_model_path: str, device: str = "auto"):
        """
        Initialize the ASR engine with the specified model and device.
        
        Args:
            asr_model_path: Path to the ASR model (SenseVoiceSmall)
            vad_model_path: Path to the VAD model
            device: Device to use for inference ('auto', 'cpu', 'cuda'). 
                   If 'auto', will use CUDA if available, else CPU.
        """
        self.asr_model_path = asr_model_path
        self.vad_model_path = vad_model_path
        
        # Determine the device to use
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing ASR engine with device: {self.device}")
        
        try:
            # Initialize the FunASR AutoModel
            self.model = AutoModel(
                model=asr_model_path,
                device=self.device,
                vad_model=vad_model_path,
                language='auto',
            )
            
            logger.info(f"ASR engine initialized successfully with model at {asr_model_path} and VAD model at {vad_model_path}")
            
            # Run a quick health check
            self._run_health_check()
            
        except Exception as e:
            logger.error(f"Failed to initialize ASR engine: {str(e)}")
            raise ASREngineError(f"Failed to initialize ASR engine: {str(e)}")
    
    def _run_health_check(self):
        """
        Run a quick health check to ensure the model is loaded correctly.
        
        Raises:
            ASREngineError: If the health check fails
        """
        try:
            # Create a short silent audio sample for testing
            silent_audio = torch.zeros(1, 16000).to(self.device)  # 1 second of silence
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp_path = tmp.name
            
            torchaudio.save(tmp_path, silent_audio.cpu(), 16000)
            
            try:
                # Try to run inference
                _ = self.model.generate(input=tmp_path)
                logger.info("ASR engine health check passed")
            finally:
                # Clean up the temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"ASR engine health check failed: {str(e)}")
            raise ASREngineError(f"ASR engine health check failed: {str(e)}")
    
    def _clean_asr_output(self, text: str) -> str:
        """
        Clean special tags from ASR output.
        
        Args:
            text: Raw text from ASR model
            
        Returns:
            str: Cleaned text without special tags
        """
        # Remove tags like <|en|>, <|EMO_UNKNOWN|>, <|Speech|>, <|woitn|>
        cleaned_text = re.sub(r'<\|[^|]*\|>', '', text)
        return cleaned_text.strip()
    
    def recognize(self, waveform: torch.Tensor, sample_rate: int) -> str:
        """
        Perform speech recognition on the given audio waveform.
        
        Args:
            waveform: Audio waveform as a tensor (shape: [1, samples])
            sample_rate: Sample rate of the audio (should be 16000)
            
        Returns:
            str: The recognized text
            
        Raises:
            ASREngineError: If recognition fails
        """
        # Save the waveform to a temporary file for FunASR
        try:
            tmp_path = save_audio_to_file(waveform, sample_rate)
            
            try:
                # Run speech recognition
                result = self.model.generate(input=tmp_path)
                
                # Extract text from result
                raw_text = None
                if isinstance(result, list) and len(result) > 0:
                    if isinstance(result[0], dict) and "text" in result[0]:
                        raw_text = result[0]["text"]
                    elif isinstance(result[0], str):
                        raw_text = result[0]
                    
                if raw_text is None:
                    # If we reach here, we couldn't extract text in a standard way
                    logger.warning(f"Unexpected result format from ASR engine: {result}")
                    raw_text = str(result)
                
                # Clean the output text
                cleaned_text = self._clean_asr_output(raw_text)
                logger.debug(f"Original ASR output: '{raw_text}', Cleaned: '{cleaned_text}'")
                
                return cleaned_text
                
            finally:
                # Clean up temporary file
                os.unlink(tmp_path)
                
        except Exception as e:
            logger.error(f"ASR recognition failed: {str(e)}")
            raise ASREngineError(f"ASR recognition failed: {str(e)}")
    
    def test_health(self) -> bool:
        """
        Test if the ASR engine is healthy.
        
        Returns:
            bool: True if the engine is healthy, False otherwise
        """
        try:
            self._run_health_check()
            return True
        except Exception:
            return False
    
    @classmethod
    def create(cls, device: str = "auto") -> "ASREngine":
        """
        Create an ASR engine with default models.
        
        Args:
            device: Device to use for inference ('auto', 'cpu', 'cuda')
            
        Returns:
            ASREngine: Initialized ASR engine
        """
        # Download and get paths for models
        model_manager = ModelManager()
        model_paths = model_manager.ensure_models_downloaded()
        
        # Create and return the engine
        return cls(
            asr_model_path=model_paths["asr"],
            vad_model_path=model_paths["vad"],
            device=device
        )
