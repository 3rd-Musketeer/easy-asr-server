"""
Utility module for easy_asr_server

This module provides utility functions for audio processing, logging setup, 
and error handling. The main function is process_audio, which validates
and converts audio to a format suitable for the ASR engine.
"""

import os
import io
import logging
import tempfile
import traceback
from typing import Union, Tuple, BinaryIO, Optional
from pathlib import Path

import numpy as np
import torchaudio
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
REQUIRED_SAMPLE_RATE = 16000  # 16kHz
REQUIRED_CHANNELS = 1  # Mono


class AudioProcessingError(Exception):
    """Exception raised for errors during audio processing."""
    pass


def setup_logging(level: int = logging.INFO) -> None:
    """
    Configure the logging for the application.
    
    Args:
        level: The logging level (default: INFO)
    """
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def is_valid_audio_file(file_data: BinaryIO) -> bool:
    """
    Check if the given file data is a valid audio file.
    
    Args:
        file_data: The file data to check
        
    Returns:
        bool: True if the data appears to be a valid audio file, False otherwise
    """
    try:
        # Try to read the audio info without loading the entire file
        # Save the file to a temp location for reading with torchaudio
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp:
            tmp.write(file_data.read())
            tmp_path = tmp.name
        
        try:
            # Seek back to the beginning of the file for later processing
            file_data.seek(0)
            
            # Try to get audio info
            info = torchaudio.info(tmp_path)
            return True
        finally:
            # Clean up the temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.warning(f"File validation failed: {str(e)}")
        return False


def process_audio(file_data: BinaryIO) -> Tuple[torch.Tensor, int]:
    """
    Validate and convert audio data to the required format (16kHz, mono).
    
    Args:
        file_data: Binary file-like object containing audio data
        
    Returns:
        Tuple[torch.Tensor, int]: A tuple containing:
            - Processed audio as a torch.Tensor
            - Sample rate (should be 16000)
            
    Raises:
        AudioProcessingError: If the file is not a valid audio file or processing fails
    """
    if not is_valid_audio_file(file_data):
        raise AudioProcessingError("Invalid audio file. Please upload a valid audio file.")
    
    try:
        # Save to a temporary file for torchaudio to read
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp:
            tmp.write(file_data.read())
            tmp_path = tmp.name
        
        try:
            # Load the audio
            waveform, sample_rate = torchaudio.load(tmp_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                logger.info(f"Converting audio from {waveform.shape[0]} channels to mono")
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample if needed
            if sample_rate != REQUIRED_SAMPLE_RATE:
                logger.info(f"Resampling audio from {sample_rate}Hz to {REQUIRED_SAMPLE_RATE}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=REQUIRED_SAMPLE_RATE
                )
                waveform = resampler(waveform)
                sample_rate = REQUIRED_SAMPLE_RATE
            
            return waveform, sample_rate
            
        finally:
            # Clean up the temporary file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        logger.error(traceback.format_exc())
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")


def save_audio_to_file(
    waveform: torch.Tensor, 
    sample_rate: int,
    file_path: Optional[str] = None
) -> str:
    """
    Save audio tensor to a WAV file.
    
    Args:
        waveform: The audio tensor to save
        sample_rate: The sample rate of the audio
        file_path: Optional path to save the file. If None, a temporary file is created.
        
    Returns:
        str: Path to the saved file
    """
    if file_path is None:
        temp_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        file_path = temp_file.name
        temp_file.close()
    
    torchaudio.save(file_path, waveform, sample_rate)
    return file_path


def get_audio_duration(waveform: torch.Tensor, sample_rate: int) -> float:
    """
    Calculate the duration of an audio waveform in seconds.
    
    Args:
        waveform: The audio tensor
        sample_rate: The sample rate of the audio
        
    Returns:
        float: Duration in seconds
    """
    num_frames = waveform.shape[1]
    return num_frames / sample_rate
