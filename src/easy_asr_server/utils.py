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


def resolve_device_string(device: str) -> str:
    """Resolves the device string 'auto' and validates specific device strings.

    Args:
        device: The requested device string (e.g., "auto", "cpu", "cuda", "cuda:0", "mps").

    Returns:
        The validated and resolved device string (e.g., "cpu", "cuda", "mps").

    Raises:
        ValueError: If the specified device string is invalid or the requested hardware
                    (CUDA/MPS) is not available or the CUDA index is invalid.
    """
    logger.info(f"Resolving device string: '{device}'")

    if device == "auto":
        if torch.cuda.is_available():
            resolved = "cuda"
            logger.info("'auto' resolved to 'cuda' (CUDA available).")
        elif torch.backends.mps.is_available(): # Check MPS only if CUDA not found
            resolved = "mps"
            logger.info("'auto' resolved to 'mps' (CUDA unavailable, MPS available).")
        else:
            resolved = "cpu"
            logger.info("'auto' resolved to 'cpu' (CUDA and MPS unavailable).")
        return resolved

    elif device == "cpu":
        logger.info("Using 'cpu' device.")
        return device # Always available

    elif device == "mps":
        if not torch.backends.mps.is_available():
            logger.error("MPS device requested but not available.")
            raise ValueError("MPS device requested but not available")
        logger.info("Using 'mps' device.")
        return device

    elif device == "cuda" or device.startswith("cuda:"):
        if not torch.cuda.is_available():
            logger.error("CUDA device requested but not available.")
            raise ValueError("CUDA device requested but not available")
        
        if device == "cuda":
            logger.info("Using 'cuda' device (implicitly cuda:0).")
            return device # Use default CUDA device
        
        # Validate cuda:N format and index
        try:
            parts = device.split(':')
            if len(parts) != 2 or not parts[1].isdigit():
                raise ValueError(f"Invalid CUDA device format: {device}")
            
            index = int(parts[1])
            count = torch.cuda.device_count()
            
            if index < 0:
                 raise ValueError(f"Invalid CUDA device index: {index}. Index cannot be negative.")
            if index >= count:
                available_indices = list(range(count))
                logger.error(f"Invalid CUDA device index: {index}. Available indices: {available_indices}")
                raise ValueError(f"Invalid CUDA device index: {index}. Available indices: {available_indices}")
            
            logger.info(f"Using specific CUDA device: '{device}'")
            return device # Valid index
            
        except ValueError as e: # Catch parsing errors or index errors
             logger.error(f"Error validating CUDA device string '{device}': {e}")
             # Re-raise specific errors for clarity or a general one
             if "Invalid CUDA device format" in str(e) or "Invalid CUDA device index" in str(e):
                 raise e
             else: # Catch potential int() conversion errors, though isdigit should prevent this
                 raise ValueError(f"Invalid CUDA device format: {device}")

    else:
        logger.error(f"Invalid device string specified: '{device}'")
        raise ValueError(f"Invalid device string specified: '{device}'")


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


def process_audio(file_data: BinaryIO) -> str:
    """
    Validate, convert audio data to the required format (16kHz, mono, WAV),
    and save it to a temporary file.
    
    Args:
        file_data: Binary file-like object containing audio data.
        
    Returns:
        str: The path to the temporary WAV file containing the processed audio.
            The caller is responsible for cleaning up this file.
            
    Raises:
        AudioProcessingError: If the file is not a valid audio file or processing fails.
    """
    # Check validity first without consuming the stream if possible
    original_pos = file_data.tell()
    if not is_valid_audio_file(file_data):
        # is_valid_audio_file resets the position
        raise AudioProcessingError("Invalid or unsupported audio file format.")
    # Ensure position is reset again just in case
    file_data.seek(original_pos)
    
    temp_input_path = None
    processed_output_path = None
    try:
        # Save original to a temporary file for torchaudio to read reliably
        with tempfile.NamedTemporaryFile(delete=False) as tmp_in:
            tmp_in.write(file_data.read())
            temp_input_path = tmp_in.name
        
        # Load the audio using the temporary input path
        waveform, sample_rate = torchaudio.load(temp_input_path)
        
        # --- Perform processing (mono conversion, resampling) --- 
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
            sample_rate = REQUIRED_SAMPLE_RATE # Update sample rate after resampling
            
        # --- Save processed audio to a new temporary WAV file --- 
        # Create a new temp file for the output, ensuring it gets cleaned up eventually
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_out:
            processed_output_path = tmp_out.name
            
        logger.debug(f"Saving processed audio to temporary file: {processed_output_path}")
        # Use torchaudio.save to ensure WAV format
        torchaudio.save(processed_output_path, waveform, sample_rate)
        
        return processed_output_path
        
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}", exc_info=True)
        # Clean up the output file if it was created before the error
        if processed_output_path and os.path.exists(processed_output_path):
            try:
                os.unlink(processed_output_path)
            except OSError:
                logger.warning(f"Could not delete temporary output file during error handling: {processed_output_path}")
        raise AudioProcessingError(f"Failed to process audio: {str(e)}")
    finally:
        # Clean up the temporary input file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
            except OSError:
                logger.warning(f"Could not delete temporary input file: {temp_input_path}")


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
        str: Path to the saved file. If file_path was None, returns a temporary file path
             that the CALLER IS RESPONSIBLE FOR DELETING after use.
        
    Warning:
        When file_path is None, this function creates a temporary file that will NOT be
        automatically deleted. The caller must delete the returned file path to avoid
        disk space leaks.
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


def read_hotwords(file_path: Optional[str]) -> str:
    """Reads hotwords from a file (one per line) and returns a space-separated string.

    Args:
        file_path: Path to the hotword file.

    Returns:
        A single string with hotwords separated by spaces, or an empty string if
        file_path is None, the file doesn't exist, or an error occurs.
    """
    if not file_path:
        return ""

    try:
        if not os.path.isfile(file_path):
            logger.warning(f"Hotword file not found: {file_path}")
            return ""
            
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        # Process lines: strip whitespace, filter empty lines
        hotwords = [line.strip() for line in lines if line.strip()]
        
        # Join with spaces
        return " ".join(hotwords)
        
    except FileNotFoundError:
        logger.warning(f"Hotword file not found during read: {file_path}")
        return ""
    except IOError as e:
        logger.error(f"Error reading hotword file {file_path}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Unexpected error reading hotword file {file_path}: {e}")
        return ""


def read_audio_bytes(audio_bytes: BinaryIO) -> np.ndarray:
    """
    Read audio data from BytesIO and process it to standard format numpy array.
    
    Args:
        audio_bytes: BytesIO object containing audio data
        
    Returns:
        np.ndarray: float32, 1D array at 16kHz sample rate
        
    Raises:
        AudioProcessingError: If audio processing fails
    """
    temp_input_path = None
    try:
        # Reset BytesIO position to beginning
        original_pos = audio_bytes.tell()
        audio_bytes.seek(0)
        
        # Save BytesIO content to temporary file for torchaudio to read
        with tempfile.NamedTemporaryFile(suffix='.tmp', delete=False) as tmp:
            tmp.write(audio_bytes.read())
            temp_input_path = tmp.name
        
        # Reset BytesIO position back to original
        audio_bytes.seek(original_pos)
        
        # Load the audio using torchaudio
        try:
            waveform, sample_rate = torchaudio.load(temp_input_path)
        except Exception as e:
            raise AudioProcessingError(f"Failed to load audio data: {str(e)}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            logger.debug(f"Converting audio from {waveform.shape[0]} channels to mono")
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if needed
        if sample_rate != REQUIRED_SAMPLE_RATE:
            logger.debug(f"Resampling audio from {sample_rate}Hz to {REQUIRED_SAMPLE_RATE}Hz")
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, 
                new_freq=REQUIRED_SAMPLE_RATE
            )
            waveform = resampler(waveform)
        
        # Convert to numpy array and flatten to 1D
        audio_array = waveform.squeeze().numpy().astype(np.float32)
        
        # Ensure 1D array (in case of single-sample audio)
        if audio_array.ndim == 0:
            audio_array = np.array([audio_array], dtype=np.float32)
        
        logger.debug(f"Successfully processed audio to numpy array: shape={audio_array.shape}, dtype={audio_array.dtype}")
        return audio_array
        
    except AudioProcessingError:
        # Re-raise AudioProcessingError as-is
        raise
    except Exception as e:
        logger.error(f"Unexpected error in read_audio_bytes: {str(e)}", exc_info=True)
        raise AudioProcessingError(f"Failed to process audio bytes: {str(e)}")
    finally:
        # Clean up temporary file
        if temp_input_path and os.path.exists(temp_input_path):
            try:
                os.unlink(temp_input_path)
                logger.debug(f"Cleaned up temporary file: {temp_input_path}")
            except OSError as e:
                logger.warning(f"Failed to delete temporary file {temp_input_path}: {e}")
