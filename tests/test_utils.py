"""
Tests for the utils module.
"""

import os
import io
import unittest
import tempfile
import torch
import torchaudio
import numpy as np
from pathlib import Path
from unittest import mock

from easy_asr_server.utils import (
    is_valid_audio_file,
    process_audio,
    save_audio_to_file,
    get_audio_duration,
    AudioProcessingError,
    REQUIRED_SAMPLE_RATE,
    read_hotwords
)


class TestAudioProcessingFunctions(unittest.TestCase):
    """Test cases for audio processing functions in utils.py"""
    
    def setUp(self):
        """Setup for tests"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Generate a test audio file
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self._create_test_audio_file(self.test_audio_path)
        
        # Generate a non-audio file
        self.non_audio_path = os.path.join(self.temp_dir, "test_non_audio.txt")
        with open(self.non_audio_path, "w") as f:
            f.write("This is not an audio file")
    
    def tearDown(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_audio_file(self, file_path, sample_rate=22050, mono=True, duration=1.0):
        """Create a test audio file with the specified parameters"""
        # Generate a sine wave
        num_channels = 1 if mono else 2
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        # 440 Hz sine wave
        wave = torch.sin(2 * torch.pi * 440 * t)
        
        # Convert to stereo if needed
        if not mono:
            # Make the second channel a different frequency
            wave2 = torch.sin(2 * torch.pi * 880 * t)
            wave = torch.stack([wave, wave2])
        else:
            wave = wave.unsqueeze(0)
        
        # Save the file
        torchaudio.save(file_path, wave, sample_rate)
        return file_path
    
    def _create_audio_bytes_io(self, sample_rate=22050, mono=True, duration=1.0):
        """Create a BytesIO object containing audio data"""
        # Create temporary file first
        temp_path = os.path.join(self.temp_dir, f"temp_audio_{sample_rate}_{mono}_{duration}.wav")
        self._create_test_audio_file(temp_path, sample_rate, mono, duration)
        
        # Read file into BytesIO
        with open(temp_path, 'rb') as f:
            audio_bytes = io.BytesIO(f.read())
        
        # Clean up temp file
        os.unlink(temp_path)
        return audio_bytes

    # === New tests for read_audio_bytes function ===
    def test_read_audio_bytes_valid_audio(self):
        """Test read_audio_bytes with valid audio BytesIO"""
        from easy_asr_server.utils import read_audio_bytes
        
        # Create test audio BytesIO
        audio_bytes = self._create_audio_bytes_io(sample_rate=16000, mono=True)
        
        # Call the function
        result = read_audio_bytes(audio_bytes)
        
        # Should return numpy array
        self.assertIsInstance(result, np.ndarray)
        # Should be 1D array for mono audio
        self.assertEqual(len(result.shape), 1)
        # Should have reasonable length (1 second at 16kHz)
        self.assertGreater(len(result), 15000)
        self.assertLess(len(result), 17000)

    def test_read_audio_bytes_resampling(self):
        """Test read_audio_bytes correctly resamples audio"""
        from easy_asr_server.utils import read_audio_bytes
        
        # Create audio with different sample rate
        audio_bytes = self._create_audio_bytes_io(sample_rate=44100, mono=True, duration=1.0)
        
        result = read_audio_bytes(audio_bytes)
        
        # Should be resampled to target length (1 second at 16kHz)
        self.assertIsInstance(result, np.ndarray)
        self.assertGreater(len(result), 15000)
        self.assertLess(len(result), 17000)

    def test_read_audio_bytes_stereo_to_mono(self):
        """Test read_audio_bytes converts stereo to mono"""
        from easy_asr_server.utils import read_audio_bytes
        
        # Create stereo audio
        audio_bytes = self._create_audio_bytes_io(sample_rate=16000, mono=False, duration=1.0)
        
        result = read_audio_bytes(audio_bytes)
        
        # Should be 1D array (mono)
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(len(result.shape), 1)

    def test_read_audio_bytes_invalid_data(self):
        """Test read_audio_bytes with invalid audio data"""
        from easy_asr_server.utils import read_audio_bytes
        
        # Create BytesIO with non-audio data
        invalid_bytes = io.BytesIO(b"This is not audio data")
        
        # Should raise AudioProcessingError
        with self.assertRaises(AudioProcessingError):
            read_audio_bytes(invalid_bytes)

    def test_read_audio_bytes_empty_data(self):
        """Test read_audio_bytes with empty BytesIO"""
        from easy_asr_server.utils import read_audio_bytes
        
        empty_bytes = io.BytesIO(b"")
        
        with self.assertRaises(AudioProcessingError):
            read_audio_bytes(empty_bytes)

    def test_read_audio_bytes_large_file_limit(self):
        """Test read_audio_bytes with long audio files (should work without limits)"""
        from easy_asr_server.utils import read_audio_bytes
        
        # Create a long audio file (5 seconds) - should work fine without limits
        audio_bytes = self._create_audio_bytes_io(sample_rate=16000, mono=True, duration=5.0)
        
        # Should work for any reasonable size since we removed size limits
        result = read_audio_bytes(audio_bytes)
        self.assertIsInstance(result, np.ndarray)
        
        # Should be approximately 5 seconds * 16000 Hz
        expected_length = 5.0 * 16000
        self.assertGreater(len(result), expected_length * 0.9)  # Allow some tolerance
        self.assertLess(len(result), expected_length * 1.1)

    # === Essential utility tests ===
    
    def test_is_valid_audio_file_basic(self):
        """Test the is_valid_audio_file function with basic cases"""
        # Test with valid audio file
        with open(self.test_audio_path, "rb") as f:
            self.assertTrue(is_valid_audio_file(f))
        
        # Test with non-audio file  
        with open(self.non_audio_path, "rb") as f:
            self.assertFalse(is_valid_audio_file(f))


class TestHotwordReading(unittest.TestCase):
    """Test cases for the read_hotwords function."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def _create_hotword_file(self, content: str) -> str:
        """Helper to create a temporary hotword file."""
        file_path = os.path.join(self.temp_dir, "hotwords.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return file_path

    def test_read_hotwords_valid_file(self):
        """Test reading from a valid hotword file."""
        # Case 1: Multiple words
        content1 = "hello\nworld\n  OpenAI  \n\n   GPT "
        path1 = self._create_hotword_file(content1)
        expected1 = "hello world OpenAI GPT"
        self.assertEqual(read_hotwords(path1), expected1)
        
        # Case 2: Single word
        content2 = "  testing  \n"
        path2 = self._create_hotword_file(content2)
        expected2 = "testing"
        self.assertEqual(read_hotwords(path2), expected2)

        # Case 3: Empty file
        content3 = ""
        path3 = self._create_hotword_file(content3)
        expected3 = ""
        self.assertEqual(read_hotwords(path3), expected3)
        
        # Case 4: File with only whitespace lines
        content4 = "  \n   \n\t\n"
        path4 = self._create_hotword_file(content4)
        expected4 = ""
        self.assertEqual(read_hotwords(path4), expected4)

    def test_read_hotwords_none_path(self):
        """Test read_hotwords with None as file path."""
        self.assertEqual(read_hotwords(None), "")
        
    def test_read_hotwords_file_not_found(self):
        """Test read_hotwords when the file does not exist."""
        non_existent_path = os.path.join(self.temp_dir, "non_existent_file.txt")
        # Mock logger to verify warning
        with mock.patch('easy_asr_server.utils.logger.warning') as mock_warning:
            self.assertEqual(read_hotwords(non_existent_path), "")
            mock_warning.assert_called_once_with(f"Hotword file not found: {non_existent_path}")

    def test_read_hotwords_io_error(self):
        """Test read_hotwords handles IOError during file open."""
        # Create a path but don't mock os.path.isfile, so it tries to open
        some_path = os.path.join(self.temp_dir, "io_error_test.txt")
        error_message = "Permission denied"
        
        # Mock open to raise IOError
        with mock.patch('builtins.open', mock.mock_open()) as mock_file:
            mock_file.side_effect = IOError(error_message)
            # Mock logger to verify error message
            with mock.patch('easy_asr_server.utils.logger.error') as mock_logger_error:
                # Mock os.path.isfile to return True to force attempt to open
                with mock.patch('os.path.isfile', return_value=True):
                    result = read_hotwords(some_path)
                    self.assertEqual(result, "")
                    # Check that the error was logged
                    mock_logger_error.assert_called_once()
                    log_call_args, _ = mock_logger_error.call_args
                    self.assertIn(f"Error reading hotword file {some_path}", log_call_args[0])
                    self.assertIn(error_message, log_call_args[0])


if __name__ == '__main__':
    unittest.main()
