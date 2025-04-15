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
    
    def test_is_valid_audio_file(self):
        """Test the is_valid_audio_file function"""
        # Test with valid audio file
        with open(self.test_audio_path, "rb") as f:
            self.assertTrue(is_valid_audio_file(f))
        
        # Test with non-audio file
        with open(self.non_audio_path, "rb") as f:
            self.assertFalse(is_valid_audio_file(f))
    
    def test_process_audio_with_valid_file(self):
        """Test processing a valid audio file."""
        output_dir = None # Let process_audio handle temp file creation
        try:
            with open(self.test_audio_path, 'rb') as f:
                processed_path = process_audio(f)
            
            # Assert the return value is a string path
            self.assertIsInstance(processed_path, str)
            self.assertTrue(os.path.exists(processed_path))
            self.assertTrue(processed_path.endswith(".wav"))
            
            # Load the processed file to check its properties
            waveform, sample_rate = torchaudio.load(processed_path)
            self.assertEqual(sample_rate, REQUIRED_SAMPLE_RATE)
            self.assertEqual(waveform.shape[0], 1) # Should be mono
        
        finally:
            # Clean up the temporary file created by process_audio
            if 'processed_path' in locals() and os.path.exists(processed_path):
                os.unlink(processed_path)
    
    def test_process_audio_resampling(self):
        """Test that process_audio correctly resamples audio."""
        output_dir = None
        try:
            # Create an audio file with a different sample rate
            different_rate_path = os.path.join(self.temp_dir, "different_rate.wav")
            self._create_test_audio_file(different_rate_path, sample_rate=44100)
            
            with open(different_rate_path, 'rb') as f:
                processed_path = process_audio(f)
            
            self.assertIsInstance(processed_path, str)
            self.assertTrue(os.path.exists(processed_path))
            
            # Load the processed file and check sample rate
            waveform, sample_rate = torchaudio.load(processed_path)
            self.assertEqual(sample_rate, REQUIRED_SAMPLE_RATE)
        
        finally:
            if 'processed_path' in locals() and os.path.exists(processed_path):
                os.unlink(processed_path)
    
    def test_process_audio_mono_conversion(self):
        """Test that process_audio correctly converts stereo to mono."""
        output_dir = None
        try:
            # Create a stereo audio file
            stereo_path = os.path.join(self.temp_dir, "stereo.wav")
            self._create_test_audio_file(stereo_path, mono=False)
            
            with open(stereo_path, 'rb') as f:
                processed_path = process_audio(f)
            
            self.assertIsInstance(processed_path, str)
            self.assertTrue(os.path.exists(processed_path))
            
            # Load the processed file and check channels
            waveform, sample_rate = torchaudio.load(processed_path)
            self.assertEqual(waveform.shape[0], 1) # Check for mono channel
            self.assertEqual(sample_rate, REQUIRED_SAMPLE_RATE) # Also check SR
            
        finally:
            if 'processed_path' in locals() and os.path.exists(processed_path):
                os.unlink(processed_path)
    
    def test_process_audio_invalid_file(self):
        """Test that process_audio raises an error for invalid files"""
        with open(self.non_audio_path, "rb") as f:
            with self.assertRaises(AudioProcessingError):
                process_audio(f)
    
    def test_save_audio_to_file(self):
        """Test saving audio to a file."""
        output_dir = self.temp_dir
        waveform = torch.randn(1, 16000) # 1 channel, 1 second at 16kHz
        sample_rate = 16000
        
        # Test saving without specifying path (should create temp file)
        saved_path_temp = None
        try:
            saved_path_temp = save_audio_to_file(waveform, sample_rate)
            self.assertTrue(os.path.exists(saved_path_temp))
            self.assertTrue(saved_path_temp.endswith(".wav"))
            # Check content
            loaded_waveform, loaded_sr = torchaudio.load(saved_path_temp)
            self.assertEqual(loaded_sr, sample_rate)
            self.assertTrue(torch.allclose(loaded_waveform, waveform))
        finally:
            if saved_path_temp and os.path.exists(saved_path_temp):
                os.unlink(saved_path_temp)
                
        # Test saving with a specific path
        specific_path = os.path.join(output_dir, "specific_test_save.wav")
        try:
            returned_path = save_audio_to_file(waveform, sample_rate, file_path=specific_path)
            self.assertEqual(returned_path, specific_path)
            self.assertTrue(os.path.exists(specific_path))
            # Check content
            loaded_waveform, loaded_sr = torchaudio.load(specific_path)
            self.assertEqual(loaded_sr, sample_rate)
            self.assertTrue(torch.allclose(loaded_waveform, waveform))
        finally:
            if os.path.exists(specific_path):
                os.unlink(specific_path)
    
    def test_get_audio_duration(self):
        """Test the get_audio_duration function"""
        # Create an audio file with a known duration
        duration = 2.5  # seconds
        audio_path = os.path.join(self.temp_dir, "duration_test.wav")
        self._create_test_audio_file(audio_path, duration=duration)
        
        # Load the audio
        waveform, sample_rate = torchaudio.load(audio_path)
        
        # Calculate the duration
        calculated_duration = get_audio_duration(waveform, sample_rate)
        
        # Should be close to the expected duration
        self.assertAlmostEqual(calculated_duration, duration, places=1)

        # Check that duration is returned (can be None if failed)
        self.assertIsNotNone(calculated_duration)
        self.assertIsInstance(calculated_duration, float)

    def test_is_valid_audio_file_exception(self):
        """Test is_valid_audio_file when torchaudio.info raises an exception."""
        # Create dummy file data
        dummy_data = io.BytesIO(b"simulated invalid data")
        
        # Mock torchaudio.info to raise a generic exception
        # Patch within the `easy_asr_server.utils` namespace where it's used
        with mock.patch('easy_asr_server.utils.torchaudio.info', side_effect=Exception("Simulated info error")) as mock_info:
            # Call the function
            is_valid = is_valid_audio_file(dummy_data)
            
            # Verify that torchaudio.info was called (it's called on a temp file path)
            mock_info.assert_called_once()
            # We could assert the specific temp path if needed, but checking call is usually sufficient
            
            # Verify False is returned when an exception occurs
            self.assertFalse(is_valid)
            
            # Ensure the file pointer was reset
            self.assertEqual(dummy_data.tell(), 0)


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
