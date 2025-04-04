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

from easy_asr_server.utils import (
    is_valid_audio_file,
    process_audio,
    save_audio_to_file,
    get_audio_duration,
    AudioProcessingError,
    REQUIRED_SAMPLE_RATE
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
        """Test the process_audio function with a valid audio file"""
        with open(self.test_audio_path, "rb") as f:
            waveform, sample_rate = process_audio(f)
        
        # Check the resulting sample rate
        self.assertEqual(sample_rate, REQUIRED_SAMPLE_RATE)
        
        # Check the waveform shape (should be 1 channel)
        self.assertEqual(waveform.shape[0], 1)
        
        # Should be a torch tensor
        self.assertIsInstance(waveform, torch.Tensor)
    
    def test_process_audio_resampling(self):
        """Test that process_audio correctly resamples audio"""
        # Create an audio file with a different sample rate
        different_rate_path = os.path.join(self.temp_dir, "different_rate.wav")
        self._create_test_audio_file(different_rate_path, sample_rate=44100)
        
        with open(different_rate_path, "rb") as f:
            waveform, sample_rate = process_audio(f)
        
        # Should be resampled to the required sample rate
        self.assertEqual(sample_rate, REQUIRED_SAMPLE_RATE)
    
    def test_process_audio_mono_conversion(self):
        """Test that process_audio correctly converts stereo to mono"""
        # Create a stereo audio file
        stereo_path = os.path.join(self.temp_dir, "stereo.wav")
        self._create_test_audio_file(stereo_path, mono=False)
        
        with open(stereo_path, "rb") as f:
            waveform, sample_rate = process_audio(f)
        
        # Should be converted to mono (1 channel)
        self.assertEqual(waveform.shape[0], 1)
    
    def test_process_audio_invalid_file(self):
        """Test that process_audio raises an error for invalid files"""
        with open(self.non_audio_path, "rb") as f:
            with self.assertRaises(AudioProcessingError):
                process_audio(f)
    
    def test_save_audio_to_file(self):
        """Test the save_audio_to_file function"""
        # Generate some audio data
        waveform = torch.sin(torch.linspace(0, 1, 16000)).unsqueeze(0)
        sample_rate = 16000
        
        # Test with a specified path
        output_path = os.path.join(self.temp_dir, "output.wav")
        result_path = save_audio_to_file(waveform, sample_rate, output_path)
        
        # Path should be the one we specified
        self.assertEqual(result_path, output_path)
        
        # File should exist
        self.assertTrue(os.path.exists(output_path))
        
        # Test with auto-generated path
        result_path = save_audio_to_file(waveform, sample_rate)
        
        # Path should be a string
        self.assertIsInstance(result_path, str)
        
        # File should exist
        self.assertTrue(os.path.exists(result_path))
    
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


if __name__ == '__main__':
    unittest.main()
