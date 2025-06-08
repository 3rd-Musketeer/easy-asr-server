"""
Tests for the asr_engine module.
"""

import os
import io
import unittest
import tempfile
from unittest import mock
import numpy as np
import json
import torch
import torchaudio
import shutil

from easy_asr_server.asr_engine import ASREngine
from easy_asr_server.model_manager import ModelManager


class MockModelManager:
    def __init__(self):
        self.generate_called = False
        self.generate_input_audio = None
        self.generate_kwargs = None
        # Allow setting custom return value or exception
        self._generate_return_value = "mock manager generated text"
        self._generate_side_effect = None # To store exception or callable

    def generate(self, input_audio, **kwargs):
        self.generate_called = True
        self.generate_input_audio = input_audio
        self.generate_kwargs = kwargs
        # Check if a side effect (like an exception) is set
        if self._generate_side_effect is not None:
            if isinstance(self._generate_side_effect, Exception):
                raise self._generate_side_effect
            else: # Assume it's a callable
                return self._generate_side_effect(input_audio, **kwargs)
        # Otherwise, return the configured value
        return self._generate_return_value


class TestASREngine(unittest.TestCase):
    """Test cases for the refactored ASREngine class"""
    
    def setUp(self):
        """Setup for tests"""
        # Create a mock ModelManager instance
        self.mock_manager = MockModelManager()
        
        # Initialize the engine with the mock manager
        self.engine = ASREngine(model_manager=self.mock_manager)
        
        # Create a dummy audio file path for testing recognize
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_asr_engine_audio.wav")
        # Create a minimal valid wav file
        torchaudio.save(self.test_audio_path, torch.zeros((1, 1600)), 16000)

    def tearDown(self):
        """Cleanup after tests"""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_test_audio_numpy(self, sample_rate=16000, duration=1.0):
        """Create a test numpy audio array"""
        samples = int(sample_rate * duration)
        t = np.linspace(0, duration, samples)
        # 440 Hz sine wave
        audio_array = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio_array

    # === Tests for file path input (existing functionality) ===
    def test_recognize_success(self):
        """Test successful recognition call."""
        audio_path = "dummy_audio.wav"
        expected_text = "recognized text result"
        input_hotword = "engine hotword test"
        # Set the return value on the mock manager instance
        self.mock_manager._generate_return_value = expected_text
        self.mock_manager._generate_side_effect = None # Ensure no exception
        
        # Call recognize with hotword
        result = self.engine.recognize(audio_path, hotword=input_hotword)
        
        self.assertEqual(result, expected_text)
        # Verify generate was called by checking attributes on the mock manager instance
        self.assertTrue(self.mock_manager.generate_called)
        self.assertEqual(self.mock_manager.generate_input_audio, audio_path)
        self.assertEqual(self.mock_manager.generate_kwargs, {"hotword": input_hotword})

    def test_recognize_failure(self):
        """Test recognition when ModelManager raises an exception."""
        audio_path = "failing_audio.wav"
        error_message = "Model generation failed"
        # Set the side effect (exception) on the mock manager instance
        self.mock_manager._generate_side_effect = RuntimeError(error_message)
        
        # Expect recognize to propagate the exception
        with self.assertRaisesRegex(RuntimeError, error_message):
            self.engine.recognize(audio_path, hotword="fail_hotword") # Pass hotword
            
        # Verify generate was called by checking attributes on the mock manager instance
        self.assertTrue(self.mock_manager.generate_called)
        self.assertEqual(self.mock_manager.generate_input_audio, audio_path)
        self.assertEqual(self.mock_manager.generate_kwargs, {"hotword": "fail_hotword"})

    # === New tests for numpy array input ===
    def test_recognize_numpy_array_success(self):
        """Test successful recognition with numpy array input"""
        audio_array = self._create_test_audio_numpy()
        expected_text = "numpy recognition result"
        input_hotword = "numpy test"
        
        self.mock_manager._generate_return_value = expected_text
        self.mock_manager._generate_side_effect = None
        
        # Call recognize with numpy array
        result = self.engine.recognize(audio_array, hotword=input_hotword)
        
        self.assertEqual(result, expected_text)
        # Verify generate was called with numpy array
        self.assertTrue(self.mock_manager.generate_called)
        self.assertIsInstance(self.mock_manager.generate_input_audio, np.ndarray)
        np.testing.assert_array_equal(self.mock_manager.generate_input_audio, audio_array)
        self.assertEqual(self.mock_manager.generate_kwargs, {"hotword": input_hotword})

    def test_recognize_input_validation_comprehensive(self):
        """Test recognition input validation comprehensively"""
        
        # Test invalid numpy array shape (should be 1D)
        invalid_shape_array = np.random.rand(2, 1000).astype(np.float32)
        with self.assertRaises(ValueError) as cm:
            self.engine.recognize(invalid_shape_array, hotword="test")
        self.assertIn("1-dimensional", str(cm.exception))
        
        # Test invalid numpy array dtype (should be float32)
        invalid_dtype_array = np.random.randint(0, 100, size=16000, dtype=np.int32)
        with self.assertRaises(ValueError) as cm:
            self.engine.recognize(invalid_dtype_array, hotword="test")
        self.assertIn("float32", str(cm.exception))
        
        # Test too short audio array
        short_array = np.random.rand(100).astype(np.float32)
        with self.assertRaises(ValueError) as cm:
            self.engine.recognize(short_array, hotword="test")
        self.assertIn("too short", str(cm.exception))
        
        # Test invalid input types
        invalid_inputs = [123, [], {}, None]
        for invalid_input in invalid_inputs:
            with self.assertRaises((ValueError, TypeError)):
                self.engine.recognize(invalid_input, hotword="test")

    def test_recognize_backwards_compatibility(self):
        """Test that file path input still works (backwards compatibility)"""
        # This test ensures existing code won't break
        audio_path = self.test_audio_path  # Use real file
        expected_text = "backwards compatibility test"
        
        self.mock_manager._generate_return_value = expected_text
        
        result = self.engine.recognize(audio_path, hotword="compat")
        
        self.assertEqual(result, expected_text)
        self.assertEqual(self.mock_manager.generate_input_audio, audio_path)

    # === Existing health check tests ===
    def test_health_check_success(self):
        """Test the health check passes when ModelManager is healthy"""
        # Mock os.unlink to prevent errors trying to delete the temp file created by health_check
        with mock.patch('os.unlink') as mock_unlink:
            self.assertTrue(self.engine.test_health())
            # Check that generate was called during health check
            self.assertTrue(self.mock_manager.generate_called)
            # Verify the input was a string (path) and ends with .wav
            self.assertIsInstance(self.mock_manager.generate_input_audio, str)
            self.assertTrue(self.mock_manager.generate_input_audio.endswith(".wav"))
            # Assert os.unlink was called, indicating cleanup was attempted
            mock_unlink.assert_called_once()
        
    def test_health_check_failure(self):
        """Test the health check fails when ModelManager.generate fails"""
        # Set the side effect for generate to raise an error
        self.mock_manager._generate_side_effect = RuntimeError("Health check generate failed")
        # Mock os.unlink
        with mock.patch('os.unlink') as mock_unlink:
            self.assertFalse(self.engine.test_health())
            # Assert os.unlink was still called in the finally block
            mock_unlink.assert_called_once()


if __name__ == '__main__':
    unittest.main() 