"""
Tests for the asr_engine module.
"""

import os
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