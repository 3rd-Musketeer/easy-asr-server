"""
Tests for the asr_engine module.
"""

import os
import unittest
import tempfile
from unittest import mock
import torch
import torchaudio
import json

from easy_asr_server.asr_engine import ASREngine, ASREngineError


class MockAutoModel:
    """Mock for FunASR AutoModel class"""
    
    def __init__(self, model, device=None, **kwargs):
        self.model = model
        self.device = device
        self.kwargs = kwargs
        
    def generate(self, input, **kwargs):
        """Mock generate method that returns a fixed result"""
        # Check if the input is a file path and it exists
        if isinstance(input, str) and os.path.exists(input):
            # Return a mock result with recognized text
            return [{"text": "mock recognized text"}]
        else:
            raise ValueError(f"Invalid input: {input}")


class TestASREngine(unittest.TestCase):
    """Test cases for the ASREngine class"""
    
    def setUp(self):
        """Setup for tests"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a mock model path
        self.mock_model_path = os.path.join(self.temp_dir, "mock_model")
        os.makedirs(self.mock_model_path, exist_ok=True)
        
        # Create a mock VAD model path
        self.mock_vad_model_path = os.path.join(self.temp_dir, "mock_vad_model")
        os.makedirs(self.mock_vad_model_path, exist_ok=True)
        
        # Create a test audio file
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self._create_test_audio_file(self.test_audio_path)
        
        # Mock the AutoModel class
        self.patcher = mock.patch('easy_asr_server.asr_engine.AutoModel', MockAutoModel)
        self.mock_auto_model = self.patcher.start()
    
    def tearDown(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Stop the patcher
        self.patcher.stop()
    
    def _create_test_audio_file(self, file_path, sample_rate=16000, duration=1.0):
        """Create a test audio file"""
        # Generate a sine wave
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave, mono
        
        # Save the file
        torchaudio.save(file_path, wave, sample_rate)
        return file_path
    
    def test_initialization(self):
        """Test that ASREngine initializes correctly"""
        # Initialize the engine
        engine = ASREngine(self.mock_model_path, self.mock_vad_model_path)
        
        # Check that the model was initialized
        self.assertIsNotNone(engine.model)
        
        # Check that the model path was set
        self.assertEqual(engine.asr_model_path, self.mock_model_path)
        self.assertEqual(engine.vad_model_path, self.mock_vad_model_path)
    
    def test_device_detection(self):
        """Test that ASREngine correctly handles device specification"""
        # Test with auto detection
        engine_auto = ASREngine(self.mock_model_path, self.mock_vad_model_path, device="auto")
        self.assertIn(engine_auto.device, ["cpu", "cuda"])
        
        # Test with explicit CPU
        engine_cpu = ASREngine(self.mock_model_path, self.mock_vad_model_path, device="cpu")
        self.assertEqual(engine_cpu.device, "cpu")
        
        # Test with explicit CUDA (if available)
        if torch.cuda.is_available():
            engine_cuda = ASREngine(self.mock_model_path, self.mock_vad_model_path, device="cuda")
            self.assertEqual(engine_cuda.device, "cuda")
    
    def test_recognize(self):
        """Test the recognize method"""
        # Load the test audio
        waveform, sample_rate = torchaudio.load(self.test_audio_path)
        
        # Initialize the engine
        engine = ASREngine(self.mock_model_path, self.mock_vad_model_path)
        
        # Call recognize
        text = engine.recognize(waveform, sample_rate)
        
        # Should return the mock text
        self.assertEqual(text, "mock recognized text")
    
    def test_health_check(self):
        """Test the health check functionality"""
        # Initialize the engine
        engine = ASREngine(self.mock_model_path, self.mock_vad_model_path)
        
        # Check health
        self.assertTrue(engine.test_health())
    
    def test_create_class_method(self):
        """Test the create class method"""
        # Mock the ModelManager
        with mock.patch('easy_asr_server.asr_engine.ModelManager') as mock_manager:
            # Configure the mock
            mock_manager_instance = mock_manager.return_value
            mock_manager_instance.ensure_models_downloaded.return_value = {
                "asr": self.mock_model_path,
                "vad": self.mock_vad_model_path
            }
            
            # Call the create method
            engine = ASREngine.create()
            
            # Check that ModelManager was called correctly
            mock_manager.assert_called_once()
            mock_manager_instance.ensure_models_downloaded.assert_called_once()
            
            # Check that the engine was initialized with the mock model paths
            self.assertEqual(engine.asr_model_path, self.mock_model_path)
            self.assertEqual(engine.vad_model_path, self.mock_vad_model_path)


if __name__ == '__main__':
    unittest.main() 