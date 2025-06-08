"""
Tests for the high-level API module (EasyASR and related functions).
"""

import os
import io
import unittest
import tempfile
from unittest import mock
import numpy as np
import torch
import torchaudio

# Import the new high-level API components
from easy_asr_server.api import (
    EasyASR,
    create_asr_engine,
    recognize,
    get_available_pipelines,
    get_default_pipeline
)
from easy_asr_server.model_manager import ModelManager, MODEL_CONFIGS
from easy_asr_server.asr_engine import ASREngine


class TestEasyASRAPI(unittest.TestCase):
    """Test cases for the high-level EasyASR API."""
    
    def setUp(self):
        """Setup for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self._create_test_audio_file(self.test_audio_path)
        
        # Create test audio array
        self.test_audio_array = np.random.rand(16000).astype(np.float32)
    
    def tearDown(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_audio_file(self, file_path, sample_rate=16000, duration=1.0):
        """Create a test audio file"""
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        torchaudio.save(file_path, wave, sample_rate)
        return file_path

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_initialization_success(self, mock_asr_engine_class, mock_model_manager_class, mock_resolve_device):
        """Test successful EasyASR initialization"""
        # Setup mocks
        mock_resolve_device.return_value = "cpu"
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Test initialization
        asr = EasyASR(pipeline="sensevoice", device="cpu", hotwords="test", auto_init=True)
        
        # Verify calls
        mock_resolve_device.assert_called_once_with("cpu")
        mock_model_manager_class.assert_called_once()
        mock_model_manager.load_pipeline.assert_called_once_with(
            pipeline_type="sensevoice", 
            device="cpu"
        )
        mock_asr_engine_class.assert_called_once_with(model_manager=mock_model_manager)
        mock_asr_engine.test_health.assert_called_once()
        
        # Verify state
        self.assertTrue(asr._initialized)
        self.assertTrue(asr.is_healthy())

    def test_easy_asr_invalid_pipeline(self):
        """Test EasyASR with invalid pipeline"""
        with self.assertRaises(ValueError) as context:
            EasyASR(pipeline="invalid_pipeline")
        
        self.assertIn("Invalid pipeline 'invalid_pipeline'", str(context.exception))

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    def test_easy_asr_initialization_failure(self, mock_model_manager_class, mock_resolve_device):
        """Test EasyASR initialization failure"""
        # Setup mocks to fail
        mock_resolve_device.return_value = "cpu"
        mock_model_manager_class.side_effect = RuntimeError("Model load failed")
        
        with self.assertRaises(RuntimeError) as context:
            EasyASR(pipeline="sensevoice", auto_init=True)
        
        self.assertIn("ASR initialization failed", str(context.exception))

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_recognition_success(self, mock_asr_engine_class, mock_model_manager_class, mock_resolve_device):
        """Test successful recognition with EasyASR"""
        # Setup mocks
        mock_resolve_device.return_value = "cpu"
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine.recognize.return_value = "test recognition result"
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Test recognition
        asr = EasyASR(pipeline="sensevoice", hotwords="test hotwords")
        result = asr.recognize(self.test_audio_path)
        
        # Verify result
        self.assertEqual(result, "test recognition result")
        mock_asr_engine.recognize.assert_called_once_with(
            self.test_audio_path, 
            hotword="test hotwords"
        )

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_recognition_with_array(self, mock_asr_engine_class, mock_model_manager_class, mock_resolve_device):
        """Test recognition with numpy array input"""
        # Setup mocks
        mock_resolve_device.return_value = "cpu"
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine.recognize.return_value = "array recognition result"
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Test recognition with numpy array
        asr = EasyASR(pipeline="sensevoice")
        result = asr.recognize(self.test_audio_array, hotwords="override hotwords")
        
        # Verify result
        self.assertEqual(result, "array recognition result")
        mock_asr_engine.recognize.assert_called_once_with(
            self.test_audio_array, 
            hotword="override hotwords"
        )

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_context_manager(self, mock_asr_engine_class, mock_model_manager_class, mock_resolve_device):
        """Test EasyASR as context manager"""
        # Setup mocks
        mock_resolve_device.return_value = "cpu"
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine.recognize.return_value = "context result"
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Test context manager
        with EasyASR(pipeline="sensevoice", auto_init=False) as asr:
            # Should auto-initialize on context entry
            result = asr.recognize(self.test_audio_path)
            self.assertEqual(result, "context result")
            self.assertTrue(asr._initialized)
        
        # Should be cleaned up after context exit
        self.assertFalse(asr._initialized)

    @mock.patch('easy_asr_server.api.EasyASR')
    def test_create_asr_engine_function(self, mock_easy_asr_class):
        """Test create_asr_engine convenience function"""
        mock_asr = mock.MagicMock()
        mock_easy_asr_class.return_value = mock_asr
        
        result = create_asr_engine(
            pipeline="paraformer",
            device="cuda",
            hotwords="test",
            log_level="INFO"
        )
        
        # Verify function call
        mock_easy_asr_class.assert_called_once_with(
            pipeline="paraformer",
            device="cuda",
            hotwords="test",
            log_level="INFO",
            auto_init=True
        )
        self.assertEqual(result, mock_asr)

    @mock.patch('easy_asr_server.api.EasyASR')
    def test_recognize_function_one_shot(self, mock_easy_asr_class):
        """Test recognize convenience function for one-shot usage"""
        # Setup context manager mock properly
        mock_asr = mock.MagicMock()
        mock_asr.recognize.return_value = "one-shot result"
        mock_asr.__enter__.return_value = mock_asr
        mock_asr.__exit__.return_value = None
        mock_easy_asr_class.return_value = mock_asr
        
        # Test one-shot recognition
        result = recognize(
            audio_input=self.test_audio_path,
            pipeline="sensevoice",
            device="cpu",
            hotwords="test hotwords"
        )
        
        # Verify result
        self.assertEqual(result, "one-shot result")
        mock_easy_asr_class.assert_called_once_with(
            pipeline="sensevoice",
            device="cpu",
            hotwords="test hotwords"
        )
        mock_asr.__enter__.assert_called_once()
        mock_asr.recognize.assert_called_once_with(self.test_audio_path)
        mock_asr.__exit__.assert_called_once()

    @mock.patch('easy_asr_server.api.EasyASR')
    def test_recognize_function_with_reused_engine(self, mock_easy_asr_class):
        """Test recognize function with pre-existing engine"""
        # Create a mock engine
        mock_existing_asr = mock.MagicMock()
        mock_existing_asr.recognize.return_value = "reused engine result"
        
        # Test with existing engine
        result = recognize(
            audio_input=self.test_audio_array,
            hotwords="override",
            asr_engine=mock_existing_asr
        )
        
        # Should not create new engine
        mock_easy_asr_class.assert_not_called()
        mock_existing_asr.recognize.assert_called_once_with(
            self.test_audio_array,
            hotwords="override"
        )
        self.assertEqual(result, "reused engine result")

    def test_get_available_pipelines(self):
        """Test get_available_pipelines function"""
        pipelines = get_available_pipelines()
        
        # Should return a copy of MODEL_CONFIGS
        self.assertIsInstance(pipelines, dict)
        self.assertEqual(pipelines, MODEL_CONFIGS)
        
        # Should be a copy, not the same object
        self.assertIsNot(pipelines, MODEL_CONFIGS)

    def test_get_default_pipeline(self):
        """Test get_default_pipeline function"""
        default = get_default_pipeline()
        
        # Should return the default pipeline name
        self.assertIsInstance(default, str)
        self.assertIn(default, MODEL_CONFIGS)

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_get_info(self, mock_asr_engine_class, mock_model_manager_class, mock_resolve_device):
        """Test EasyASR get_info method"""
        # Setup mocks
        mock_resolve_device.return_value = "cuda"  # Return the resolved device
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Create engine
        asr = EasyASR(
            pipeline="paraformer",
            device="cuda",
            hotwords="info test"
        )
        
        # Get info
        info = asr.get_info()
        
        # Verify info structure
        expected_keys = ["pipeline", "device", "resolved_device", "hotwords", "initialized", "healthy"]
        self.assertEqual(set(info.keys()), set(expected_keys))
        self.assertEqual(info["pipeline"], "paraformer")
        self.assertEqual(info["device"], "cuda")
        self.assertEqual(info["resolved_device"], "cuda")  # Should be the resolved device
        self.assertEqual(info["hotwords"], "info test")
        self.assertTrue(info["initialized"])
        self.assertTrue(info["healthy"])

    def test_easy_asr_uninitialized_recognition(self):
        """Test recognition fails on uninitialized engine"""
        asr = EasyASR(auto_init=False)
        
        with self.assertRaises(RuntimeError) as context:
            asr.recognize(self.test_audio_path)
        
        self.assertIn("ASR engine not initialized", str(context.exception))

    @mock.patch('easy_asr_server.api.resolve_device_string')
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_manual_initialization(self, mock_asr_engine_class, mock_model_manager_class, mock_resolve_device):
        """Test manual initialization after auto_init=False"""
        # Setup mocks
        mock_resolve_device.return_value = "cpu"
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Create without auto-init
        asr = EasyASR(pipeline="sensevoice", auto_init=False)
        self.assertFalse(asr._initialized)
        
        # Initialize manually
        success = asr.initialize()
        self.assertTrue(success)
        self.assertTrue(asr._initialized)
        
        # Should work after manual init
        mock_asr_engine.recognize.return_value = "manual init result"
        result = asr.recognize(self.test_audio_path)
        self.assertEqual(result, "manual init result")


if __name__ == '__main__':
    unittest.main() 