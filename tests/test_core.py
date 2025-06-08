"""
Core functionality tests for easy_asr_server.
Tests the essential features: model download, model loading, ASR recognition, API endpoints, and CLI.
"""

import os
import io
import unittest
import tempfile
import shutil
from unittest import mock
import torch
import torchaudio
import numpy as np
from fastapi.testclient import TestClient

# Import all core components
from easy_asr_server.model_manager import ModelManager, MODEL_CONFIGS, DEFAULT_PIPELINE
from easy_asr_server.asr_engine import ASREngine
from easy_asr_server.api import app, get_asr_engine, app_state
from easy_asr_server.utils import process_audio, read_hotwords, AudioProcessingError


class MockModelManager:
    """Simple mock for ModelManager"""
    def __init__(self):
        self.generate_called = False
        self.generate_return = "mock recognition result"
        
    def generate(self, input_audio, **kwargs):
        self.generate_called = True
        if isinstance(self.generate_return, Exception):
            raise self.generate_return
        return self.generate_return


class MockAutoModel:
    """Simple mock for FunASR AutoModel"""
    def __init__(self, **kwargs):
        self.constructor_args = kwargs
        
    def generate(self, input, **kwargs):
        return [{"text": "mock generated text"}]


class CoreFunctionalityTests(unittest.TestCase):
    """Essential tests for all core functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = self._create_test_audio()
        
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
        ModelManager._instance = None  # Reset singleton
        
    def _create_test_audio(self, duration=1.0):
        """Create a simple test audio file"""
        path = os.path.join(self.temp_dir, "test.wav")
        samples = int(16000 * duration)
        wave = torch.sin(2 * torch.pi * 440 * torch.linspace(0, duration, samples)).unsqueeze(0)
        torchaudio.save(path, wave, 16000)
        return path

    # === Model Manager Tests ===
    
    @mock.patch('easy_asr_server.model_manager.snapshot_download')
    @mock.patch('easy_asr_server.model_manager.AutoModel')
    def test_model_manager_download_and_load(self, mock_automodel, mock_download):
        """Test model download and pipeline loading"""
        # Setup mocks
        mock_download.return_value = "/fake/model/path"
        mock_automodel.return_value = MockAutoModel()
        
        manager = ModelManager()
        manager._cache_dir = self.temp_dir
        
        # Test model download and pipeline loading with comprehensive mocking
        def mock_exists_side_effect(path):
            if "download_complete" in path:
                return False  # Simulate completion file doesn't exist initially
            elif path == "/fake/model/path":
                return True   # Mock downloaded path exists
            return True  # Other paths exist
            
        with mock.patch('os.path.exists', side_effect=mock_exists_side_effect), \
             mock.patch('os.remove'), \
             mock.patch('os.makedirs'), \
             mock.patch('builtins.open', mock.mock_open(read_data="/fake/model/path")), \
             mock.patch('filelock.FileLock') as mock_filelock:
            
            # Mock the file lock context manager
            mock_filelock.return_value.__enter__ = mock.Mock(return_value=None)
            mock_filelock.return_value.__exit__ = mock.Mock(return_value=None)
            
            # Test model download
            path = manager.get_model_path("test/model")
            self.assertEqual(path, "/fake/model/path")
            mock_download.assert_called_once()
            
            # Test pipeline loading
            manager.load_pipeline("sensevoice", "cpu")
            mock_automodel.assert_called_once()
            
            # Test generation
            result = manager.generate("test_audio.wav", hotword="test")
            self.assertEqual(result, "mock generated text")

    # === ASR Engine Tests ===
    
    def test_asr_engine_recognition(self):
        """Test ASR engine recognition with both file and numpy input"""
        mock_manager = MockModelManager()
        engine = ASREngine(model_manager=mock_manager)
        
        # Test file path input
        result = engine.recognize(self.test_audio_path, hotword="test")
        self.assertEqual(result, "mock recognition result")
        self.assertTrue(mock_manager.generate_called)
        
        # Test numpy array input
        mock_manager.generate_called = False
        audio_array = np.random.rand(16000).astype(np.float32)
        result = engine.recognize(audio_array, hotword="test")
        self.assertEqual(result, "mock recognition result")
        self.assertTrue(mock_manager.generate_called)
        
    def test_asr_engine_health_check(self):
        """Test ASR engine health check"""
        mock_manager = MockModelManager()
        engine = ASREngine(model_manager=mock_manager)
        
        with mock.patch('os.unlink'):
            # Healthy case
            self.assertTrue(engine.test_health())
            
            # Unhealthy case
            mock_manager.generate_return = RuntimeError("Model error")
            self.assertFalse(engine.test_health())

    # === API Tests ===
    
    def test_api_endpoints(self):
        """Test essential API endpoints"""
        # Setup mock engine
        mock_engine = mock.MagicMock(spec=ASREngine)
        mock_engine.test_health.return_value = True
        mock_engine.recognize.return_value = "api test result"
        
        async def mock_get_engine():
            return mock_engine
            
        # Override dependency
        app.dependency_overrides[get_asr_engine] = mock_get_engine
        app_state.update({"pipeline_type": "test", "device": "cpu"})
        
        try:
            client = TestClient(app)
            
            # Test health endpoint
            response = client.get("/asr/health")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "healthy")
            
            # Test recognition endpoint
            with mock.patch('easy_asr_server.server.read_audio_bytes') as mock_read:
                mock_read.return_value = np.random.rand(16000).astype(np.float32)
                
                with open(self.test_audio_path, "rb") as f:
                    response = client.post(
                        "/asr/recognize",
                        files={"audio": ("test.wav", f, "audio/wav")}
                    )
                
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["text"], "api test result")
                mock_engine.recognize.assert_called_once()
                
        finally:
            app.dependency_overrides.clear()
            app_state.clear()

    # === Utils Tests ===
    
    def test_audio_processing(self):
        """Test essential audio processing functions"""
        # Test process_audio
        with open(self.test_audio_path, 'rb') as f:
            processed_path = process_audio(f)
            
        self.assertTrue(os.path.exists(processed_path))
        waveform, sr = torchaudio.load(processed_path)
        self.assertEqual(sr, 16000)  # Should be resampled
        self.assertEqual(waveform.shape[0], 1)  # Should be mono
        os.unlink(processed_path)
        
        # Test invalid audio
        with io.BytesIO(b"not audio") as f:
            with self.assertRaises(AudioProcessingError):
                process_audio(f)
                
    def test_hotwords_reading(self):
        """Test hotwords file reading"""
        # Test valid file
        hotword_path = os.path.join(self.temp_dir, "hotwords.txt")
        with open(hotword_path, 'w') as f:
            f.write("hello\nworld\n  test  \n")
            
        result = read_hotwords(hotword_path)
        self.assertEqual(result, "hello world test")
        
        # Test non-existent file
        result = read_hotwords("/non/existent/path")
        self.assertEqual(result, "")
        
        # Test None input
        result = read_hotwords(None)
        self.assertEqual(result, "")

    # === CLI Tests ===
    
    def test_cli_functionality(self):
        """Test essential CLI functionality"""
        from typer.testing import CliRunner
        from easy_asr_server.api import app_cli
        
        runner = CliRunner()
        
        # Test CLI with mocked server start
        with mock.patch('easy_asr_server.cli._start_uvicorn') as mock_start, \
             mock.patch('easy_asr_server.utils.setup_logging'):
            
            result = runner.invoke(app_cli, ["run", "--device", "cpu", "--pipeline", "sensevoice"])
            
            # Should set environment variables
            self.assertEqual(os.environ.get("EASY_ASR_DEVICE"), "cpu")
            self.assertEqual(os.environ.get("EASY_ASR_PIPELINE"), "sensevoice")
            mock_start.assert_called_once()
            
        # Test invalid pipeline
        with mock.patch('easy_asr_server.cli._start_uvicorn'), \
             mock.patch('easy_asr_server.utils.setup_logging'):
            
            result = runner.invoke(app_cli, ["run", "--pipeline", "invalid"])
            self.assertNotEqual(result.exit_code, 0)


if __name__ == '__main__':
    unittest.main() 