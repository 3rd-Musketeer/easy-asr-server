"""
Tests for the FastAPI server endpoints.
"""

import os
import io
import unittest
import tempfile
from unittest import mock
import numpy as np
import torch
import torchaudio
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient

# Import server components
from easy_asr_server.server import app, get_asr_engine, app_state
from easy_asr_server.asr_engine import ASREngine
from easy_asr_server.utils import AudioProcessingError


# Create a mock ASREngine for dependency injection
mock_engine = mock.MagicMock(spec=ASREngine)


# Dependency override function
async def override_get_asr_engine() -> ASREngine:
    """Override the dependency to return our mock engine."""
    if not mock_engine.test_health():
        raise HTTPException(
            status_code=503,
            detail="ASR service is currently unhealthy (mocked)."
        )
    return mock_engine


class TestServerEndpoints(unittest.TestCase):
    """Test cases for the FastAPI server endpoints."""
    
    def setUp(self):
        """Setup for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self._create_test_audio_file(self.test_audio_path)
        
        # Reset mock engine
        mock_engine.reset_mock()
        mock_engine.test_health.return_value = True
        mock_engine.recognize.return_value = "mock recognized text"
        mock_engine.recognize.side_effect = None
        
        # Set mock app state
        self.original_app_state = app_state.copy()
        app_state["pipeline_type"] = "mock_pipeline"
        app_state["device"] = "mock_device"
        app_state["hotword_file_path"] = None
        
        # Apply dependency override
        app.dependency_overrides[get_asr_engine] = override_get_asr_engine
        self.client = TestClient(app)
    
    def tearDown(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
        app.dependency_overrides.clear()
        app_state.clear()
        app_state.update(self.original_app_state)
    
    def _create_test_audio_file(self, file_path, sample_rate=16000, duration=1.0):
        """Create a test audio file"""
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
        torchaudio.save(file_path, wave, sample_rate)
        return file_path
    
    def test_health_endpoint_success(self):
        """Test successful health check"""
        response = self.client.get("/asr/health")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "status": "healthy", 
            "pipeline": "mock_pipeline", 
            "device": "mock_device"
        })
        mock_engine.test_health.assert_called_once()
    
    def test_health_endpoint_unhealthy_engine(self):
        """Test health check with unhealthy engine"""
        mock_engine.test_health.return_value = False
        
        response = self.client.get("/asr/health")
        
        self.assertEqual(response.status_code, 503)
        self.assertIn("currently unhealthy", response.json()["detail"])
    
    @mock.patch('easy_asr_server.server.read_audio_bytes')
    @mock.patch('easy_asr_server.server.read_hotwords')
    def test_recognize_endpoint_success(self, mock_read_hotwords, mock_read_bytes):
        """Test successful audio recognition"""
        # Setup mocks
        test_audio_array = np.random.rand(16000).astype(np.float32)
        expected_hotword = "test hotword"
        
        mock_read_bytes.return_value = test_audio_array
        mock_read_hotwords.return_value = expected_hotword
        
        # Test recognition
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/asr/recognize",
                files={"audio": ("test_audio.wav", f, "audio/wav")}
            )
        
        # Verify response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "mock recognized text"})
        
        # Verify calls
        mock_read_bytes.assert_called_once()
        mock_read_hotwords.assert_called_once_with(None)
        mock_engine.recognize.assert_called_once()
        
        # Verify recognition was called with correct parameters
        call_args = mock_engine.recognize.call_args
        self.assertIsInstance(call_args[0][0], np.ndarray)
        self.assertEqual(call_args[1]['hotword'], expected_hotword)
    
    @mock.patch('easy_asr_server.server.read_audio_bytes')
    def test_recognize_endpoint_invalid_audio(self, mock_read_bytes):
        """Test recognition with invalid audio data"""
        mock_read_bytes.side_effect = AudioProcessingError("Invalid audio data")
        
        response = self.client.post(
            "/asr/recognize",
            files={"audio": ("invalid.wav", io.BytesIO(b"invalid data"), "audio/wav")}
        )
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("Invalid or unsupported audio file", response.json()["detail"])
        mock_engine.recognize.assert_not_called()
    
    @mock.patch('easy_asr_server.server.read_audio_bytes')
    @mock.patch('easy_asr_server.server.read_hotwords')
    def test_recognize_endpoint_engine_error(self, mock_read_hotwords, mock_read_bytes):
        """Test recognition with ASR engine error"""
        # Setup mocks
        test_audio_array = np.random.rand(16000).astype(np.float32)
        mock_read_bytes.return_value = test_audio_array
        mock_read_hotwords.return_value = ""
        mock_engine.recognize.side_effect = RuntimeError("ASR engine error")
        
        with open(self.test_audio_path, "rb") as f:
            response = self.client.post(
                "/asr/recognize",
                files={"audio": ("test_audio.wav", f, "audio/wav")}
            )
        
        self.assertEqual(response.status_code, 500)
        self.assertIn("internal engine error", response.json()["detail"].lower())
        mock_engine.recognize.assert_called_once()
    
    def test_get_hotwords_no_file_configured(self):
        """Test GET hotwords when no file is configured"""
        response = self.client.get("/asr/hotwords")
        
        self.assertEqual(response.status_code, 404)
        self.assertIn("not configured", response.json()["detail"])
    
    @mock.patch('builtins.open', mock.mock_open(read_data="word1\nword2\n\nword3\n"))
    @mock.patch('os.path.isfile')
    def test_get_hotwords_success(self, mock_isfile):
        """Test successful GET hotwords"""
        # Configure hotword file
        app_state["hotword_file_path"] = "/test/hotwords.txt"
        mock_isfile.return_value = True
        
        response = self.client.get("/asr/hotwords")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), ["word1", "word2", "word3"])
    
    @mock.patch('os.path.isfile')
    def test_get_hotwords_file_not_found(self, mock_isfile):
        """Test GET hotwords when file doesn't exist"""
        app_state["hotword_file_path"] = "/nonexistent/hotwords.txt"
        mock_isfile.return_value = False
        
        response = self.client.get("/asr/hotwords")
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [])
    
    def test_put_hotwords_no_file_configured(self):
        """Test PUT hotwords when no file is configured"""
        response = self.client.put("/asr/hotwords", json=["word1", "word2"])
        
        self.assertEqual(response.status_code, 400)
        self.assertIn("not configured", response.json()["detail"])
    
    @mock.patch('builtins.open', mock.mock_open())
    def test_put_hotwords_success(self):
        """Test successful PUT hotwords"""
        app_state["hotword_file_path"] = "/test/hotwords.txt"
        hotwords = ["word1", "word2", "word3"]
        
        response = self.client.put("/asr/hotwords", json=hotwords)
        
        self.assertEqual(response.status_code, 204)
    
    def test_put_hotwords_invalid_format(self):
        """Test PUT hotwords with invalid format"""
        app_state["hotword_file_path"] = "/test/hotwords.txt"
        
        # Test various invalid formats
        invalid_payloads = [
            "not a list",
            ["valid", 123],  # Non-string in list
            {"invalid": "dict"}
        ]
        
        for payload in invalid_payloads:
            response = self.client.put("/asr/hotwords", json=payload)
            self.assertEqual(response.status_code, 422)


if __name__ == '__main__':
    unittest.main() 