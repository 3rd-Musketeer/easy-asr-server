"""
Tests for the API module.
"""

import os
import io
import unittest
import tempfile
from unittest import mock
import torch
import torchaudio
from fastapi.testclient import TestClient

# Import the module first to avoid triggering the dependency in get_asr_engine
import easy_asr_server.api
from easy_asr_server.asr_engine import ASREngine


class TestAPI(unittest.TestCase):
    """Test cases for the API endpoints"""
    
    def setUp(self):
        """Setup for tests"""
        # Create a temporary directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Create a test audio file
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self._create_test_audio_file(self.test_audio_path)
        
        # Create a mock ASR engine
        self.mock_engine = mock.MagicMock(spec=ASREngine)
        self.mock_engine.test_health.return_value = True
        self.mock_engine.recognize.return_value = "mock recognized text"
        
        # Set the global ASR engine to our mock
        self.original_engine = easy_asr_server.api.asr_engine
        easy_asr_server.api.asr_engine = self.mock_engine
        
        # Create the test client after setting the mock engine
        self.client = TestClient(easy_asr_server.api.app)
    
    def tearDown(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
        
        # Restore the original ASR engine
        easy_asr_server.api.asr_engine = self.original_engine
    
    def _create_test_audio_file(self, file_path, sample_rate=16000, duration=1.0):
        """Create a test audio file"""
        # Generate a sine wave
        samples = int(sample_rate * duration)
        t = torch.linspace(0, duration, samples)
        wave = torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)  # 440 Hz sine wave, mono
        
        # Save the file
        torchaudio.save(file_path, wave, sample_rate)
        return file_path
    
    def test_health_endpoint(self):
        """Test the health check endpoint"""
        response = self.client.get("/asr/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy"})
        
        # Verify that the health check was called
        self.mock_engine.test_health.assert_called_once()
    
    def test_recognize_endpoint(self):
        """Test the recognize endpoint with a valid audio file"""
        # Open the test audio file
        with open(self.test_audio_path, "rb") as f:
            audio_data = f.read()
        
        # Send a request with the audio file
        response = self.client.post(
            "/asr/recognize",
            files={"audio": ("test_audio.wav", audio_data)}
        )
        
        # Check the response
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"text": "mock recognized text"})
        
        # Verify that the recognize method was called
        self.mock_engine.recognize.assert_called_once()
    
    def test_recognize_endpoint_with_invalid_audio(self):
        """Test the recognize endpoint with invalid audio data"""
        # Mock the process_audio function to raise an AudioProcessingError
        with mock.patch('easy_asr_server.api.process_audio') as mock_process:
            from easy_asr_server.utils import AudioProcessingError
            mock_process.side_effect = AudioProcessingError("Invalid audio")
            
            # Send a request with invalid audio data
            response = self.client.post(
                "/asr/recognize",
                files={"audio": ("invalid_audio.txt", b"not an audio file")}
            )
            
            # Should return a 400 bad request
            self.assertEqual(response.status_code, 400)
            self.assertIn("Invalid audio", response.json()["detail"])
    
    def test_recognize_endpoint_with_asr_error(self):
        """Test the recognize endpoint when ASR recognition fails"""
        # Set the mock engine to raise an ASREngineError
        from easy_asr_server.asr_engine import ASREngineError
        self.mock_engine.recognize.side_effect = ASREngineError("ASR failed")
        
        # Open the test audio file
        with open(self.test_audio_path, "rb") as f:
            audio_data = f.read()
        
        # Send a request with the audio file
        response = self.client.post(
            "/asr/recognize",
            files={"audio": ("test_audio.wav", audio_data)}
        )
        
        # Should return a 500 internal server error
        self.assertEqual(response.status_code, 500)
        self.assertIn("ASR failed", response.json()["detail"])
    
    def test_health_endpoint_with_unhealthy_engine(self):
        """Test the health check endpoint when the engine is unhealthy"""
        # Set the mock engine to report as unhealthy
        self.mock_engine.test_health.return_value = False
        
        # Send a request to the health endpoint
        response = self.client.get("/asr/health")
        
        # Should return a 503 service unavailable
        self.assertEqual(response.status_code, 503)
        self.assertIn("health check failed", response.json()["detail"].lower())


if __name__ == '__main__':
    unittest.main() 