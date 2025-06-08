"""
Integration tests for the ASR server and high-level API.

These tests verify that components work together correctly in realistic scenarios.
"""

import os
import unittest
import pytest
import tempfile
from unittest import mock
import subprocess
import sys
import time
import requests
import torch
import torchaudio
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import server components
from easy_asr_server.server import app, get_asr_engine, app_state
from easy_asr_server.asr_engine import ASREngine
from easy_asr_server.model_manager import DEFAULT_PIPELINE

# Import high-level API
from easy_asr_server import EasyASR, recognize


# Mock engine for TestClient tests
mock_engine_override = mock.MagicMock(spec=ASREngine)

async def override_get_asr_engine_integration() -> ASREngine:
    """Override dependency to return the mock engine."""
    if not mock_engine_override.test_health():
        raise HTTPException(
            status_code=503,
            detail="ASR service is currently unhealthy (mocked integration)."
        )
    return mock_engine_override


class TestServerIntegration(unittest.TestCase):
    """Integration tests for the FastAPI server."""
    
    def test_health_endpoint_with_mocked_lifespan(self):
        """Test the health endpoint with mocked lifespan initialization."""
        # Reset mock and configure for this test
        mock_engine_override.reset_mock()
        mock_engine_override.test_health.return_value = True
        original_app_state = app_state.copy()
        
        test_pipeline = "test_pipeline_client"
        test_device = "cpu"
        
        app.dependency_overrides[get_asr_engine] = override_get_asr_engine_integration
        
        # Mock environment variables for lifespan
        mock_env = {
            "EASY_ASR_PIPELINE": test_pipeline,
            "EASY_ASR_DEVICE": test_device,
            "EASY_ASR_LOG_LEVEL": "WARNING"
        }
        
        try:
            with mock.patch.dict(os.environ, mock_env, clear=True), \
                 TestClient(app) as client:
                     
                response = client.get("/asr/health")
                assert response.status_code == 200
                assert response.json() == {
                    "status": "healthy", 
                    "pipeline": test_pipeline, 
                    "device": test_device
                }
                mock_engine_override.test_health.assert_called_once()
        finally:
            app.dependency_overrides.clear()
            app_state.clear()
            app_state.update(original_app_state)


class TestHighLevelAPIIntegration(unittest.TestCase):
    """Integration tests for the high-level API."""
    
    def setUp(self):
        """Setup for tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_audio_path = os.path.join(self.temp_dir, "test_audio.wav")
        self._create_test_audio_file(self.test_audio_path)
    
    def tearDown(self):
        """Cleanup after tests"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_test_audio_file(self, file_path, sample_rate=16000, duration=0.5):
        """Create a short test audio file."""
        samples = int(sample_rate * duration)
        wave = torch.zeros((1, samples), dtype=torch.float32)
        torchaudio.save(file_path, wave, sample_rate)
        return file_path
    
    @mock.patch('easy_asr_server.api.ModelManager')
    @mock.patch('easy_asr_server.api.ASREngine')
    def test_easy_asr_full_workflow(self, mock_asr_engine_class, mock_model_manager_class):
        """Test the complete EasyASR workflow from initialization to recognition."""
        # Setup mocks
        mock_model_manager = mock.MagicMock()
        mock_model_manager_class.return_value = mock_model_manager
        
        mock_asr_engine = mock.MagicMock()
        mock_asr_engine.test_health.return_value = True
        mock_asr_engine.recognize.return_value = "integration test result"
        mock_asr_engine_class.return_value = mock_asr_engine
        
        # Test complete workflow
        with EasyASR(
            pipeline="sensevoice", 
            device="cpu", 
            hotwords="integration test"
        ) as asr:
            # Verify initialization
            self.assertTrue(asr.is_healthy())
            
            # Test file recognition
            result1 = asr.recognize(self.test_audio_path)
            self.assertEqual(result1, "integration test result")
            
            # Test array recognition
            import numpy as np
            test_array = np.random.rand(16000).astype(np.float32)
            result2 = asr.recognize(test_array, hotwords="override hotwords")
            self.assertEqual(result2, "integration test result")
            
            # Verify get_info works
            info = asr.get_info()
            self.assertEqual(info["pipeline"], "sensevoice")
            self.assertEqual(info["device"], "cpu")
            self.assertTrue(info["initialized"])
            self.assertTrue(info["healthy"])
        
        # Verify all expected calls were made
        mock_model_manager.load_pipeline.assert_called_once_with(
            pipeline_type="sensevoice", 
            device="cpu"
        )
        self.assertEqual(mock_asr_engine.recognize.call_count, 2)
        mock_asr_engine.test_health.assert_called()
    
    @mock.patch('easy_asr_server.api.EasyASR')
    def test_convenience_functions_integration(self, mock_easy_asr_class):
        """Test that convenience functions work together properly."""
        # Setup context manager mock properly
        mock_asr = mock.MagicMock()
        mock_asr.recognize.return_value = "convenience result"
        mock_asr.__enter__.return_value = mock_asr
        mock_asr.__exit__.return_value = None
        mock_easy_asr_class.return_value = mock_asr
        
        # Test one-shot recognition
        result = recognize(
            audio_input=self.test_audio_path,
            pipeline="paraformer",
            device="auto",
            hotwords="convenience test"
        )
        
        self.assertEqual(result, "convenience result")
        mock_easy_asr_class.assert_called_once_with(
            pipeline="paraformer",
            device="auto",
            hotwords="convenience test"
        )
        mock_asr.__enter__.assert_called_once()
        mock_asr.recognize.assert_called_once_with(self.test_audio_path)
        mock_asr.__exit__.assert_called_once()


@pytest.mark.integration
@pytest.mark.parametrize("server_host,server_port", [("127.0.0.1", 8765)])
def test_real_server_startup_and_health(server_host, server_port):
    """
    Integration test that starts a real server and checks health.
    This is a comprehensive test that verifies the entire server startup process.
    """
    # Skip test in CI environments to avoid model download issues
    is_ci = os.environ.get("CI", "false").lower() == "true"
    if is_ci:
        pytest.skip("Skipping real server test in CI environment")
    
    # Server command arguments
    server_args = {
        "--host": server_host,
        "--port": str(server_port),
        "--workers": "1",
        "--device": "cpu",  # Use CPU to avoid GPU requirements
        "--pipeline": DEFAULT_PIPELINE,
        "--log-level": "info"
    }

    cmd = [sys.executable, "-m", "easy_asr_server.cli", "run"]
    for key, value in server_args.items():
        cmd.extend([key, value])

    server_process = None
    try:
        print(f"\nStarting server subprocess: {' '.join(cmd)}")
        server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for server to start
        max_attempts = 60  # Reduced for faster tests
        wait_time = 2.0
        server_ready = False
        health_url = f"http://{server_host}:{server_port}/asr/health"
        
        print(f"Waiting for server at {health_url}...")
        for attempt in range(max_attempts):
            # Check if process exited prematurely
            if server_process.poll() is not None:
                stdout, stderr = server_process.communicate()
                pytest.fail(
                    f"Server process exited prematurely with code {server_process.returncode}.\n"
                    f"STDOUT:\n{stdout}\nSTDERR:\n{stderr}"
                )
                
            try:
                response = requests.get(health_url, timeout=5)
                if response.status_code == 200:
                    print("Server responded to health check.")
                    server_ready = True
                    break
                else:
                    print(f"Attempt {attempt+1}: Server responded with status {response.status_code}")
            except requests.exceptions.RequestException as e:
                print(f"Attempt {attempt+1}: Server not ready yet ({type(e).__name__})...")
            
            time.sleep(wait_time)
        
        if not server_ready:
            pytest.fail(f"Server did not become responsive after {max_attempts * wait_time:.1f} seconds")
        
        # Verify health endpoint response
        print("Performing final health check...")
        response = requests.get(health_url, timeout=5)
        assert response.status_code == 200
        
        response_data = response.json()
        assert response_data["status"] == "healthy"
        assert response_data["pipeline"] == server_args["--pipeline"]
        assert response_data["device"] == server_args["--device"]
        
        print("Real server integration test passed.")

    finally:
        # Clean up server process
        print("\nTerminating server subprocess...")
        if server_process and server_process.poll() is None:
            try:
                server_process.terminate()
                stdout, stderr = server_process.communicate(timeout=10)
                print("Server process terminated gracefully.")
            except subprocess.TimeoutExpired:
                print("Server did not terminate gracefully, force killing...")
                server_process.kill()
                stdout, stderr = server_process.communicate(timeout=5)
                print("Server process killed.")
            except Exception as e:
                print(f"Error during server cleanup: {e}")
        elif server_process:
            print(f"Server process already exited with code {server_process.returncode}.")
        else:
            print("Server process was not started.")


def _create_test_audio_file(file_path, sample_rate=16000, duration=0.5):
    """Helper function to create a test audio file."""
    samples = int(sample_rate * duration)
    wave = torch.zeros((1, samples), dtype=torch.float32)
    torchaudio.save(file_path, wave, sample_rate)
    return file_path 