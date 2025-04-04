"""
Integration tests for the ASR server.

These tests start the FastAPI application and test the endpoints to ensure they work correctly.
"""

import os
import unittest
import pytest
import tempfile
from unittest import mock
import multiprocessing
import time
import requests
from fastapi.testclient import TestClient

from easy_asr_server.api import app, asr_engine
from easy_asr_server.asr_engine import ASREngine


class MockASREngine:
    """Mock for ASREngine class that always passes health checks"""
    
    def __init__(self, *args, **kwargs):
        self.asr_model_path = "mock_model_path"
        self.vad_model_path = "mock_vad_model_path"
        self.device = "cpu"
    
    def test_health(self):
        """Always return healthy"""
        return True
    
    def recognize(self, waveform, sample_rate):
        """Return a mock transcription"""
        return "mock transcription"
    
    @classmethod
    def create(cls, device="auto"):
        """Create a mock ASREngine"""
        return cls()


@pytest.fixture
def mock_asr_engine():
    """Fixture to mock the ASR engine"""
    # Save the original module
    original_engine = ASREngine
    
    # Replace with our mock
    from easy_asr_server import api
    api.ASREngine = MockASREngine
    api.asr_engine = MockASREngine()
    
    try:
        yield api.asr_engine
    finally:
        # Restore the original module
        api.ASREngine = original_engine
        api.asr_engine = None


def test_health_endpoint(mock_asr_engine):
    """Test that the health endpoint works correctly"""
    with TestClient(app) as client:
        response = client.get("/asr/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


@pytest.mark.parametrize("server_host,server_port", [("127.0.0.1", 8765)])
def test_real_server(server_host, server_port):
    """
    Integration test that starts a real server in a subprocess.
    
    This test is conditionally skipped in CI environments.
    """
    # Skip test in CI environments
    is_ci = os.environ.get("CI", "false").lower() == "true"
    if is_ci:
        pytest.skip("Skipping real server test in CI environment")
    
    # Apply the mock for the ASR engine
    with mock.patch("easy_asr_server.api.ASREngine", MockASREngine):
        with mock.patch("easy_asr_server.asr_engine.ASREngine", MockASREngine):
            # Set up a process to run the server
            from easy_asr_server.api import run_server
            server_process = multiprocessing.Process(
                target=run_server,
                kwargs={"host": server_host, "port": server_port, "workers": 1, "device": "cpu"}
            )
            
            try:
                # Start the server
                server_process.start()
                
                # Wait for server to start (adjust timeout as needed)
                max_attempts = 10
                for attempt in range(max_attempts):
                    try:
                        # Check if server is responsive
                        response = requests.get(f"http://{server_host}:{server_port}/asr/health")
                        if response.status_code == 200:
                            break
                    except requests.RequestException:
                        # Server not ready yet, wait a bit
                        time.sleep(0.5)
                    
                    if attempt == max_attempts - 1:
                        pytest.fail("Server did not become responsive")
                
                # Test health endpoint
                response = requests.get(f"http://{server_host}:{server_port}/asr/health")
                assert response.status_code == 200
                assert response.json() == {"status": "healthy"}
                
            finally:
                # Clean up the server process
                server_process.terminate()
                server_process.join(timeout=2)
                
                # Kill it if it hasn't terminated
                if server_process.is_alive():
                    server_process.kill()
                    server_process.join() 