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
import torch
import torchaudio
from fastapi import HTTPException
from fastapi.testclient import TestClient

# Import the FastAPI app, dependency function, and state
from easy_asr_server.api import app, get_asr_engine, app_state, DEFAULT_PIPELINE
from easy_asr_server.asr_engine import ASREngine


# Remove the old MockASREngine class
# class MockASREngine: ...

# Create a mock ASREngine instance for dependency override tests
mock_engine_override = mock.MagicMock(spec=ASREngine)

# Override function for TestClient tests
async def override_get_asr_engine_integration() -> ASREngine:
    """Override dependency to return the mock engine, simulating health check."""
    if not mock_engine_override.test_health():
        raise HTTPException(
            status_code=503,
            detail="ASR service is currently unhealthy (mocked integration)."
        )
    return mock_engine_override


# Remove the old pytest fixture
# @pytest.fixture
# def mock_asr_engine(): ...


def test_health_endpoint_with_client():
    """Test the health endpoint works correctly using TestClient and dependency override."""
    # Reset mock and configure app state for this test
    mock_engine_override.reset_mock()
    mock_engine_override.test_health.return_value = True
    original_app_state = app_state.copy()
    app_state["pipeline_type"] = "test_pipeline_client"
    app_state["device"] = "test_device_client"
    
    app.dependency_overrides[get_asr_engine] = override_get_asr_engine_integration
    
    try:
        with TestClient(app) as client:
            response = client.get("/asr/health")
            assert response.status_code == 200
            assert response.json() == {
                "status": "healthy", 
                "pipeline": "test_pipeline_client", 
                "device": "test_device_client"
            }
            # Verify the mock engine's health check was called via the dependency
            mock_engine_override.test_health.assert_called_once()
    finally:
        # Clean up overrides and state
        app.dependency_overrides.clear()
        app_state.clear()
        app_state.update(original_app_state)


@pytest.mark.integration  # Mark as integration test
@pytest.mark.parametrize("server_host,server_port", [("127.0.0.1", 8765)])
def test_real_server_startup_and_health(server_host, server_port):
    """
    Integration test that starts a real server in a subprocess and checks health.
    This tests the lifespan initialization.
    Conditionally skipped in CI environments.
    """
    # Skip test in CI environments
    is_ci = os.environ.get("CI", "false").lower() == "true"
    if is_ci:
        pytest.skip("Skipping real server test in CI environment")
    
    # --- This test no longer uses mocks for the server process --- 
    # It will trigger real ModelManager initialization and downloads if needed.
    
    # Define server parameters
    server_kwargs = {
        "host": server_host,
        "port": server_port,
        "workers": 1,
        "device": "cpu", # Use CPU for test consistency
        "pipeline": DEFAULT_PIPELINE, # Use the default pipeline
        "log_level": "warning" # Reduce log noise during test
    }

    # Set up a process to run the server
    # Import the renamed CLI command function
    from easy_asr_server.api import run # Import here to avoid issues
    server_process = multiprocessing.Process(
        # Target the renamed function
        target=run,
        kwargs=server_kwargs,
        daemon=True # Ensure process exits if main test process crashes
    )
            
    try:
        print(f"\nStarting server subprocess for integration test ({server_host}:{server_port})...")
        server_process.start()
        
        # Wait for server to start - check health endpoint
        max_attempts = 80 # Further increased attempts/timeout
        wait_time = 1.5 # Wait longer between checks
        server_ready = False
        health_url = f"http://{server_host}:{server_port}/asr/health"
        print(f"Waiting for server at {health_url}...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(health_url, timeout=5) # Add timeout
                if response.status_code == 200:
                    print("Server responded to health check.")
                    server_ready = True
                    break
                else:
                     print(f"Attempt {attempt+1}: Server responded with status {response.status_code}")
            except requests.exceptions.ConnectionError:
                print(f"Attempt {attempt+1}: Server not ready yet (connection error)...")
            except requests.exceptions.Timeout:
                 print(f"Attempt {attempt+1}: Server not ready yet (timeout)...")
            except requests.RequestException as e:
                 print(f"Attempt {attempt+1}: Server not ready yet ({type(e).__name__})...")
            
            time.sleep(wait_time)
        
        if not server_ready:
            pytest.fail(f"Server did not become responsive at {health_url} after {max_attempts * wait_time:.1f} seconds")
        
        # Test health endpoint again now that it's confirmed ready
        print("Performing final health check...")
        response = requests.get(health_url, timeout=5)
        assert response.status_code == 200
        # Check against the parameters passed to the server process
        assert response.json() == {
            "status": "healthy", 
            "pipeline": server_kwargs["pipeline"], 
            "device": server_kwargs["device"]
        }
        print("Real server health check passed.")
        
        # Optionally: Add a test for the /asr/recognize endpoint here
        # This would require creating a test audio file and sending it.
        # Note: This will perform *real* inference unless you add specific test hooks.
        # Example:
        # with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio:
        #     _create_test_audio_file(tmp_audio.name) # Need this helper function
        #     print(f"Testing /asr/recognize with {tmp_audio.name}...")
        #     with open(tmp_audio.name, "rb") as f:
        #         recognize_response = requests.post(
        #             f"http://{server_host}:{server_port}/asr/recognize",
        #             files={"audio": (os.path.basename(tmp_audio.name), f)}
        #         )
        # assert recognize_response.status_code == 200
        # assert "text" in recognize_response.json()
        # print(f"Recognition result: {recognize_response.json()}")
        # os.remove(tmp_audio.name)

    finally:
        # Clean up the server process robustly
        print("\nTerminating server subprocess...")
        if server_process.is_alive():
            server_process.terminate() # Send SIGTERM first
            server_process.join(timeout=5) # Wait for graceful shutdown
            if server_process.is_alive():
                print("Server did not terminate gracefully, sending SIGKILL...")
                server_process.kill() # Force kill if needed
                server_process.join(timeout=5)
        print("Server subprocess terminated.")

# Helper function to create audio file (needs to be defined at module level or passed)
def _create_test_audio_file(file_path, sample_rate=16000, duration=0.5):
    """Create a short silent test audio file."""
    samples = int(sample_rate * duration)
    # Use zeros for silence, avoids potential issues with sine wave generation complexity
    wave = torch.zeros((1, samples), dtype=torch.float32) 
    torchaudio.save(file_path, wave, sample_rate)
    return file_path 