"""
API Module for easy_asr_server

This module defines the FastAPI application and CLI entry point for the ASR server.
It handles HTTP requests for speech recognition and health checks.
"""

import os
import sys
import logging
from typing import Dict, Optional, List, Any
from contextlib import asynccontextmanager

import click
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE
)

from .asr_engine import ASREngine, ASREngineError
from .utils import process_audio, AudioProcessingError, setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# Global variable to hold the ASR engine instance
asr_engine: Optional[ASREngine] = None

# Configuration for ASR engine initialization
engine_config = {
    "device": "auto"
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for handling startup and shutdown events.
    """
    global asr_engine
    
    # Startup: Initialize ASR engine
    if asr_engine is None:
        try:
            logger.info(f"Initializing ASR engine in worker process with device={engine_config['device']}")
            asr_engine = ASREngine.create(device=engine_config["device"])
            logger.info("ASR engine initialized successfully in worker process")
        except Exception as e:
            logger.error(f"Failed to initialize ASR engine in worker process: {str(e)}")
            # We don't exit here, as that would kill the worker process
    
    yield  # Server is running here
    
    # Shutdown: Clean up resources
    logger.info("Shutting down the ASR server")
    asr_engine = None


# Create the FastAPI app
app = FastAPI(
    title="Easy ASR Server",
    description="A simple high-concurrency speech recognition service based on FunASR",
    version="0.1.0",
    lifespan=lifespan
)


async def get_asr_engine() -> ASREngine:
    """
    Dependency to get the ASR engine.
    
    Returns:
        ASREngine: The ASR engine instance
        
    Raises:
        HTTPException: If the ASR engine is not initialized or not healthy
    """
    if asr_engine is None:
        logger.error("ASR engine not initialized")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR engine not initialized. Service is starting up or encountered an error."
        )
    
    if not asr_engine.test_health():
        logger.error("ASR engine not healthy")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR engine health check failed."
        )
    
    return asr_engine


@app.post("/asr/recognize")
async def recognize_audio(
    audio: UploadFile = File(...),
    engine: ASREngine = Depends(get_asr_engine)
) -> Dict[str, str]:
    """
    Recognize speech in the uploaded audio file.
    
    Args:
        audio: The audio file to recognize
        engine: The ASR engine (injected via dependency)
        
    Returns:
        Dict[str, str]: JSON response with the recognized text
        
    Raises:
        HTTPException: If audio processing or recognition fails
    """
    try:
        # Process the uploaded audio file
        contents = await audio.read()
        
        # Create a file-like object from the contents
        import io
        file_data = io.BytesIO(contents)
        
        try:
            # Process audio (validates and converts to required format)
            waveform, sample_rate = process_audio(file_data)
            
            # Perform speech recognition
            text = engine.recognize(waveform, sample_rate)
            
            return {"text": text}
            
        except AudioProcessingError as e:
            logger.warning(f"Audio processing error: {str(e)}")
            raise HTTPException(
                status_code=HTTP_400_BAD_REQUEST,
                detail=f"Invalid audio file: {str(e)}"
            )
            
        except ASREngineError as e:
            logger.error(f"ASR engine error: {str(e)}")
            raise HTTPException(
                status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Speech recognition failed: {str(e)}"
            )
            
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    finally:
        # Reset file position
        await audio.seek(0)


@app.get("/asr/health")
async def health_check(engine: ASREngine = Depends(get_asr_engine)) -> Dict[str, str]:
    """
    Check if the ASR engine is healthy.
    
    Args:
        engine: The ASR engine (injected via dependency)
        
    Returns:
        Dict[str, str]: JSON response with the health status
    """
    # If we got here, the dependency already verified the engine is healthy
    return {"status": "healthy"}


def run_server(
    host: str = "127.0.0.1",
    port: int = 8000,
    workers: int = 1,
    device: str = "auto"
):
    """
    Run the ASR server with the specified options.
    
    Args:
        host: Host to bind the server to
        port: Port to bind the server to
        workers: Number of worker processes
        device: Device to use for inference ('auto', 'cpu', 'cuda')
    """
    global engine_config
    
    # Configure logging
    setup_logging()
    
    try:
        # Store engine configuration for later use by workers
        engine_config["device"] = device
        
        # Run the server
        logger.info(f"Starting ASR server on {host}:{port} with {workers} workers")
        uvicorn.run(
            "easy_asr_server.api:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info"
        )
    except Exception as e:
        logger.error(f"Failed to start ASR server: {str(e)}")
        sys.exit(1)


# Simple CLI interface using click
@click.group()
def cli():
    """Easy ASR Server - A simple high-concurrency speech recognition service."""
    pass


@cli.command("run")
@click.option('--host', default="127.0.0.1", help="Host to bind the server to")
@click.option('--port', default=8000, help="Port to bind the server to")
@click.option('--workers', default=1, help="Number of worker processes")
@click.option('--device', default="auto", help="Device to use ('auto', 'cpu', 'cuda')")
def run_command(host, port, workers, device):
    """Run the ASR server."""
    run_server(host=host, port=port, workers=workers, device=device)


def main():
    """Entry point for the CLI application."""
    # If no arguments are provided, run with defaults
    if len(sys.argv) == 1:
        run_server()
    else:
        cli()


if __name__ == "__main__":
    main()
