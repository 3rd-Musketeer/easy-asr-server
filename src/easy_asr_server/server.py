"""
Server Module for easy_asr_server

This module defines the FastAPI application and server-related functionality.
It handles HTTP requests for speech recognition and health checks.
"""

import os
import logging
from typing import Dict, Optional, List, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse
from starlette.status import (
    HTTP_400_BAD_REQUEST,
    HTTP_500_INTERNAL_SERVER_ERROR,
    HTTP_503_SERVICE_UNAVAILABLE,
    HTTP_422_UNPROCESSABLE_ENTITY,
    HTTP_404_NOT_FOUND
)

from .asr_engine import ASREngine
from .model_manager import ModelManager, MODEL_CONFIGS, DEFAULT_PIPELINE
from .utils import process_audio, AudioProcessingError, setup_logging, read_hotwords, resolve_device_string, read_audio_bytes

# Configure logging
logger = logging.getLogger(__name__)

# Global state dictionary (better practice than global variables)
app_state: Dict[str, Any] = {
    "model_manager": None,
    "asr_engine": None,
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for handling startup and shutdown events.
    Loads the pipeline via ModelManager and initializes ASREngine.
    """
    # Startup: Initialize ModelManager and ASREngine
    logger.info("Starting up ASR server worker...")
    
    # --- Configuration setup within worker --- 
    # Read configurations from environment variables set by the main process
    log_level_str = os.environ.get("EASY_ASR_LOG_LEVEL", "INFO")
    setup_logging(level=logging._checkLevel(log_level_str)) # Use _checkLevel for str->int
    
    app_state["hotword_file_path"] = os.environ.get("EASY_ASR_HOTWORD_FILE")
    if app_state["hotword_file_path"]:
        logger.info(f"Worker using hotword file from environment: {app_state['hotword_file_path']}")
    else:
        logger.info("Worker started without a configured hotword file.")
        
    # Existing pipeline and device setup (already uses app_state)
    pipeline_type = os.environ.get("EASY_ASR_PIPELINE", DEFAULT_PIPELINE)
    requested_device = os.environ.get("EASY_ASR_DEVICE", "auto")
    app_state["pipeline_type"] = pipeline_type # Store for health check etc.
    
    resolved_device = None # Initialize
    
    try:
        # Resolve the device string (e.g., "auto" -> "cuda" or "cpu")
        resolved_device = resolve_device_string(requested_device)
        app_state["device"] = resolved_device # Store the actual device being used
        
        logger.info(f"Initializing ModelManager and loading pipeline '{pipeline_type}' on resolved device '{resolved_device}'...")
        model_manager = ModelManager()
        # Pass the resolved device string
        model_manager.load_pipeline(pipeline_type=pipeline_type, device=resolved_device)
        app_state["model_manager"] = model_manager # Store instance in state
        
        logger.info("Initializing ASREngine...")
        asr_engine = ASREngine(model_manager=model_manager)
        app_state["asr_engine"] = asr_engine # Store instance in state
        
        logger.info("ASR Engine and Model Manager initialized successfully in worker process.")
        
    except ValueError as e: # Catch errors from resolve_device_string
        logger.error(f"Fatal error during worker startup - Invalid device configuration: {str(e)}", exc_info=True)
        app_state["asr_engine"] = None
        app_state["model_manager"] = None 
    except Exception as e:
        logger.error(f"Fatal error during worker startup: {str(e)}", exc_info=True)
        # Optionally, could set a flag indicating the worker is unhealthy
        app_state["asr_engine"] = None # Ensure engine is None on failure
        app_state["model_manager"] = None 
        # Depending on deployment, might want to exit or signal failure

    yield  # Server is running here
    
    # Shutdown: Clean up resources
    logger.info("Shutting down ASR server worker.")
    app_state["asr_engine"] = None
    app_state["model_manager"] = None
    # Any other cleanup needed for ModelManager or ASREngine? (e.g., releasing GPU memory explicitly?)


# Create the FastAPI app
app = FastAPI(
    title="Easy ASR Server",
    description="A simple high-concurrency speech recognition service based on FunASR",
    version="0.2.0",
    lifespan=lifespan
)


# Dependency to get the initialized ASREngine instance
async def get_asr_engine() -> ASREngine:
    """
    Dependency to get the ASREngine from the application state.
    
    Returns:
        ASREngine: The ASREngine instance.
        
    Raises:
        HTTPException: If the engine is not available or unhealthy.
    """
    asr_engine_instance = app_state.get("asr_engine")
    if asr_engine_instance is None:
        logger.error("ASR engine not available in app state.")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR service is not ready. Initialization might have failed or is in progress."
        )
    
    # Perform health check via the engine
    if not asr_engine_instance.test_health():
        logger.error("ASR engine health check failed.")
        raise HTTPException(
            status_code=HTTP_503_SERVICE_UNAVAILABLE,
            detail="ASR service is currently unhealthy."
        )
        
    return asr_engine_instance


@app.post("/asr/recognize", response_model=Dict[str, str])
async def recognize_audio(
    audio: UploadFile = File(...),
    engine: ASREngine = Depends(get_asr_engine)
) -> Dict[str, str]:
    """
    Recognize speech in the uploaded audio file.
    
    Args:
        audio: The audio file to recognize (expects standard audio formats).
        engine: The ASREngine instance (injected dependency).
        
    Returns:
        Dict[str, str]: JSON response containing the recognition results 
                        (structure depends on the loaded pipeline's output, 
                         typically a list of segments with 'text').
        
    Raises:
        HTTPException: If audio processing or recognition fails.
    """
    logger.debug(f"Received request for /asr/recognize with file: {audio.filename}")
    try:
        # Process the uploaded audio file using the new bytes-based utility function
        # read_audio_bytes handles validation, conversion, and returns numpy array
        audio_array = read_audio_bytes(audio.file) # Pass the file-like object
        logger.debug(f"Audio successfully processed to numpy array: shape={audio_array.shape}")
        
        # Read hotwords based on configured path
        hotword_file_path = app_state.get("hotword_file_path")
        hotword_string = read_hotwords(hotword_file_path)
        logger.debug(f"Using hotwords: '{hotword_string[:50]}...'" if hotword_string else "Using no hotwords.")

        # Pass numpy array to engine
        results = engine.recognize(audio_array, hotword=hotword_string)
        logger.debug("Recognition call completed.")

        # Return the string result in the expected format
        return {"text": results} 
        
    except AudioProcessingError as e:
        logger.warning(f"Audio processing error for file '{audio.filename}': {str(e)}")
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=f"Invalid or unsupported audio file: {str(e)}"
        )
        
    except RuntimeError as e:
        # Catch generic runtime errors from ASREngine/ModelManager
        logger.error(f"ASR engine runtime error during recognition: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Speech recognition failed due to an internal engine error."
        )
        
    except HTTPException: # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during recognition: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during processing."
        )


@app.get("/asr/health", response_model=Dict[str, str])
async def health_check(engine: ASREngine = Depends(get_asr_engine)) -> Dict[str, str]:
    """
    Performs a health check on the ASR service.
    Relies on the `get_asr_engine` dependency to perform the actual check.
    """
    # If the dependency resolved without raising an exception, the engine is healthy.
    pipeline_type = app_state.get("pipeline_type", "unknown")
    device = app_state.get("device", "unknown")
    logger.info(f"Health check passed for pipeline '{pipeline_type}' on device '{device}'.")
    return {"status": "healthy", "pipeline": pipeline_type, "device": device}


@app.get("/asr/hotwords", response_model=List[str])
async def get_hotwords() -> List[str]:
    """Gets the current list of hotwords from the configured file."""
    hotword_file_path = app_state.get("hotword_file_path")
    if not hotword_file_path:
        raise HTTPException(
            status_code=HTTP_404_NOT_FOUND, 
            detail="Hotword file not configured for this server instance."
        )
        
    try:
        if not os.path.isfile(hotword_file_path):
             logger.warning(f"Configured hotword file not found at: {hotword_file_path}")
             return [] # Return empty list if file doesn't exist
             
        with open(hotword_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        hotwords = [line.strip() for line in lines if line.strip()]
        logger.info(f"Retrieved {len(hotwords)} hotwords from {hotword_file_path}")
        return hotwords
        
    except IOError as e:
        logger.error(f"Error reading hotword file {hotword_file_path} for GET request: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read hotword file: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error reading hotword file {hotword_file_path}: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while reading hotwords."
        )


@app.put("/asr/hotwords", status_code=204) # 204 No Content on success
async def update_hotwords(hotwords: List[str]):
    """Updates the hotword file with the provided list, overwriting existing content."""
    hotword_file_path = app_state.get("hotword_file_path")
    if not hotword_file_path:
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST, # 400 because the server isn't configured for this
            detail="Hotword file not configured. Cannot update hotwords."
        )

    # Basic validation (ensure it's a list of strings)
    if not isinstance(hotwords, list) or not all(isinstance(item, str) for item in hotwords):
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Invalid input format. Expected a JSON list of strings."
        )

    # Format the list back into newline-separated strings
    # Filter empty strings just in case
    formatted_content = "\n".join([hw.strip() for hw in hotwords if hw.strip()]) + "\n" # Add trailing newline
    
    try:
        # Overwrite the file
        with open(hotword_file_path, 'w', encoding='utf-8') as f:
            f.write(formatted_content)
        logger.info(f"Successfully updated hotwords in {hotword_file_path} with {len(hotwords)} entries.")
        # No content to return on success (HTTP 204)
        return
    except IOError as e:
        logger.error(f"Error writing hotword file {hotword_file_path}: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to write hotword file: {e}"
        )
    except Exception as e:
        logger.error(f"Unexpected error writing hotword file {hotword_file_path}: {e}")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while updating hotwords."
        ) 