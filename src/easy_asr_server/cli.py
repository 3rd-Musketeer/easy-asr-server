"""
CLI Module for easy_asr_server

This module defines the command-line interface for the ASR server.
"""

import os
import sys
import logging
from typing import Optional

import typer
import uvicorn

from .model_manager import MODEL_CONFIGS, DEFAULT_PIPELINE, ModelManager
from .utils import setup_logging

# Configure logging
logger = logging.getLogger(__name__)

# Create Typer app for CLI
app_cli = typer.Typer()


def _start_uvicorn(host: str, port: int, workers: int, log_level: str):
    """Internal function to start the Uvicorn server."""
    try:
        uvicorn.run(
            "easy_asr_server.server:app", # Point to the FastAPI app instance in server module
            host=host,
            port=port,
            workers=workers,
            log_level=log_level.lower(),
            reload=False # Typically False for production/stable runs
            # lifespan="on" is default with lifespan context manager
        )
    except Exception as e:
        # Log error from the actual Uvicorn runner
        logger.error(f"Failed to run Uvicorn server: {str(e)}", exc_info=True)
        sys.exit(1)


# Use function name as command: 'run'
@app_cli.command()
def run(
    host: str = typer.Option("127.0.0.1", help="Host address to bind the server to."),
    port: int = typer.Option(8000, help="Port number to bind the server to."),
    workers: int = typer.Option(1, help="Number of Uvicorn worker processes."),
    device: str = typer.Option("auto", help="Device for model inference ('auto', 'cpu', 'cuda', 'cuda:0', etc.)."),
    pipeline: str = typer.Option(
        DEFAULT_PIPELINE, 
        help=f"ASR pipeline type to use. Choices: {list(MODEL_CONFIGS.keys())}"
    ),
    log_level: str = typer.Option("info", help="Log level (e.g., debug, info, warning, error)."),
    hotword_file: Optional[str] = typer.Option(None, "--hotword-file", "-hf", help="Path to a file containing hotwords (one per line).")
):
    """
    Starts the Easy ASR FastAPI server with specified configurations.
    """
    # Validate pipeline choice
    if pipeline not in MODEL_CONFIGS:
        print(f"Error: Invalid pipeline type '{pipeline}'. Choices are: {list(MODEL_CONFIGS.keys())}")
        raise typer.Exit(code=1)

    # Setup logging for the main process (workers set up their own via lifespan)
    setup_logging(level=log_level.upper())
    
    logger.info(f"Preparing to start ASR server...")
    logger.info(f"  Host: {host}")
    logger.info(f"  Port: {port}")
    logger.info(f"  Workers: {workers}")
    logger.info(f"  Device: {device}")
    logger.info(f"  Pipeline: {pipeline}")
    logger.info(f"  Log Level: {log_level}")

    # Set environment variables for worker configuration BEFORE starting Uvicorn
    os.environ["EASY_ASR_PIPELINE"] = pipeline
    os.environ["EASY_ASR_DEVICE"] = device
    os.environ["EASY_ASR_LOG_LEVEL"] = log_level.upper()
    
    if hotword_file:
        logger.info(f"  Setting EASY_ASR_HOTWORD_FILE environment variable to: {hotword_file}")
        os.environ["EASY_ASR_HOTWORD_FILE"] = hotword_file
    else:
        # Ensure env var is not set if option is not provided (or explicitly cleared if needed)
        os.environ.pop("EASY_ASR_HOTWORD_FILE", None)

    # Call the separate function to run Uvicorn
    _start_uvicorn(host=host, port=port, workers=workers, log_level=log_level)


@app_cli.command()
def download(
    pipeline: str = typer.Argument(
        help=f"Pipeline to download models for. Choices: {list(MODEL_CONFIGS.keys()) + ['all']}"
    ),
    log_level: str = typer.Option("info", help="Log level (e.g., debug, info, warning, error).")
):
    """
    Downloads the models for the specified ASR pipeline.
    
    PIPELINE can be one of the available pipeline types or 'all' to download all pipelines.
    """
    # Setup logging
    setup_logging(level=log_level.upper())
    
    # Validate pipeline choice
    valid_choices = list(MODEL_CONFIGS.keys()) + ["all"]
    if pipeline not in valid_choices:
        print(f"Error: Invalid pipeline '{pipeline}'. Choices are: {valid_choices}")
        raise typer.Exit(code=1)
    
    # Determine which pipelines to download
    if pipeline == "all":
        pipelines_to_download = list(MODEL_CONFIGS.keys())
        logger.info("Downloading models for all available pipelines...")
    else:
        pipelines_to_download = [pipeline]
        logger.info(f"Downloading models for pipeline: {pipeline}")
    
    # Initialize ModelManager
    try:
        model_manager = ModelManager()
        
        # Download models for each pipeline
        for pipeline_name in pipelines_to_download:
            logger.info(f"Processing pipeline: {pipeline_name}")
            config = MODEL_CONFIGS[pipeline_name]
            components = config.get("components", {})
            
            if not components:
                logger.warning(f"No components defined for pipeline '{pipeline_name}', skipping.")
                continue
            
            # Download each component
            for component_type, model_id in components.items():
                logger.info(f"Downloading {component_type} model: {model_id}")
                try:
                    path = model_manager.get_model_path(model_id)
                    logger.info(f"✓ {component_type} model downloaded to: {path}")
                except Exception as e:
                    logger.error(f"✗ Failed to download {component_type} model ({model_id}): {str(e)}")
                    # Continue with other components rather than failing completely
                    continue
        
        logger.info("Download process completed.")
        print("Model download completed successfully!")
        
    except Exception as e:
        logger.error(f"Failed to download models: {str(e)}")
        print(f"Error: Failed to download models: {str(e)}")
        raise typer.Exit(code=1)


def main():
    """Main entry point for the CLI."""
    app_cli()


if __name__ == "__main__":
    main() 