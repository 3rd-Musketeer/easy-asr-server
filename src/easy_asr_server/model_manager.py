"""
Model Manager Module for easy_asr_server

This module is responsible for downloading and managing the ASR and VAD models.
It provides a singleton ModelManager class to ensure thread-safe model downloading
and path management.
"""

import os
import logging
import threading
import filelock
from typing import Optional, Dict
from modelscope import snapshot_download

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Default model IDs
DEFAULT_ASR_MODEL_ID = "iic/SenseVoiceSmall"
DEFAULT_VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"

# Default cache directory for models
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/easy_asr_server/models")


class ModelManager:
    """
    Singleton class for managing ASR and VAD models.
    Handles downloading, caching, and providing model paths.
    """
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Ensure only one instance of ModelManager exists (Singleton pattern)"""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self):
        """Initialize the ModelManager if not already initialized"""
        if not self._initialized:
            self._model_paths: Dict[str, str] = {}
            self._download_locks: Dict[str, threading.Lock] = {}
            self._cache_dir = DEFAULT_CACHE_DIR
            os.makedirs(self._cache_dir, exist_ok=True)
            self._initialized = True
    
    def get_model_path(self, model_id: str) -> str:
        """
        Get the local path for a model. Downloads the model if not available.
        
        Args:
            model_id: The ID of the model to get the path for
            
        Returns:
            str: Local path to the model
        """
        if model_id not in self._model_paths:
            self._model_paths[model_id] = self.download_model(model_id)
        
        return self._model_paths[model_id]
    
    def download_model(self, model_id: str) -> str:
        """
        Download a model using modelscope's snapshot_download if not already available.
        Uses file locks to prevent concurrent downloads of the same model.
        
        Args:
            model_id: The ID of the model to download
            
        Returns:
            str: Local path to the downloaded model
        """
        # Create a model-specific lock if it doesn't exist
        if model_id not in self._download_locks:
            self._download_locks[model_id] = threading.Lock()
        
        # Create a directory for this specific model in the cache
        model_dir = os.path.join(self._cache_dir, model_id.replace('/', '_'))
        os.makedirs(model_dir, exist_ok=True)
        
        # Path to the lock file
        lock_file = os.path.join(model_dir, "download.lock")
        
        # Check if the model is already downloaded
        model_complete_file = os.path.join(model_dir, "download_complete")
        model_path = os.path.join(model_dir, "model")
        
        # If the model is already downloaded, return its path
        if os.path.exists(model_complete_file):
            logger.info(f"Model {model_id} already downloaded at {model_path}")
            return model_path
        
        # Try to acquire the model-specific thread lock
        with self._download_locks[model_id]:
            # Use a file lock for process-level synchronization
            with filelock.FileLock(lock_file):
                # Check again in case another process downloaded it while we were waiting
                if os.path.exists(model_complete_file):
                    logger.info(f"Model {model_id} already downloaded at {model_path}")
                    return model_path
                
                logger.info(f"Downloading model {model_id}...")
                try:
                    # Download the model using modelscope
                    download_path = snapshot_download(model_id, cache_dir=model_dir)
                    
                    # Create the model path symlink for convenience
                    if os.path.exists(model_path):
                        os.remove(model_path)
                    os.symlink(download_path, model_path)
                    
                    # Create a marker file to indicate download is complete
                    with open(model_complete_file, 'w') as f:
                        f.write(f"Downloaded {model_id} to {download_path}")
                    
                    logger.info(f"Model {model_id} downloaded successfully to {model_path}")
                    return model_path
                    
                except Exception as e:
                    logger.error(f"Failed to download model {model_id}: {str(e)}")
                    raise
    
    def ensure_models_downloaded(self) -> Dict[str, str]:
        """
        Ensure both ASR and VAD models are downloaded and return their paths.
        
        Returns:
            Dict[str, str]: Dictionary mapping model types to their local paths
        """
        result = {
            "asr": self.get_model_path(DEFAULT_ASR_MODEL_ID),
            "vad": self.get_model_path(DEFAULT_VAD_MODEL_ID)
        }
        return result
    
    def clear_cache(self) -> None:
        """Clear the model cache (mainly for testing purposes)"""
        import shutil
        for model_id in self._model_paths:
            model_dir = os.path.join(self._cache_dir, model_id.replace('/', '_'))
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)
        self._model_paths.clear()
