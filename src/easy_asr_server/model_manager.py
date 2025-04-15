"""
Model Manager Module for easy_asr_server

This module is responsible for downloading, managing, and loading the ASR pipelines.
It provides a singleton ModelManager class to ensure thread-safe model downloading,
path management, and pipeline loading.
"""

import os
import logging
import threading
import filelock
from typing import Optional, Dict, Any, List
from modelscope import snapshot_download
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torch
import platform

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Define ModelScope IDs and configurations for different pipelines
MODEL_CONFIGS = {
    "sensevoice": {
        # 1. Components to download (Component Type -> ModelScope ID)
        "components": {
            "asr": "iic/SenseVoiceSmall", 
            # VAD/PUNC included for potential future use or implicit needs
            "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
            "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
        },
        # 2. Mapping from AutoModel constructor args to component types
        # SenseVoice is all-in-one, only needs the main model path
        "load_params_map": {
            "model": "asr" 
        },
        # 3. Parameters for the generate() call
        "generate_params": {
            "language": "auto",
            "use_itn": True,
            "batch_size_s": 60,
            "merge_vad": True,
            "merge_length_s": 15,
        },
        # 4. Postprocessing function for generate() result
        "postprocess": rich_transcription_postprocess
    },
    "paraformer": {
        # 1. Components to download
        "components": {
            "asr": "iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404",
            "vad": "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch", 
            "punc": "iic/punc_ct-transformer_cn-en-common-vocab471067-large",
        },
        # 2. Mapping from AutoModel constructor args to component types
        # Paraformer requires separate VAD and PUNC models during init
        "load_params_map": {
            "model": "asr",       
            "vad_model": "vad",   
            "punc_model": "punc" 
        },
        # 3. Parameters for the generate() call
        "generate_params": {
             "batch_size_s": 300, 
        },
        # 4. Postprocessing function for generate() result
        "postprocess": lambda x: x # No significant postprocessing for paraformer text
    }
}
DEFAULT_PIPELINE = "sensevoice"


# Default cache directory for models
DEFAULT_CACHE_DIR = os.path.expanduser("~/.cache/easy_asr_server/models")


class ModelManager:
    """
    Singleton class for managing ASR pipelines (downloading, loading, execution).
    Handles downloading via ModelScope, caching, loading AutoModel with local paths,
    and providing a unified generate interface.
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
            
            self._loaded_pipeline_instance: Optional[AutoModel] = None
            self._pipeline_type: Optional[str] = None
            self._device: Optional[str] = None

            self._initialized = True

    def _resolve_device(self, requested_device: str) -> str:
        """Resolves 'auto' to a specific device (cuda, mps, cpu)."""
        if requested_device == "auto":
            if torch.cuda.is_available():
                logger.info("Auto-detected CUDA device.")
                return "cuda"
            # Check for MPS (Apple Silicon GPU) support
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built() and platform.system() == "Darwin":
                 logger.info("Auto-detected MPS device (Apple Silicon GPU).")
                 return "mps"
            else:
                logger.info("Auto-detected CPU device.")
                return "cpu"
        # If a specific device is requested, validate it minimally
        # The underlying library (PyTorch) will do the final validation
        elif requested_device not in ["cpu", "cuda", "mps"]: # Add other valid torch devices if needed
             logger.warning(f"Requested device '{requested_device}' is not explicitly validated by ModelManager, passing through.")
             # We pass it through, PyTorch/FunASR will validate. Avoids strict validation here.
             # Example: User might request "cuda:0"
             # However, we could add more robust validation if necessary
             pass
             
        logger.info(f"Using explicitly requested device: {requested_device}")
        return requested_device

    def get_model_path(self, model_id: str) -> str:
        """
        Get the local path for a model component. Downloads if not available.
        Args:
            model_id: The ModelScope ID of the model component.
        Returns:
            str: Local path to the model component.
        """
        if model_id not in self._model_paths:
            self._model_paths[model_id] = self.download_model(model_id)
        
        # Verify the path exists after potentially downloading
        path = self._model_paths[model_id]
        if not os.path.exists(path):
             # This might happen if download failed silently or cache was cleared externally
             logger.warning(f"Model path for {model_id} not found at {path}. Attempting redownload.")
             # Clear the potentially incorrect path and retry download
             if model_id in self._model_paths:
                 del self._model_paths[model_id] 
             # Re-run download logic
             self._model_paths[model_id] = self.download_model(model_id)
             path = self._model_paths[model_id]
             if not os.path.exists(path):
                 raise FileNotFoundError(f"Failed to obtain valid model path for {model_id} at {path}")
                 
        return path


    def download_model(self, model_id: str) -> str:
        """
        Download a model component using modelscope's snapshot_download if not already available.
        Uses file locks to prevent concurrent downloads of the same component.
        Args:
            model_id: The ModelScope ID of the component to download.
        Returns:
            str: Local path to the downloaded model component directory.
        """
        # Create a model-specific lock if it doesn't exist
        if model_id not in self._download_locks:
             # Use a temporary lock for creation phase
             with threading.Lock(): 
                 if model_id not in self._download_locks: # Double check after acquiring lock
                     self._download_locks[model_id] = threading.Lock()

        # Sanitize model ID for directory naming
        sanitized_model_id = model_id.replace('/', '_')
        model_dir = os.path.join(self._cache_dir, sanitized_model_id)
        os.makedirs(model_dir, exist_ok=True)

        # Path to the lock file and completion marker
        lock_file = os.path.join(model_dir, "download.lock")
        model_complete_file = os.path.join(model_dir, "download_complete")
        # We expect snapshot_download to return the actual model directory path
        # Let's assume the presence of the completion file indicates success
        
        # Check if the marker file exists (fast path)
        if os.path.exists(model_complete_file):
            # Read the actual path from the completion file
            try:
                with open(model_complete_file, 'r') as f:
                    actual_download_path = f.read().strip()
                if os.path.exists(actual_download_path): # Check if path is still valid
                    logger.info(f"Model {model_id} already downloaded at {actual_download_path}")
                    return actual_download_path
                else:
                    logger.warning(f"Completion marker for {model_id} exists, but path {actual_download_path} not found. Will attempt redownload.")
                    os.remove(model_complete_file) # Remove invalid marker
            except Exception as e:
                 logger.warning(f"Error reading completion marker for {model_id}: {e}. Will attempt redownload.")
                 if os.path.exists(model_complete_file):
                     os.remove(model_complete_file) # Remove potentially corrupted marker

        # Try to acquire the model-specific thread lock
        with self._download_locks[model_id]:
            # Use a file lock for process-level synchronization
            # Timeout to prevent indefinite blocking if lock holder crashes
            try:
                 with filelock.FileLock(lock_file, timeout=300): # 5 minute timeout
                    # Check again in case another thread/process downloaded it while waiting
                    if os.path.exists(model_complete_file):
                         try:
                            with open(model_complete_file, 'r') as f:
                                actual_download_path = f.read().strip()
                            if os.path.exists(actual_download_path):
                                logger.info(f"Model {model_id} already downloaded at {actual_download_path} (checked after lock)")
                                return actual_download_path
                            else:
                                logger.warning(f"Completion marker for {model_id} exists (after lock), but path {actual_download_path} not found. Proceeding with download.")
                                os.remove(model_complete_file)
                         except Exception as e:
                            logger.warning(f"Error reading completion marker for {model_id} (after lock): {e}. Proceeding with download.")
                            if os.path.exists(model_complete_file):
                                os.remove(model_complete_file)
                    
                    logger.info(f"Downloading model component {model_id}...")
                    try:
                        # Download the model using modelscope, storing within the sanitized ID folder
                        # snapshot_download returns the path to the actual model files/directory
                        actual_download_path = snapshot_download(model_id, cache_dir=self._cache_dir)
                        
                        if not os.path.exists(actual_download_path):
                             raise FileNotFoundError(f"snapshot_download reported success for {model_id} but path {actual_download_path} does not exist.")

                        # Create a marker file storing the actual download path
                        with open(model_complete_file, 'w') as f:
                            f.write(actual_download_path)
                        
                        logger.info(f"Model component {model_id} downloaded successfully to {actual_download_path}")
                        return actual_download_path
                        
                    except Exception as e:
                        logger.error(f"Failed to download model component {model_id}: {str(e)}")
                        # Clean up potentially incomplete download marker if error occurred
                        if os.path.exists(model_complete_file):
                             os.remove(model_complete_file)
                        raise
            except filelock.Timeout:
                 logger.error(f"Could not acquire file lock for downloading {model_id} within timeout period.")
                 raise TimeoutError(f"Failed to acquire download lock for {model_id}")


    def load_pipeline(self, pipeline_type: str, device: str) -> None:
        """
        Loads the specified ASR pipeline onto the specified device.
        Ensures required model components are downloaded and cached.

        Args:
            pipeline_type: The key identifying the pipeline in MODEL_CONFIGS (e.g., 'sensevoice').
            device: The target device ('auto', 'cuda', 'cpu', 'mps', etc.).
        Raises:
            ValueError: If the pipeline_type is invalid.
            RuntimeError: If another pipeline is already loaded or loading fails.
            FileNotFoundError: If a required model component file cannot be found/downloaded.
        """
        if pipeline_type not in MODEL_CONFIGS:
            raise ValueError(f"Invalid pipeline_type: {pipeline_type}. Available: {list(MODEL_CONFIGS.keys())}")

        # Resolve the device string ('auto' -> 'cuda'/'mps'/'cpu') *before* checking if already loaded
        resolved_device = self._resolve_device(device)

        # Check if the *resolved* pipeline/device combo is already loaded
        with self._lock: # Ensure thread safety for checking/setting loaded state
            if self._loaded_pipeline_instance is not None:
                if self._pipeline_type == pipeline_type and self._device == resolved_device:
                    logger.info(f"Pipeline '{pipeline_type}' on device '{resolved_device}' is already loaded.")
                    return
                else:
                    # Cannot load a different pipeline/device if one is already active
                    raise RuntimeError(f"Another pipeline ('{self._pipeline_type}' on device '{self._device}') is already loaded. Cannot load '{pipeline_type}' on '{resolved_device}'.")

            # --- Proceed with loading --- 
            logger.info(f"Loading pipeline: {pipeline_type} onto resolved device: {resolved_device}")
            config = MODEL_CONFIGS[pipeline_type]
            downloaded_paths = {} # Store paths of downloaded components
            
            try:
                # 1. Ensure all required components are downloaded and get their paths
                logger.info("Ensuring model components are downloaded...")
                components_to_download = config.get("components", {})
                if not components_to_download:
                    raise ValueError(f"Pipeline '{pipeline_type}' has no components defined for download.")
                    
                for component_type, model_id in components_to_download.items():
                    logger.info(f"Checking/Downloading {component_type} model: {model_id}")
                    path = self.get_model_path(model_id) # This handles download and path retrieval
                    downloaded_paths[component_type] = path
                    logger.info(f"Using {component_type} model from: {path}")
                
                # 2. Load the AutoModel using the local paths based on the pipeline's mapping
                logger.info("Preparing AutoModel arguments...")
                load_param_mapping = config.get("load_params_map", {})
                if not load_param_mapping:
                    raise ValueError(f"Pipeline '{pipeline_type}' has no load_params_map defined.")
                    
                auto_model_kwargs = {}
                for constructor_arg, component_type in load_param_mapping.items():
                    if component_type not in downloaded_paths:
                        raise FileNotFoundError(f"Component '{component_type}' needed for AutoModel arg '{constructor_arg}' was not downloaded or its path is missing for pipeline '{pipeline_type}'.")
                    auto_model_kwargs[constructor_arg] = downloaded_paths[component_type]
                
                # Add the *resolved* device
                auto_model_kwargs['device'] = resolved_device
                
                logger.info(f"Initializing AutoModel with arguments: {list(auto_model_kwargs.keys())}")
                # Load the model within the lock to ensure thread safety during initialization
                self._loaded_pipeline_instance = AutoModel(
                    **auto_model_kwargs
                )
                self._pipeline_type = pipeline_type
                self._device = resolved_device # Store the resolved device name
                logger.info(f"Successfully loaded pipeline: {pipeline_type} using local models on device: {resolved_device}")

            except FileNotFoundError as e:
                logger.error(f"Failed to find a required model component file for pipeline {pipeline_type}: {e}")
                # Reset state even if loading fails
                self._loaded_pipeline_instance = None
                self._pipeline_type = None
                self._device = None
                raise
            except Exception as e:
                # Catch PyTorch device errors etc.
                logger.error(f"Failed to load AutoModel for pipeline {pipeline_type} on device {resolved_device}: {str(e)}")
                # Reset state even if loading fails
                self._loaded_pipeline_instance = None
                self._pipeline_type = None
                self._device = None
                raise


    def generate(self, input_audio: Any, hotword: str = "", **kwargs) -> str:
        """
        Performs ASR using the loaded pipeline. Acts as a stable interface
        to the underlying FunASR AutoModel's generation method.

        Args:
            input_audio: The audio input in a format accepted by the loaded AutoModel's 
                         generate method (e.g., file path, bytes, numpy array).
            hotword: The hotword to use in the call to the underlying pipeline's generate method.
            **kwargs: Additional keyword arguments passed directly to the 
                      AutoModel's generate method.

        Returns:
            List[Dict[str, Any]]: The recognition results from the AutoModel. 
                                  The exact structure depends on FunASR's output format.
                                  (e.g., list of segments with 'text', 'start', 'end').

        Raises:
            RuntimeError: If the pipeline has not been loaded via `load_pipeline`.
            Exception: Propagates exceptions from the AutoModel's generate method.
        """
        if not self._loaded_pipeline_instance:
            raise RuntimeError("Pipeline not loaded. Call load_pipeline first before calling generate.")
        
        # Remove audio length check based on tensor shape
        # min_samples = 500 
        # if hasattr(input_audio, 'shape') and len(input_audio.shape) > 0:
        #      audio_len = input_audio.shape[-1] 
        #      if audio_len < min_samples:
        #          logger.warning(f"Input audio length ({audio_len} samples) is too short...")
        #          return "" 
        # else: logger.warning("Cannot determine audio length for input type {type(input_audio)}...")

        # Input is now expected to be a file path (str)
        logger.debug(f"Calling generate on loaded {self._pipeline_type} pipeline with input path: {input_audio}")
        try:
            # Assuming the primary method is 'generate' and it accepts 'input'
            # Verify this against the FunASR AutoModel documentation if issues arise
            config = MODEL_CONFIGS[self._pipeline_type]
            generate_params = config.get("generate_params", {}) # Get pipeline-specific gen params
            post_process_func = config.get("postprocess")
            
            # Merge pipeline params with call-specific kwargs (kwargs take precedence)
            final_generate_args = {**generate_params, **kwargs} 
            
            logger.debug(f"Calling underlying AutoModel.generate with args: {list(final_generate_args.keys())}")
            # Pass the provided hotword parameter to the underlying generate call
            results = self._loaded_pipeline_instance.generate(input=input_audio, hotword=hotword, **final_generate_args)
            
            logger.debug("Pipeline generate call successful.")
            
            # Extract text (assuming FunASR standard output format)
            if not results or not isinstance(results, list) or not isinstance(results[0], dict) or "text" not in results[0]:
                logger.warning(f"Unexpected raw result format from pipeline {self._pipeline_type}: {results}")
                return "" # Return empty string if format is unexpected
                
            text = results[0]["text"]
            
            # Apply postprocessing if defined
            if post_process_func is not None:
                logger.debug(f"Applying postprocess function: {post_process_func.__name__}")
                text = post_process_func(text)
            else:
                logger.debug("No postprocess function defined for this pipeline.")

            return text
        except Exception as e:
             logger.error(f"Error during pipeline generate call: {str(e)}")
             raise # Re-raise the exception for the caller (e.g., ASREngine) to handle


    def clear_cache(self) -> None:
        """
        Clear the downloaded model cache directory.
        Note: This does NOT unload the currently loaded pipeline instance from memory.
        A restart is typically required to load a different pipeline after clearing cache.
        """
        import shutil
        if os.path.exists(self._cache_dir):
             logger.info(f"Clearing model cache directory: {self._cache_dir}")
             # Release file locks before attempting removal
             # This is tricky as locks might be held by other processes/threads
             # A simple approach is to just attempt removal, it might fail if locks are held
             try:
                 # Clear internal path cache first
                 self._model_paths.clear()
                 # Attempt to remove the directory
                 shutil.rmtree(self._cache_dir)
                 os.makedirs(self._cache_dir, exist_ok=True) # Recreate base dir
                 logger.info("Model cache cleared.")
             except Exception as e:
                 logger.error(f"Failed to completely clear cache directory {self._cache_dir}: {e}. Some files might be locked or in use.")
        else:
             logger.info("Model cache directory does not exist, nothing to clear.")
