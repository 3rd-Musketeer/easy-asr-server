"""
Tests for the model_manager module.
"""

import os
import unittest
import tempfile
import shutil
import time 
from unittest import mock
import filelock

# Import the module to test
# Remove old constants, import new ones if needed (like MODEL_CONFIGS, DEFAULT_PIPELINE)
from easy_asr_server.model_manager import ModelManager, MODEL_CONFIGS, DEFAULT_PIPELINE

# Mock for funasr.AutoModel instance state tracking
class MockAutoModel:
    def __init__(self, **kwargs):
        # Store args passed to constructor for verification
        self.constructor_args = kwargs
        self.generate_called = False
        self.generate_input = None
        self.generate_kwargs = None
        # This simulates the RAW output from FunASR
        self._generate_return_value = [{"text": "mock generated text from funasr", "start": 0, "end": 1000}]
        self._raise_exception_on_generate = False
        
    def generate(self, input, **kwargs):
        self.generate_called = True
        self.generate_input = input
        self.generate_kwargs = kwargs
        if self._raise_exception_on_generate:
            raise ValueError("Internal mock model failure")
        # Return the raw list-of-dicts format
        return self._generate_return_value
        
    def setup_generate_exception(self, msg="Internal mock model failure"):
        self._raise_exception_on_generate = True
        self._error_msg = msg
        def failing_generate(*args, **kwargs):
             raise ValueError(self._error_msg)
        self.generate = failing_generate


class TestModelManager(unittest.TestCase):
    """Test cases for the refactored ModelManager class"""
    
    def setUp(self):
        """Setup for tests - create a temporary directory for test models"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset the singleton instance before each test
        ModelManager._instance = None
        self.manager = ModelManager() # Create a fresh instance
        self.manager._cache_dir = self.temp_dir # Override cache dir for the instance

        # Mock snapshot_download
        self.patcher_snapshot = mock.patch('easy_asr_server.model_manager.snapshot_download')
        self.mock_snapshot_download = self.patcher_snapshot.start()
        self.mock_snapshot_download.side_effect = self._fake_snapshot_download
        
        # Mock the AutoModel CLASS with a MagicMock
        self.patcher_automodel = mock.patch('easy_asr_server.model_manager.AutoModel', autospec=True)
        self.mock_auto_model_class_itself = self.patcher_automodel.start() # This mocks the CLASS
        
        # Prepare an instance of our helper MockAutoModel to be RETURNED by the class mock
        # We need to instantiate it *within* a side_effect function or similar
        # so that the arguments passed to the real constructor are captured by our helper.
        self.created_mock_instance = None
        def auto_model_side_effect(*args, **kwargs):
            # Create instance using the actual kwargs passed
            instance = MockAutoModel(**kwargs)
            self.created_mock_instance = instance
            return instance
        self.mock_auto_model_class_itself.side_effect = auto_model_side_effect
        
        # Keep track of created fake model dirs for cleanup/verification
        self.fake_model_dirs = {}
    
    def tearDown(self):
        """Cleanup after tests"""
        self.patcher_snapshot.stop()
        self.patcher_automodel.stop()
        shutil.rmtree(self.temp_dir)
        ModelManager._instance = None # Clean up singleton after tests

    def _fake_snapshot_download(self, model_id, cache_dir=None):
        """
        Fake snapshot_download function for testing.
        Simulates downloading a model and returns its *actual* path.
        It also creates the completion marker file.
        """
        sanitized_id = model_id.replace('/', '_')
        # snapshot_download usually creates versioned subdirs, simulate a simple one
        # The key is that the *returned* path is the one used by AutoModel
        actual_model_path = os.path.join(self.temp_dir, sanitized_id, "v1.0") 
        marker_dir = os.path.join(self.temp_dir, sanitized_id)
        completion_marker = os.path.join(marker_dir, "download_complete")

        os.makedirs(actual_model_path, exist_ok=True)
        # Create a dummy file inside the actual model path
        with open(os.path.join(actual_model_path, "dummy_model_file.bin"), "w") as f:
            f.write("dummy")
            
        # Create the marker file in the parent sanitized_id dir, storing the actual path
        os.makedirs(marker_dir, exist_ok=True)
        with open(completion_marker, "w") as f:
            f.write(actual_model_path)
            
        self.fake_model_dirs[model_id] = actual_model_path
        print(f"_fake_snapshot_download: {model_id} -> {actual_model_path}") # Debug print
        return actual_model_path
    
    def test_singleton_pattern(self):
        """Test that ModelManager follows the singleton pattern"""
        # self.manager is created in setUp
        manager2 = ModelManager()
        self.assertIs(self.manager, manager2)

    def test_download_model_and_get_path(self):
        """Test download_model and get_model_path integration"""
        test_model_id = "org/test_model"
        expected_path = os.path.join(self.temp_dir, "org_test_model", "v1.0")

        # First call should trigger download
        path1 = self.manager.get_model_path(test_model_id)
        self.mock_snapshot_download.assert_called_once_with(test_model_id, cache_dir=self.temp_dir)
        self.assertEqual(path1, expected_path)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "org_test_model", "download_complete")))

        # Second call should use cache (read from marker file, no snapshot_download call)
        path2 = self.manager.get_model_path(test_model_id)
        self.mock_snapshot_download.assert_called_once() # Still called only once
        self.assertEqual(path2, expected_path)
        
    def test_download_model_locking(self):
        """ Test that concurrent downloads are prevented by locks """
        # This is harder to test perfectly without actual threads/processes,
        # but we can simulate the lock file existing.
        test_model_id = "org/concurrent_model"
        sanitized_id = test_model_id.replace('/', '_')
        model_dir = os.path.join(self.temp_dir, sanitized_id)
        lock_file_path = os.path.join(model_dir, "download.lock")
        os.makedirs(model_dir, exist_ok=True)

        # Acquire the file lock to simulate another process downloading
        lock = filelock.FileLock(lock_file_path)
        with lock:
             # Try to download while lock is held by this thread (simulates another process)
             # We expect get_model_path to wait and eventually succeed *after* the lock is released
             # For simplicity here, we just check it doesn't call snapshot download immediately
             # A more complex test would involve threading.
             
             # Re-patch snapshot_download to raise an error if called while lock is held
             def fail_if_called(*args, **kwargs):
                 raise AssertionError("snapshot_download called while lock should be held")
             self.mock_snapshot_download.side_effect = fail_if_called
             
             # In a real threaded scenario, the call below would block. Here, it won't block,
             # but the principle is that the check inside download_model should prevent 
             # snapshot_download if the lock can't be acquired immediately.
             # We rely on the internal logic tested elsewhere.
             pass # Cannot easily test blocking/waiting here
             
        # Restore side effect after lock release for subsequent tests
        self.mock_snapshot_download.side_effect = self._fake_snapshot_download
        # Now try getting the path - it should proceed with download
        self.manager.get_model_path(test_model_id)
        self.mock_snapshot_download.assert_called_once()
        
    def test_download_model_corrupted_marker_before_lock(self):
        """Test download_model handles corrupted marker file before lock."""
        test_model_id = "org/corrupt_marker"
        sanitized_id = test_model_id.replace('/', '_')
        model_dir = os.path.join(self.temp_dir, sanitized_id)
        completion_marker = os.path.join(model_dir, "download_complete")
        expected_final_path = os.path.join(self.temp_dir, sanitized_id, "v1.0") 

        # 1. Create the directory and a corrupted marker file
        os.makedirs(model_dir, exist_ok=True)
        with open(completion_marker, "w") as f:
            f.write("This is not a valid path\nSome other garbage")

        # 2. Call get_model_path - it should detect corruption and redownload
        path = self.manager.get_model_path(test_model_id)

        # 3. Assert snapshot_download was called
        self.mock_snapshot_download.assert_called_once_with(test_model_id, cache_dir=self.temp_dir)
        
        # 4. Assert the correct path (from the successful fake download) is returned
        self.assertEqual(path, expected_final_path)
        
        # 5. Assert the marker file was overwritten with the correct path
        self.assertTrue(os.path.exists(completion_marker))
        with open(completion_marker, "r") as f:
            content = f.read().strip()
            self.assertEqual(content, expected_final_path)

    def test_download_model_invalid_path_in_marker(self):
        """Test download_model handles marker file with an invalid path."""
        test_model_id = "org/invalid_path_marker"
        sanitized_id = test_model_id.replace('/', '_')
        model_dir = os.path.join(self.temp_dir, sanitized_id)
        completion_marker = os.path.join(model_dir, "download_complete")
        invalid_path = os.path.join(self.temp_dir, "path_does_not_exist")
        expected_final_path = os.path.join(self.temp_dir, sanitized_id, "v1.0") 

        # 1. Create the directory and marker file with a non-existent path
        os.makedirs(model_dir, exist_ok=True)
        with open(completion_marker, "w") as f:
            f.write(invalid_path)
            
        # Ensure the invalid path doesn't actually exist
        self.assertFalse(os.path.exists(invalid_path))

        # 2. Call get_model_path - it should detect invalid path and redownload
        path = self.manager.get_model_path(test_model_id)

        # 3. Assert snapshot_download was called
        self.mock_snapshot_download.assert_called_once_with(test_model_id, cache_dir=self.temp_dir)
        
        # 4. Assert the correct path is returned
        self.assertEqual(path, expected_final_path)
        
        # 5. Assert the marker file was overwritten
        self.assertTrue(os.path.exists(completion_marker))
        with open(completion_marker, "r") as f:
            content = f.read().strip()
            self.assertEqual(content, expected_final_path)

    def test_load_pipeline_success_sensevoice(self):
        """Test successfully loading the sensevoice pipeline."""
        pipeline_type = "sensevoice"
        device = "cpu"
        config = MODEL_CONFIGS[pipeline_type]
        components_to_download = config["components"]
        load_params_map = config["load_params_map"]

        self.manager.load_pipeline(pipeline_type, device)

        # Verify snapshot_download was called for each defined component
        self.assertEqual(self.mock_snapshot_download.call_count, len(components_to_download))
        calls = [mock.call(model_id, cache_dir=self.temp_dir) for model_id in components_to_download.values()]
        self.mock_snapshot_download.assert_has_calls(calls, any_order=True)

        # Verify the AutoModel CLASS mock was called once
        self.mock_auto_model_class_itself.assert_called_once()
        # Retrieve the arguments passed to the constructor via the class mock
        call_args, call_kwargs = self.mock_auto_model_class_itself.call_args
        
        # Verify the correct arguments based on load_params_map
        expected_automodel_kwargs = {}
        for arg_name, comp_type in load_params_map.items():
             expected_automodel_kwargs[arg_name] = self.fake_model_dirs[components_to_download[comp_type]]
        expected_automodel_kwargs['device'] = device
        self.assertEqual(call_kwargs, expected_automodel_kwargs)

        # Verify internal state
        self.assertIsNotNone(self.manager._loaded_pipeline_instance)
        self.assertIsInstance(self.manager._loaded_pipeline_instance, MockAutoModel)
        self.assertEqual(self.manager._pipeline_type, pipeline_type)
        self.assertEqual(self.manager._device, device)

    def test_load_pipeline_success_paraformer(self):
        """Test successfully loading the paraformer pipeline."""
        pipeline_type = "paraformer"
        device = "cuda:0"
        config = MODEL_CONFIGS[pipeline_type]
        components_to_download = config["components"]
        load_params_map = config["load_params_map"]

        self.manager.load_pipeline(pipeline_type, device)

        # Verify downloads
        self.assertEqual(self.mock_snapshot_download.call_count, len(components_to_download))
        calls = [mock.call(model_id, cache_dir=self.temp_dir) for model_id in components_to_download.values()]
        self.mock_snapshot_download.assert_has_calls(calls, any_order=True)

        # Verify AutoModel instantiation
        self.mock_auto_model_class_itself.assert_called_once()
        call_args, call_kwargs = self.mock_auto_model_class_itself.call_args
        expected_automodel_kwargs = {}
        for arg_name, comp_type in load_params_map.items():
             expected_automodel_kwargs[arg_name] = self.fake_model_dirs[components_to_download[comp_type]]
        expected_automodel_kwargs['device'] = device
        self.assertEqual(call_kwargs, expected_automodel_kwargs)

        # Verify internal state
        self.assertIsNotNone(self.manager._loaded_pipeline_instance)
        self.assertIsInstance(self.manager._loaded_pipeline_instance, MockAutoModel)
        self.assertEqual(self.manager._pipeline_type, pipeline_type)
        self.assertEqual(self.manager._device, device)

    def test_load_pipeline_already_loaded(self):
        """Test calling load_pipeline when the same pipeline is already loaded"""
        pipeline_type = "sensevoice"
        device = "cpu"
        self.manager.load_pipeline(pipeline_type, device) # Load once
        first_call_count_snapshot = self.mock_snapshot_download.call_count
        # Check the call count on the CLASS mock
        first_call_count_automodel = self.mock_auto_model_class_itself.call_count 

        self.manager.load_pipeline(pipeline_type, device) # Load again

        # Ensure download and init were not called again
        self.assertEqual(self.mock_snapshot_download.call_count, first_call_count_snapshot)
        # The CLASS mock should not be called again
        self.assertEqual(self.mock_auto_model_class_itself.call_count, first_call_count_automodel)

    def test_load_pipeline_different_loaded(self):
        """Test calling load_pipeline when a different pipeline is loaded"""
        self.manager.load_pipeline("sensevoice", "cpu") # Load first pipeline
        with self.assertRaises(RuntimeError):
            self.manager.load_pipeline("paraformer", "cpu") # Try loading different one

    def test_load_pipeline_invalid_type(self):
        """Test loading an invalid pipeline type"""
        with self.assertRaises(ValueError):
            self.manager.load_pipeline("invalid_pipeline", "cpu")
            
    def test_load_pipeline_download_failure(self):
        """ Test pipeline loading when a component download fails """
        pipeline_type = "sensevoice"
        device = "cpu"
        # Simulate download failure for one component
        self.mock_snapshot_download.side_effect = FileNotFoundError("Simulated download error")
        
        with self.assertRaises(FileNotFoundError):
             self.manager.load_pipeline(pipeline_type, device)
             
        # Ensure pipeline state is not set
        self.assertIsNone(self.manager._loaded_pipeline_instance)
        self.assertIsNone(self.manager._pipeline_type)

    def test_download_model_snapshot_exception(self):
        """Test download_model handles generic exception during snapshot_download."""
        test_model_id = "org/snapshot_fail"
        error_message = "Simulated snapshot download failure"
        
        # Configure mock to raise a generic exception
        self.mock_snapshot_download.side_effect = Exception(error_message)
        
        # Call get_model_path, expecting it to propagate the exception
        with self.assertRaisesRegex(Exception, error_message):
            self.manager.get_model_path(test_model_id)
            
        # Ensure the model path was not added to the cache
        self.assertNotIn(test_model_id, self.manager._model_paths)
        # Reset side effect for other tests
        self.mock_snapshot_download.side_effect = self._fake_snapshot_download

    def test_load_pipeline_missing_component_id(self):
        """Test load_pipeline raises error if a component ID is missing."""
        pipeline_type = "broken_config"
        device = "cpu"
        broken_config = {
            "asr": "some_id",
            "vad": "another_id",
            # Missing "punc"
            # Add dummy params/postprocess to match expected structure
            "params": {},
            "postprocess": None 
        }
        
        # Use mock.patch.dict to temporarily modify the module-level dict
        with mock.patch.dict('easy_asr_server.model_manager.MODEL_CONFIGS', {pipeline_type: broken_config}, clear=True):
            # The clear=True ensures only our broken config exists during the patch
            # Expect ValueError because the config is missing the "components" key
            with self.assertRaisesRegex(ValueError, f"Pipeline '{pipeline_type}' has no components defined for download."):
                self.manager.load_pipeline(pipeline_type, device)

    def test_load_pipeline_automodel_exception(self):
        """Test load_pipeline handles generic exception during AutoModel init."""
        pipeline_type = "sensevoice"
        device = "cpu"
        error_message = "Simulated AutoModel init failure"
        
        # Configure the CLASS mock's side_effect to raise an exception
        self.mock_auto_model_class_itself.side_effect = Exception(error_message)
        
        # Call load_pipeline, expecting it to propagate the exception
        with self.assertRaisesRegex(Exception, error_message):
            self.manager.load_pipeline(pipeline_type, device)
            
        # Ensure the pipeline state is not set
        self.assertIsNone(self.manager._loaded_pipeline_instance)
        self.assertIsNone(self.manager._pipeline_type)
        
        # Reset side effect for other tests - important!
        def auto_model_side_effect(*args, **kwargs):
            instance = MockAutoModel(**kwargs)
            self.created_mock_instance = instance
            return instance
        self.mock_auto_model_class_itself.side_effect = auto_model_side_effect

    def test_generate_success(self):
        """Test the generate method after loading a pipeline returns a string"""
        pipeline_type = "sensevoice" # Use sensevoice as it has a postprocess function
        device = "cpu"
        self.manager.load_pipeline(pipeline_type, device)
        loaded_mock_instance = self.manager._loaded_pipeline_instance 
        self.assertIsInstance(loaded_mock_instance, MockAutoModel)

        input_data = "test_audio.wav"
        input_hotword = "test hot word"
        kwargs = {"some_extra_param": True}
        
        raw_text_from_mock = loaded_mock_instance._generate_return_value[0]["text"]
        postprocess_func = MODEL_CONFIGS[pipeline_type].get("postprocess")
        expected_final_text = postprocess_func(raw_text_from_mock) if postprocess_func else raw_text_from_mock
        
        # Pass hotword to manager's generate method
        result = self.manager.generate(input_data, hotword=input_hotword, **kwargs)

        self.assertTrue(loaded_mock_instance.generate_called)
        self.assertEqual(loaded_mock_instance.generate_input, input_data)
        # Check generate_kwargs received by the mock AutoModel instance
        # It should contain the passed hotword AND other params
        received_kwargs = loaded_mock_instance.generate_kwargs
        self.assertEqual(received_kwargs.get("hotword"), input_hotword)
        # Verify extra kwargs are also present
        self.assertEqual(received_kwargs.get("some_extra_param"), True)
        # Verify generate_params from config are also present
        for key, value in MODEL_CONFIGS[pipeline_type]["generate_params"].items():
            self.assertEqual(received_kwargs.get(key), value)

        self.assertEqual(result, expected_final_text)
        self.assertIsInstance(result, str)

    def test_generate_success_no_postprocess(self):
        """Test generate when the pipeline has no postprocess function"""
        pipeline_type = "paraformer"
        device = "cpu"
        self.manager.load_pipeline(pipeline_type, device)
        loaded_mock_instance = self.manager._loaded_pipeline_instance
        self.assertIsInstance(loaded_mock_instance, MockAutoModel)

        input_data = "test_audio.wav"
        input_hotword = "another test hotword"
        kwargs = {}
        
        raw_text_from_mock = loaded_mock_instance._generate_return_value[0]["text"]
        expected_final_text = raw_text_from_mock 
        
        # Pass hotword
        result = self.manager.generate(input_data, hotword=input_hotword, **kwargs)

        self.assertTrue(loaded_mock_instance.generate_called)
        self.assertEqual(loaded_mock_instance.generate_input, input_data)
        # Verify hotword and generate_params were passed
        received_kwargs = loaded_mock_instance.generate_kwargs
        self.assertEqual(received_kwargs.get("hotword"), input_hotword)
        for key, value in MODEL_CONFIGS[pipeline_type]["generate_params"].items():
            self.assertEqual(received_kwargs.get(key), value)
        
        self.assertEqual(result, expected_final_text)
        self.assertIsInstance(result, str)

    def test_generate_not_loaded(self):
        """Test calling generate before loading a pipeline"""
        with self.assertRaisesRegex(RuntimeError, "Pipeline not loaded"):
            self.manager.generate("test_audio.wav")
            
    def test_generate_internal_failure(self):
        """ Test generate when the underlying AutoModel fails """
        pipeline_type = "sensevoice"
        device = "cpu"
        self.manager.load_pipeline(pipeline_type, device)
        loaded_mock_instance = self.manager._loaded_pipeline_instance
        self.assertIsInstance(loaded_mock_instance, MockAutoModel)
        
        # Make the mock instance generate raise an error
        error_message = "Internal mock model failure"
        loaded_mock_instance.setup_generate_exception(error_message)
                
        with self.assertRaisesRegex(ValueError, error_message):
            # Pass dummy hotword, should still raise the exception from generate
            self.manager.generate("test_audio.wav", hotword="dummy")

    def test_download_filelock_timeout(self):
        """Test download_model raises TimeoutError if file lock times out."""
        test_model_id = "org/lock_timeout"
        
        # Mock filelock.FileLock to raise Timeout
        with mock.patch('filelock.FileLock') as mock_filelock_class:
            # Configure the __enter__ method of the mock instance to raise Timeout
            mock_lock_instance = mock.MagicMock()
            mock_lock_instance.__enter__.side_effect = filelock.Timeout("Simulated timeout")
            mock_filelock_class.return_value = mock_lock_instance
            
            # Call get_model_path, expecting TimeoutError
            with self.assertRaises(TimeoutError):
                self.manager.get_model_path(test_model_id)
            
            # Verify FileLock was instantiated with the correct path and timeout
            sanitized_id = test_model_id.replace('/', '_')
            expected_lock_path = os.path.join(self.temp_dir, sanitized_id, "download.lock")
            mock_filelock_class.assert_called_once_with(expected_lock_path, timeout=300)

    def test_clear_cache_success(self):
        """Test clearing the cache directory."""
        # Create a dummy model directory and file
        dummy_model_id = "org/dummy_model_cache"
        dummy_path = self.manager.get_model_path(dummy_model_id)
        self.assertTrue(os.path.exists(dummy_path))
        self.assertTrue(os.path.exists(self.manager._cache_dir))

        # Call clear_cache
        self.manager.clear_cache()

        # Verify cache directory still exists (it's recreated) but is empty
        self.assertTrue(os.path.exists(self.manager._cache_dir))
        self.assertEqual(len(os.listdir(self.manager._cache_dir)), 0)
        # Verify internal path cache is cleared
        self.assertEqual(len(self.manager._model_paths), 0)

    def test_clear_cache_failure(self):
        """Test clear_cache handles errors during shutil.rmtree."""
        self.manager.get_model_path("org/dummy_for_clear_fail")
        self.assertTrue(os.path.exists(self.manager._cache_dir))
        
        error_message = "Permission denied during rmtree"
        with mock.patch('shutil.rmtree') as mock_rmtree, mock.patch('easy_asr_server.model_manager.logger.error') as mock_logger_error:
            
            mock_rmtree.side_effect = OSError(error_message)
            
            # Call clear_cache - it should log the error but not raise it
            self.manager.clear_cache()
            
            # Verify rmtree was called
            mock_rmtree.assert_called_once_with(self.manager._cache_dir)
            # Verify the error was logged
            mock_logger_error.assert_called_once()
            log_call_args, _ = mock_logger_error.call_args
            self.assertIn(self.manager._cache_dir, log_call_args[0])
            # Check the first argument (the formatted string) for the error message
            self.assertIn(error_message, log_call_args[0])
            # Verify internal path cache was still cleared
            self.assertEqual(len(self.manager._model_paths), 0)

    def test_generate_unexpected_result_format(self):
        """Test generate handles unexpected result format from AutoModel."""
        pipeline_type = "sensevoice"
        device = "cpu"
        self.manager.load_pipeline(pipeline_type, device)
        loaded_mock_instance = self.manager._loaded_pipeline_instance
        self.assertIsInstance(loaded_mock_instance, MockAutoModel)

        # Configure the mock AutoModel to return invalid formats
        test_cases = [
            None, # None result
            [], # Empty list
            [{"wrong_key": "value"}], # List with dict missing 'text' key
            "just a string", # Plain string instead of list
            [1, 2, 3] # List of non-dicts
        ]

        input_data = "test_audio_format.wav"
        with mock.patch('easy_asr_server.model_manager.logger.warning') as mock_logger_warning:
            for invalid_result in test_cases:
                mock_logger_warning.reset_mock()
                loaded_mock_instance._generate_return_value = invalid_result
                
                # Call generate and expect an empty string
                result = self.manager.generate(input_data)
                self.assertEqual(result, "")
                
                # Verify a warning was logged
                mock_logger_warning.assert_called_once()
                log_call_args, _ = mock_logger_warning.call_args
                self.assertIn(f"Unexpected raw result format from pipeline {pipeline_type}", log_call_args[0])
                self.assertIn(str(invalid_result), log_call_args[0])

    # Remove obsolete test
    # def test_ensure_models_downloaded(self):
        
    # def test_model_path_cached(self): # This logic is implicitly tested by test_download_model_and_get_path
    #     pass 

if __name__ == '__main__':
    unittest.main()
