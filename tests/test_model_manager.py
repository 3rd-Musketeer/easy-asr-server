"""
Tests for the model_manager module.
"""

import os
import unittest
import tempfile
import shutil
from unittest import mock

# Import the module to test
from easy_asr_server.model_manager import ModelManager, DEFAULT_ASR_MODEL_ID, DEFAULT_VAD_MODEL_ID


class TestModelManager(unittest.TestCase):
    """Test cases for the ModelManager class"""
    
    def setUp(self):
        """Setup for tests - create a temporary directory for test models"""
        # Create a temporary cache directory
        self.temp_dir = tempfile.mkdtemp()
        
        # Save the original cache directory and patch it for testing
        self.original_cache_dir = ModelManager._instance._cache_dir if ModelManager._instance else None
        
        # Reset the singleton for clean testing 
        ModelManager._instance = None
        
        # Mock the download method to avoid actual downloads during tests
        self.patcher = mock.patch('easy_asr_server.model_manager.snapshot_download')
        self.mock_snapshot_download = self.patcher.start()
        
        # Have the mock return a path within our temp directory
        self.mock_snapshot_download.side_effect = self._fake_download
    
    def tearDown(self):
        """Cleanup after tests"""
        # Stop the patcher
        self.patcher.stop()
        
        # Cleanup the temporary directory
        shutil.rmtree(self.temp_dir)
        
        # Reset the singleton for clean testing
        ModelManager._instance = None
        
        # Restore the original cache directory if it existed
        if self.original_cache_dir:
            if ModelManager._instance:
                ModelManager._instance._cache_dir = self.original_cache_dir
    
    def _fake_download(self, model_id, cache_dir=None):
        """
        Fake download function for testing.
        Creates a directory structure similar to what modelscope would create.
        """
        # Create a directory for the model within our temporary directory
        model_dir = os.path.join(self.temp_dir, model_id.replace('/', '_'))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # Create a dummy file to simulate model files
        with open(os.path.join(model_dir, "model.bin"), "w") as f:
            f.write("Dummy model content")
        
        return model_dir
    
    def test_singleton_pattern(self):
        """Test that ModelManager follows the singleton pattern"""
        manager1 = ModelManager()
        manager2 = ModelManager()
        
        # Both instances should be the same object
        self.assertIs(manager1, manager2)
    
    def test_get_model_path(self):
        """Test that get_model_path returns the expected path"""
        manager = ModelManager()
        
        # Override cache directory for testing
        manager._cache_dir = self.temp_dir
        
        # Get the path for the ASR model
        path = manager.get_model_path(DEFAULT_ASR_MODEL_ID)
        
        # Check that the method called our mock
        self.mock_snapshot_download.assert_called_once()
        
        # The path should be within our temp directory
        self.assertIn(self.temp_dir, path)
        
        # Path should include the model ID in a sanitized form
        self.assertIn(DEFAULT_ASR_MODEL_ID.replace('/', '_'), path)
    
    def test_download_model(self):
        """Test the download_model method"""
        manager = ModelManager()
        
        # Override cache directory for testing
        manager._cache_dir = self.temp_dir
        
        # Download the model
        path = manager.download_model(DEFAULT_ASR_MODEL_ID)
        
        # The complete file should now exist
        model_dir = os.path.join(self.temp_dir, DEFAULT_ASR_MODEL_ID.replace('/', '_'))
        complete_file = os.path.join(model_dir, "download_complete")
        self.assertTrue(os.path.exists(complete_file))
    
    def test_ensure_models_downloaded(self):
        """Test that ensure_models_downloaded downloads both models"""
        manager = ModelManager()
        
        # Override cache directory for testing
        manager._cache_dir = self.temp_dir
        
        # Ensure both models are downloaded
        paths = manager.ensure_models_downloaded()
        
        # Should have called our mock twice (once for each model)
        self.assertEqual(self.mock_snapshot_download.call_count, 2)
        
        # Should have returned paths for both models
        self.assertIn('asr', paths)
        self.assertIn('vad', paths)
        
        # Both paths should exist in our temp directory
        self.assertIn(self.temp_dir, paths['asr'])
        self.assertIn(self.temp_dir, paths['vad'])
    
    def test_model_path_cached(self):
        """Test that model paths are cached and not downloaded multiple times"""
        manager = ModelManager()
        
        # Override cache directory for testing
        manager._cache_dir = self.temp_dir
        
        # Call get_model_path twice for the same model
        path1 = manager.get_model_path(DEFAULT_ASR_MODEL_ID)
        path2 = manager.get_model_path(DEFAULT_ASR_MODEL_ID)
        
        # Should have called our mock only once
        self.assertEqual(self.mock_snapshot_download.call_count, 1)
        
        # Both paths should be the same
        self.assertEqual(path1, path2)


if __name__ == '__main__':
    unittest.main()
