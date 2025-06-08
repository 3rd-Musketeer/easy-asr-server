"""
Tests for the CLI interface.
"""

import os
import unittest
from unittest import mock
from typer.testing import CliRunner

from easy_asr_server.api import app_cli


class TestCLI(unittest.TestCase):
    """Test cases specifically for the Typer CLI interface."""

    def setUp(self):
        self.runner = CliRunner()

    def test_cli_hotword_file_option(self):
        """Test the 'run' command sets the environment variable for --hotword-file."""
        dummy_path = "/path/to/my/hotwords.txt"
        env_var = "EASY_ASR_HOTWORD_FILE"
        
        # Mock the function that actually starts the server to prevent it running
        # Also mock setup_logging as it's called in the command
        # Use patch.dict to manage the environment variable for the test
        with mock.patch('easy_asr_server.cli._start_uvicorn') as mock_start, \
             mock.patch('easy_asr_server.utils.setup_logging') as mock_log_setup, \
             mock.patch.dict(os.environ, {}, clear=True) as mock_env: # Start with clean env for test
            
            result = self.runner.invoke(
                app_cli, 
                ["run", "--hotword-file", dummy_path]
            )
            
            # Verify _start_uvicorn was called (meaning command logic ran up to that point)
            mock_start.assert_called_once() 
            # Verify the environment variable was set correctly
            self.assertEqual(os.environ.get(env_var), dummy_path)
            # Also check device/pipeline env vars are set if needed in other tests
            self.assertIn("EASY_ASR_DEVICE", os.environ)
            self.assertIn("EASY_ASR_PIPELINE", os.environ)
            self.assertIn("EASY_ASR_LOG_LEVEL", os.environ)

    def test_cli_no_hotword_file_option(self):
        """Test the 'run' command unsets the environment variable if --hotword-file is not used."""
        env_var = "EASY_ASR_HOTWORD_FILE"
        
        # Mock necessary functions and ensure the env var exists initially
        with mock.patch('easy_asr_server.cli._start_uvicorn') as mock_start, \
             mock.patch('easy_asr_server.utils.setup_logging'), \
             mock.patch.dict(os.environ, {env_var: "some_initial_value"}, clear=True) as mock_env:

            # Check it exists before invoke
            self.assertEqual(os.environ.get(env_var), "some_initial_value")
            
            result = self.runner.invoke(
                app_cli, 
                ["run"] # Add explicit "run" command
            )

            mock_start.assert_called_once()
            # Verify the environment variable was removed by the command logic
            self.assertNotIn(env_var, os.environ)
            self.assertIsNone(os.environ.get(env_var))
            # Check default env vars are still set
            self.assertIn("EASY_ASR_DEVICE", os.environ)
            self.assertIn("EASY_ASR_PIPELINE", os.environ)
            self.assertIn("EASY_ASR_LOG_LEVEL", os.environ)

    def test_cli_invalid_pipeline(self):
        """Test the 'run' command exits with an error for an invalid pipeline."""
        with mock.patch('easy_asr_server.cli._start_uvicorn') as mock_start, \
             mock.patch('easy_asr_server.utils.setup_logging'): # Mock logging too
            
            result = self.runner.invoke(
                app_cli, 
                ["run", "--pipeline", "invalid_pipeline_name"]
            )
            
            # Expect a non-zero exit code because pipeline validation fails
            self.assertNotEqual(result.exit_code, 0)
            # Check if the error message about invalid pipeline is in the output
            self.assertIn("Invalid pipeline type", result.stdout)
            # Ensure the server start function was NOT called
            mock_start.assert_not_called()

    def test_cli_comprehensive_options(self):
        """Test CLI with comprehensive options combinations"""
        with mock.patch('easy_asr_server.cli._start_uvicorn') as mock_start, \
             mock.patch('easy_asr_server.utils.setup_logging'), \
             mock.patch.dict(os.environ, {}, clear=True):
            
            result = self.runner.invoke(app_cli, [
                "run",
                "--device", "cpu",
                "--pipeline", "sensevoice", 
                "--hotword-file", "/test/hotwords.txt",
                "--log-level", "DEBUG"
            ])
            
            # Verify all environment variables were set
            self.assertEqual(os.environ.get("EASY_ASR_DEVICE"), "cpu")
            self.assertEqual(os.environ.get("EASY_ASR_PIPELINE"), "sensevoice")
            self.assertEqual(os.environ.get("EASY_ASR_HOTWORD_FILE"), "/test/hotwords.txt")
            self.assertEqual(os.environ.get("EASY_ASR_LOG_LEVEL"), "DEBUG")
            mock_start.assert_called_once()

    def test_download_command_sensevoice(self):
        """Test the 'download' command for sensevoice pipeline."""
        with mock.patch('easy_asr_server.cli.ModelManager') as mock_model_manager_class, \
             mock.patch('easy_asr_server.utils.setup_logging'):
            
            # Mock the ModelManager instance and its methods
            mock_manager = mock.Mock()
            mock_model_manager_class.return_value = mock_manager
            mock_manager.get_model_path.return_value = "/fake/path/to/model"
            
            result = self.runner.invoke(app_cli, ["download", "sensevoice"])
            
            # Verify successful execution
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Model download completed successfully!", result.stdout)
            
            # Verify ModelManager was initialized
            mock_model_manager_class.assert_called_once()
            
            # Verify get_model_path was called for sensevoice components
            expected_calls = [
                mock.call("iic/SenseVoiceSmall"),
                mock.call("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                mock.call("iic/punc_ct-transformer_cn-en-common-vocab471067-large")
            ]
            mock_manager.get_model_path.assert_has_calls(expected_calls, any_order=True)

    def test_download_command_paraformer(self):
        """Test the 'download' command for paraformer pipeline."""
        with mock.patch('easy_asr_server.cli.ModelManager') as mock_model_manager_class, \
             mock.patch('easy_asr_server.utils.setup_logging'):
            
            mock_manager = mock.Mock()
            mock_model_manager_class.return_value = mock_manager
            mock_manager.get_model_path.return_value = "/fake/path/to/model"
            
            result = self.runner.invoke(app_cli, ["download", "paraformer"])
            
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Model download completed successfully!", result.stdout)
            
            # Verify get_model_path was called for paraformer components
            expected_calls = [
                mock.call("iic/speech_paraformer-large-contextual_asr_nat-zh-cn-16k-common-vocab8404"),
                mock.call("iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"),
                mock.call("iic/punc_ct-transformer_cn-en-common-vocab471067-large")
            ]
            mock_manager.get_model_path.assert_has_calls(expected_calls, any_order=True)

    def test_download_command_all(self):
        """Test the 'download' command for all pipelines."""
        with mock.patch('easy_asr_server.cli.ModelManager') as mock_model_manager_class, \
             mock.patch('easy_asr_server.utils.setup_logging'):
            
            mock_manager = mock.Mock()
            mock_model_manager_class.return_value = mock_manager
            mock_manager.get_model_path.return_value = "/fake/path/to/model"
            
            result = self.runner.invoke(app_cli, ["download", "all"])
            
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Model download completed successfully!", result.stdout)
            
            # Verify get_model_path was called for all pipeline components
            # Should include components from both sensevoice and paraformer
            call_count = mock_manager.get_model_path.call_count
            # sensevoice: 3 components, paraformer: 3 components = 6 total
            # But some components are shared (vad, punc), so actual calls depend on implementation
            self.assertGreaterEqual(call_count, 3)  # At least unique components
            
    def test_download_command_invalid_pipeline(self):
        """Test the 'download' command with invalid pipeline name."""
        with mock.patch('easy_asr_server.utils.setup_logging'):
            
            result = self.runner.invoke(app_cli, ["download", "invalid_pipeline"])
            
            # Expect non-zero exit code
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Invalid pipeline", result.stdout)

    def test_download_command_with_log_level(self):
        """Test the 'download' command with custom log level."""
        with mock.patch('easy_asr_server.cli.ModelManager') as mock_model_manager_class, \
             mock.patch('easy_asr_server.cli.setup_logging') as mock_setup_logging:
            
            mock_manager = mock.Mock()
            mock_model_manager_class.return_value = mock_manager
            mock_manager.get_model_path.return_value = "/fake/path/to/model"
            
            result = self.runner.invoke(app_cli, ["download", "sensevoice", "--log-level", "DEBUG"])
            
            self.assertEqual(result.exit_code, 0)
            # Verify setup_logging was called with DEBUG level
            mock_setup_logging.assert_called_with(level="DEBUG")

    def test_download_command_model_download_failure(self):
        """Test the 'download' command when model download fails."""
        with mock.patch('easy_asr_server.cli.ModelManager') as mock_model_manager_class, \
             mock.patch('easy_asr_server.utils.setup_logging'):
            
            mock_manager = mock.Mock()
            mock_model_manager_class.return_value = mock_manager
            # Simulate download failure for one component
            mock_manager.get_model_path.side_effect = [
                "/fake/path/to/model",  # First call succeeds
                Exception("Download failed"),  # Second call fails
                "/fake/path/to/model"   # Third call succeeds
            ]
            
            result = self.runner.invoke(app_cli, ["download", "sensevoice"])
            
            # Should still complete successfully (graceful error handling)
            self.assertEqual(result.exit_code, 0)
            self.assertIn("Model download completed successfully!", result.stdout)


if __name__ == '__main__':
    unittest.main() 