"""
Tests for device resolution logic.
"""

import unittest
from unittest import mock

from easy_asr_server.utils import resolve_device_string


class TestDeviceResolution(unittest.TestCase):
    """Tests for the device string resolution and validation logic."""

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('torch.backends.mps.is_available', return_value=False) # Ensure MPS is mocked too
    def test_resolve_auto_cuda_available(self, mock_mps_available, mock_cuda_available):
        """Test 'auto' resolves to 'cuda' when CUDA is available."""
        self.assertEqual(resolve_device_string("auto"), "cuda")
        mock_cuda_available.assert_called_once()
        mock_mps_available.assert_not_called() # Should short-circuit

    @mock.patch('torch.cuda.is_available', return_value=False)
    @mock.patch('torch.backends.mps.is_available', return_value=True)
    def test_resolve_auto_mps_available(self, mock_mps_available, mock_cuda_available):
        """Test 'auto' resolves to 'mps' when CUDA is unavailable but MPS is."""
        self.assertEqual(resolve_device_string("auto"), "mps")
        mock_cuda_available.assert_called_once()
        mock_mps_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=False)
    @mock.patch('torch.backends.mps.is_available', return_value=False)
    def test_resolve_auto_cpu_fallback(self, mock_mps_available, mock_cuda_available):
        """Test 'auto' resolves to 'cpu' when neither CUDA nor MPS is available."""
        self.assertEqual(resolve_device_string("auto"), "cpu")
        mock_cuda_available.assert_called_once()
        mock_mps_available.assert_called_once()

    def test_resolve_cpu(self):
        """Test 'cpu' is returned directly."""
        self.assertEqual(resolve_device_string("cpu"), "cpu")

    @mock.patch('torch.backends.mps.is_available', return_value=True)
    def test_resolve_mps_available(self, mock_mps_available):
        """Test 'mps' is returned when MPS is available."""
        self.assertEqual(resolve_device_string("mps"), "mps")
        mock_mps_available.assert_called_once()

    @mock.patch('torch.backends.mps.is_available', return_value=False)
    def test_resolve_mps_unavailable(self, mock_mps_available):
        """Test 'mps' raises ValueError when MPS is unavailable."""
        with self.assertRaisesRegex(ValueError, "MPS device requested but not available"):
            resolve_device_string("mps")
        mock_mps_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=True)
    def test_resolve_cuda_available(self, mock_cuda_available):
        """Test 'cuda' is returned when CUDA is available."""
        self.assertEqual(resolve_device_string("cuda"), "cuda")
        mock_cuda_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_resolve_cuda_unavailable(self, mock_cuda_available):
        """Test 'cuda' raises ValueError when CUDA is unavailable."""
        with self.assertRaisesRegex(ValueError, "CUDA device requested but not available"):
            resolve_device_string("cuda")
        mock_cuda_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('torch.cuda.device_count', return_value=2)
    def test_resolve_cuda_index_available(self, mock_device_count, mock_cuda_available):
        """Test 'cuda:N' is returned when CUDA and the specific index are available."""
        self.assertEqual(resolve_device_string("cuda:0"), "cuda:0")
        self.assertEqual(resolve_device_string("cuda:1"), "cuda:1")
        mock_cuda_available.assert_called() # Called for each resolve
        mock_device_count.assert_called() # Called for each resolve with index

    @mock.patch('torch.cuda.is_available', return_value=False)
    def test_resolve_cuda_index_cuda_unavailable(self, mock_cuda_available):
        """Test 'cuda:N' raises ValueError when CUDA itself is unavailable."""
        with self.assertRaisesRegex(ValueError, "CUDA device requested but not available"):
            resolve_device_string("cuda:0")
        mock_cuda_available.assert_called_once()

    @mock.patch('torch.cuda.is_available', return_value=True)
    @mock.patch('torch.cuda.device_count', return_value=1) # Only device 0 exists
    def test_resolve_cuda_index_invalid_index(self, mock_device_count, mock_cuda_available):
        """Test 'cuda:N' raises ValueError for an invalid device index."""
        with self.assertRaisesRegex(ValueError, "Invalid CUDA device index: 1. Available indices: \\[0\\]"):
            resolve_device_string("cuda:1")

    @mock.patch('torch.cuda.is_available', return_value=True) # Assume CUDA available for parsing check
    def test_resolve_cuda_index_invalid_format(self, mock_cuda_available):
        """Test 'cuda:N' raises ValueError for an invalid format."""
        with self.assertRaisesRegex(ValueError, "Invalid CUDA device format"):
            resolve_device_string("cuda:abc")

    def test_resolve_invalid_string(self):
        """Test that an invalid device string raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Invalid device string"):
            resolve_device_string("invalid_device")

    def test_device_resolution_comprehensive(self):
        """Test comprehensive device resolution scenarios"""
        # Test that basic strings work
        self.assertEqual(resolve_device_string("cpu"), "cpu")
        
        # Test case sensitivity
        with self.assertRaises(ValueError):
            resolve_device_string("CPU")  # Should be lowercase
            
        with self.assertRaises(ValueError):
            resolve_device_string("CUDA")  # Should be lowercase


if __name__ == '__main__':
    unittest.main() 