import unittest
import numpy as np
import tensorflow as tf
from unittest.mock import patch
from al_bab.scripts.extract_log_mel import extract_log_mel_spectrogram


class TestExtractLogMelSpectrogram(unittest.TestCase):

    def setUp(self):
        """Set up test data for each test case"""
        # Create sample audio data - 1 second of 440Hz sine tone
        self.sample_rate = 16000
        self.duration_sec = 1.0
        t = np.linspace(0, self.duration_sec, int(
            self.sample_rate * self.duration_sec), endpoint=False)
        self.audio_1sec = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Create shorter and longer audio samples
        self.audio_short = self.audio_1sec[:int(
            self.sample_rate * 0.5)]  # 0.5 seconds
        self.audio_long = np.concatenate(
            [self.audio_1sec, self.audio_1sec])  # 2 seconds

        # Expected frame count for default parameters
        n_fft = 400
        hop_length = 160
        self.expected_frames = int(np.ceil(
            (self.sample_rate * self.duration_sec - n_fft + hop_length) / hop_length))

    def test_normal_input(self):
        """Test with normal input (1 second audio)"""
        result = extract_log_mel_spectrogram(self.audio_1sec)

        # Check shape
        self.assertEqual(result.shape[0], self.expected_frames,
                         "Output should have the expected number of frames")
        self.assertEqual(result.shape[1], 40,
                         "Output should have 40 mel bands")

        # Check type
        self.assertEqual(result.dtype, np.float32, "Output should be float32")

    def test_short_input(self):
        """Test with short input (should pad)"""
        result = extract_log_mel_spectrogram(self.audio_short)

        # Check padding worked properly
        self.assertEqual(result.shape[0], self.expected_frames,
                         "Padded audio should produce correct frame count")
        self.assertEqual(result.shape[1], 40,
                         "Output should have 40 mel bands")

    def test_long_input(self):
        """Test with long input (should truncate)"""
        result = extract_log_mel_spectrogram(self.audio_long)

        # Check truncation worked properly
        self.assertEqual(result.shape[0], self.expected_frames,
                         "Truncated audio should produce correct frame count")
        self.assertEqual(result.shape[1], 40,
                         "Output should have 40 mel bands")

    def test_custom_parameters(self):
        """Test with custom parameters"""
        custom_n_mels = 64
        custom_n_fft = 512
        custom_hop_length = 128
        custom_duration = 0.8

        result = extract_log_mel_spectrogram(
            self.audio_1sec,
            n_mels=custom_n_mels,
            n_fft=custom_n_fft,
            hop_length=custom_hop_length,
            duration_sec=custom_duration
        )

        # Calculate expected frame count for custom parameters
        expected_frames = int(np.ceil(
            (self.sample_rate * custom_duration - custom_n_fft + custom_hop_length) / custom_hop_length))

        # Check shape
        self.assertEqual(result.shape[0], expected_frames,
                         "Custom parameters should produce correct frame count")
        self.assertEqual(result.shape[1], custom_n_mels,
                         f"Output should have {custom_n_mels} mel bands")

    def test_normalization(self):
        """Test that output is normalized"""
        result = extract_log_mel_spectrogram(self.audio_1sec)

        # Check mean and std (allowing for reasonable variation due to processing steps)
        # Mean might not be exactly 0 and std might not be exactly 1 due to various signal processing operations
        self.assertLess(abs(np.mean(result)), 0.05,
                        msg="Normalized output should have mean close to 0")
        self.assertLess(abs(np.std(result) - 1), 0.02,
                        msg="Normalized output should have std close to 1")

    def test_output_range(self):
        """Test that output values are in a reasonable range"""
        result = extract_log_mel_spectrogram(self.audio_1sec)

        # Most values should be within a few standard deviations of the mean
        self.assertTrue(np.all(result > -10) and np.all(result < 10),
                        "Output values should be in a reasonable range after normalization")

    @patch('librosa.stft')
    @patch('librosa.filters.mel')
    def test_librosa_calls(self, mock_mel, mock_stft):
        """Test that librosa functions are called with correct parameters"""
        # Setup mocks
        mock_stft.return_value = np.random.randn(
            201, 100) + 1j * np.random.randn(201, 100)
        mock_mel.return_value = np.random.rand(40, 201)

        n_fft = 400
        hop_length = 160
        n_mels = 40

        extract_log_mel_spectrogram(
            self.audio_1sec,
            sample_rate=self.sample_rate,
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length
        )

        # Check librosa.stft was called with correct parameters
        mock_stft.assert_called_once()
        args, kwargs = mock_stft.call_args
        self.assertEqual(kwargs['n_fft'], n_fft)
        self.assertEqual(kwargs['hop_length'], hop_length)

        # Check librosa.filters.mel was called with correct parameters
        mock_mel.assert_called_once()
        args, kwargs = mock_mel.call_args
        self.assertEqual(kwargs['sr'], self.sample_rate)
        self.assertEqual(kwargs['n_fft'], n_fft)
        self.assertEqual(kwargs['n_mels'], n_mels)

    def test_silent_audio(self):
        """Test with silent audio"""
        silent_audio = np.zeros(int(self.sample_rate * self.duration_sec))
        result = extract_log_mel_spectrogram(silent_audio)

        # Shape should still be correct
        self.assertEqual(result.shape[0], self.expected_frames)
        self.assertEqual(result.shape[1], 40)

        # For silent audio after normalization, we might expect NaNs, but the function adds 1e-6 to std
        self.assertFalse(np.any(np.isnan(result)),
                         "Output should not contain NaN values")

    def test_tensorflow_compatibility(self):
        """Test that output can be used with TensorFlow"""
        result = extract_log_mel_spectrogram(self.audio_1sec)

        # Convert to TensorFlow tensor
        tf_tensor = tf.convert_to_tensor(result)

        # Check shape and dtype
        self.assertEqual(tf_tensor.shape[0], self.expected_frames)
        self.assertEqual(tf_tensor.shape[1], 40)
        self.assertEqual(tf_tensor.dtype, tf.float32)


if __name__ == "__main__":
    unittest.main()
