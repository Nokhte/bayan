from al_bab.scripts.mix_and_label import (
    find_audio_files,
    load_random_audio,
    generate_vad_labels_webrtc,
    mix_audio_with_snr,
    generate_labeled_example
)
import unittest
import numpy as np
import os
import tempfile
import soundfile as sf
from unittest.mock import patch, MagicMock
import shutil


class TestAudioProcessing(unittest.TestCase):

    def setUp(self):
        """Set up test data and directories"""
        # Create temporary directories for test audio files
        self.test_dir = tempfile.mkdtemp()
        self.speech_dir = os.path.join(self.test_dir, "speech")
        self.noise_dir = os.path.join(self.test_dir, "noise")
        os.makedirs(self.speech_dir, exist_ok=True)
        os.makedirs(self.noise_dir, exist_ok=True)

        # Create sample rate and duration for tests
        self.sample_rate = 16000
        self.duration_sec = 1.0

        # Generate test audio data
        self.create_test_audio_files()

    def tearDown(self):
        """Clean up temporary files and directories"""
        shutil.rmtree(self.test_dir)

    def create_test_audio_files(self):
        """Create test audio files for speech and noise"""
        # Create a simple sine wave for speech
        t = np.linspace(0, self.duration_sec, int(
            self.sample_rate * self.duration_sec), endpoint=False)
        speech_audio = 0.5 * np.sin(2 * np.pi * 440 * t)

        # Create white noise for noise
        noise_audio = np.random.normal(0, 0.1, int(
            self.sample_rate * self.duration_sec))

        # Create a longer speech file
        long_speech_audio = np.concatenate([speech_audio, speech_audio])

        # Create stereo audio
        stereo_audio = np.column_stack([speech_audio, speech_audio * 0.8])

        # Write the files
        sf.write(os.path.join(self.speech_dir, "speech1.wav"),
                 speech_audio, self.sample_rate)
        sf.write(os.path.join(self.speech_dir, "speech2.wav"),
                 long_speech_audio, self.sample_rate)
        sf.write(os.path.join(self.speech_dir, "stereo.wav"),
                 stereo_audio, self.sample_rate)
        sf.write(os.path.join(self.noise_dir, "noise1.wav"),
                 noise_audio, self.sample_rate)

        # Create a file with different sample rate
        sf.write(os.path.join(self.speech_dir,
                 "speech_8k.wav"), speech_audio, 8000)

        # Store references for tests
        self.speech_audio = speech_audio
        self.noise_audio = noise_audio

    def test_find_audio_files(self):
        """Test finding audio files in directories"""
        # Test with default extensions
        files = find_audio_files(self.speech_dir)
        self.assertEqual(len(files), 4)  # Should find 4 wav files

        # Test with specific extension
        files = find_audio_files(self.speech_dir, extensions=['.flac'])
        self.assertEqual(len(files), 0)  # No flac files

        # Test with multiple extensions
        files = find_audio_files(self.test_dir, extensions=['.wav', '.flac'])
        self.assertEqual(len(files), 5)  # 5 wav files in total

    def test_load_random_audio(self):
        """Test loading random audio files"""
        # Test basic functionality
        audio = load_random_audio(self.speech_dir, self.sample_rate, seed=42)
        self.assertIsNotNone(audio)
        self.assertTrue(isinstance(audio, np.ndarray))
        self.assertTrue(len(audio) >= self.sample_rate * self.duration_sec)

        # Test with minimum duration constraint
        audio = load_random_audio(
            self.speech_dir, self.sample_rate, min_duration=1.5, seed=42)
        self.assertTrue(len(audio) >= self.sample_rate * 1.5)

        # Test loading stereo file (should convert to mono)
        with patch('random.shuffle') as mock_shuffle:
            mock_shuffle.side_effect = lambda x: x.sort(
                key=lambda f: "stereo" in f, reverse=True)
            audio = load_random_audio(self.speech_dir, self.sample_rate)
            self.assertEqual(len(audio.shape), 1)  # Should be mono

    def test_load_random_audio_resampling(self):
        """Test resampling in load_random_audio"""
        with patch('random.shuffle') as mock_shuffle:
            # Force it to pick the 8kHz file
            mock_shuffle.side_effect = lambda x: x.sort(
                key=lambda f: "8k" in f, reverse=True)

            audio = load_random_audio(self.speech_dir, 16000)
            # Length should be doubled from the original 8kHz file
            self.assertAlmostEqual(len(audio), len(
                self.speech_audio) * 2, delta=2)

    def test_load_random_audio_errors(self):
        """Test error handling in load_random_audio"""
        # Test with empty directory
        empty_dir = os.path.join(self.test_dir, "empty")
        os.makedirs(empty_dir, exist_ok=True)
        with self.assertRaises(ValueError):
            load_random_audio(empty_dir, self.sample_rate)

        # Test with no files of sufficient duration
        with self.assertRaises(ValueError):
            load_random_audio(
                self.speech_dir, self.sample_rate, min_duration=10.0)

    def test_generate_vad_labels_webrtc(self):
        """Test WebRTC VAD label generation"""
        # Generate VAD labels
        target_frames = 99
        vad_labels = generate_vad_labels_webrtc(
            self.speech_audio, self.sample_rate, frame_duration_ms=30,
            aggressiveness=3, target_frames=target_frames
        )

        # Check shape and type
        self.assertEqual(len(vad_labels), target_frames)
        self.assertEqual(vad_labels.dtype, np.float32)

        # Values should be either 0.0 or 1.0
        unique_values = np.unique(vad_labels)
        self.assertTrue(np.all(np.isin(unique_values, [0.0, 1.0])))

        # Check error handling for invalid parameters
        with self.assertRaises(ValueError):
            generate_vad_labels_webrtc(
                self.speech_audio, 12000)  # Invalid sample rate

        with self.assertRaises(ValueError):
            generate_vad_labels_webrtc(
                self.speech_audio, self.sample_rate, frame_duration_ms=25)  # Invalid frame duration

    @patch('webrtcvad.Vad')
    def test_vad_webrtc_mocked(self, mock_vad_class):
        """Test WebRTC VAD with mocked WebRTC VAD class"""
        # Setup mock
        mock_vad = MagicMock()
        mock_vad_class.return_value = mock_vad

        # Configure the mock to alternate between speech and non-speech
        mock_vad.is_speech.side_effect = lambda frame, rate: bool(
            int.from_bytes(frame[:1], byteorder='big') % 2)

        # Generate labels
        vad_labels = generate_vad_labels_webrtc(
            self.speech_audio, self.sample_rate, frame_duration_ms=30, target_frames=99
        )

        # Check that mock was called
        self.assertTrue(mock_vad.is_speech.called)

        # Check that set_mode was called with the expected aggressiveness
        mock_vad.set_mode.assert_called_once_with(3)

    def test_mix_audio_with_snr(self):
        """Test mixing audio with specified SNR"""
        # Test mixing with positive SNR
        mixed, achieved_snr = mix_audio_with_snr(
            self.speech_audio, self.noise_audio, 10)

        # Check output shape
        self.assertEqual(len(mixed), len(self.speech_audio))

        # Check that mixing was done at approximately the requested SNR
        self.assertAlmostEqual(achieved_snr, 10, delta=1.0)

        # Check that output is normalized
        self.assertLessEqual(np.max(np.abs(mixed)), 1.0)

        # Test with negative SNR
        mixed, achieved_snr = mix_audio_with_snr(
            self.speech_audio, self.noise_audio, -5)
        self.assertAlmostEqual(achieved_snr, -5, delta=1.0)

        # Test error handling
        with self.assertRaises(ValueError):
            # Test with zero speech
            mix_audio_with_snr(np.zeros_like(
                self.speech_audio), self.noise_audio, 10)

    def test_generate_labeled_example(self):
        """Test generation of labeled examples"""
        # Generate an example
        mixed_audio, vad_labels, actual_snr = generate_labeled_example(
            self.speech_dir, self.noise_dir, self.sample_rate, self.duration_sec, snr_db=10, seed=42
        )

        # Check shapes
        self.assertEqual(len(mixed_audio), int(
            self.sample_rate * self.duration_sec))
        self.assertEqual(len(vad_labels), 99)  # Expected number of frames

        # Check SNR is approximately as requested
        self.assertAlmostEqual(actual_snr, 10, delta=2.0)

        # Check audio is normalized
        self.assertLessEqual(np.max(np.abs(mixed_audio)), 1.0)

        # Check labels are binary
        unique_values = np.unique(vad_labels)
        self.assertTrue(np.all(np.isin(unique_values, [0.0, 1.0])))


if __name__ == "__main__":
    unittest.main()
