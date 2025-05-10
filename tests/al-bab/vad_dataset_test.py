import unittest
import tensorflow as tf
import numpy as np
import os
import tempfile
import soundfile as sf
import shutil
import sys
from unittest.mock import patch
from al_bab.scripts.vad_dataset import create_tf_dataset


class TestCreateTFDataset(unittest.TestCase):

    def setUp(self):
        """Set up test data and directories"""
        # Use the actual directory structure instead of creating temporary directories
        self.base_dir = 'synthetic_test_data'
        self.speech_dir = os.path.join(
            self.base_dir, 'libri-tts-train-clean-100')
        self.noise_dir = os.path.join(
            self.base_dir, 'nhattruongdev__musan-noise/noise')

        # Make sure directories exist
        os.makedirs(self.speech_dir, exist_ok=True)
        os.makedirs(self.noise_dir, exist_ok=True)

        # Create sample rate and duration for tests
        self.sample_rate = 16000
        self.duration_sec = 1.0

        # Create several test audio files
        self.create_test_audio_files()

    def create_test_audio_files(self):
        """Create multiple test audio files for speech and noise"""
        # Create a few different speech files
        for i in range(5):
            t = np.linspace(0, self.duration_sec, int(
                self.sample_rate * self.duration_sec), endpoint=False)
            # Generate different frequencies for variety
            frequency = 440 * (1 + i * 0.2)
            speech_audio = 0.5 * np.sin(2 * np.pi * frequency * t)
            sf.write(os.path.join(self.speech_dir, f"speech_{i}.wav"),
                     speech_audio, self.sample_rate)

        # Create a few different noise files
        # Note: corrected the path to match the directory structure shown
        noise_folder = os.path.join(
            self.base_dir, 'nhattruongdev__musan-noise/noise')
        os.makedirs(noise_folder, exist_ok=True)

        for i in range(5):
            # Generate different noise intensities
            noise_intensity = 0.05 + (i * 0.02)
            noise_audio = np.random.normal(0, noise_intensity, int(
                self.sample_rate * self.duration_sec))
            sf.write(os.path.join(noise_folder, f"noise_{i}.wav"),
                     noise_audio, self.sample_rate)

    def test_create_tf_dataset_basic(self):
        """Test basic functionality of create_tf_dataset"""
        dataset = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            seed=42,
            num_examples=4  # Small number for quick testing
        )

        # Check that dataset is a TensorFlow dataset
        self.assertIsInstance(dataset, tf.data.Dataset)

        # Check shapes of elements in a batch
        for mel_specs, vad_labels in dataset.take(1):
            self.assertEqual(mel_specs.shape, (2, 99, 40, 1))
            self.assertEqual(vad_labels.shape, (2, 99))

            # Check data types
            self.assertEqual(mel_specs.dtype, tf.float32)
            self.assertEqual(vad_labels.dtype, tf.float32)

            # Check value ranges
            # Mel specs should be normalized
            self.assertLess(tf.reduce_mean(tf.abs(mel_specs)), 5.0)

            # VAD labels should be binary (0 or 1)
            unique_values = np.unique(vad_labels.numpy())
            self.assertTrue(all(val in [0.0, 1.0] for val in unique_values))

    def test_create_tf_dataset_parameters(self):
        """Test create_tf_dataset with different parameters"""
        # Test with different batch size
        dataset = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=4,
            seed=42,
            num_examples=8
        )

        for mel_specs, vad_labels in dataset.take(1):
            self.assertEqual(mel_specs.shape, (4, 99, 40, 1))
            self.assertEqual(vad_labels.shape, (4, 99))

        # Test with different SNR range
        dataset = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            seed=42,
            snr_range=(-5, 5),  # Lower SNR (more noise)
            num_examples=4
        )

        # The dataset should still work with different SNR
        for mel_specs, vad_labels in dataset.take(1):
            self.assertEqual(mel_specs.shape, (2, 99, 40, 1))

    def test_directory_not_found(self):
        """Test error handling for non-existent directories"""
        non_existent_dir = os.path.join(self.base_dir, "non_existent")

        # Test with non-existent speech directory
        with self.assertRaises(FileNotFoundError):
            create_tf_dataset(
                speech_dir=non_existent_dir,
                noise_dir=self.noise_dir,
                batch_size=2,
                num_examples=4
            )

        # Test with non-existent noise directory
        with self.assertRaises(FileNotFoundError):
            create_tf_dataset(
                speech_dir=self.speech_dir,
                noise_dir=non_existent_dir,
                batch_size=2,
                num_examples=4
            )

    @patch('al_bab.scripts.mix_and_label.generate_labeled_example')
    @patch('al_bab.scripts.extract_log_mel.extract_log_mel_spectrogram')
    def test_error_handling_in_generator(self, mock_extract, mock_generate):
        """Test error handling in the generator"""
        # Set up mocks to simulate errors
        def side_effect_generator(*args, **kwargs):
            # Throw an exception every second call
            if side_effect_generator.counter % 2 == 0:
                side_effect_generator.counter += 1
                raise ValueError("Test error")
            side_effect_generator.counter += 1
            return np.zeros(16000), np.zeros(99), 10.0
        side_effect_generator.counter = 0
        mock_generate.side_effect = side_effect_generator

        # Mock the extract function to return valid output
        mock_extract.return_value = np.zeros((99, 40))

        # Create dataset with error-generating mock
        dataset = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            num_examples=6  # Need more examples since some will be skipped
        )

        # Should still get a valid batch despite errors
        for mel_specs, vad_labels in dataset.take(1):
            self.assertEqual(mel_specs.shape, (2, 99, 40, 1))
            self.assertEqual(vad_labels.shape, (2, 99))

    def test_dataset_reproducibility(self):
        """Test dataset reproducibility with seed"""
        # Create two datasets with the same seed
        dataset1 = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            seed=42,
            num_examples=4
        )

        dataset2 = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            seed=42,
            num_examples=4
        )

        # Extract first batch from each dataset
        batch1 = next(iter(dataset1))
        batch2 = next(iter(dataset2))

        # Check if the batches are identical (reproducibility with seed)
        np.testing.assert_allclose(
            batch1[0].numpy(), batch2[0].numpy(), rtol=1e-5)
        np.testing.assert_allclose(
            batch1[1].numpy(), batch2[1].numpy(), rtol=1e-5)

    def test_batch_remainder_drop(self):
        """Test that remainder is dropped from batches"""
        # Create dataset with 5 examples and batch size 2
        # Should result in 2 batches of 2 examples (total 4), dropping 1
        dataset = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            num_examples=5
        )

        # Count actual number of examples
        count = 0
        for batch in dataset:
            batch_size = batch[0].shape[0]
            count += batch_size

        # Should be 4, not 5 (remainder dropped)
        self.assertEqual(count, 4)

    def test_with_actual_dataset_structure(self):
        """Test with the real directory structure"""
        # Test dataset creation with the actual directories
        dataset = create_tf_dataset(
            speech_dir=self.speech_dir,
            noise_dir=self.noise_dir,
            batch_size=2,
            num_examples=4
        )

        # Verify dataset is created successfully
        for mel_specs, vad_labels in dataset.take(1):
            self.assertEqual(mel_specs.shape, (2, 99, 40, 1))
            self.assertEqual(vad_labels.shape, (2, 99))

        # Create metadata file if it doesn't exist already
        metadata_path = os.path.join(self.base_dir, "metadata.txt")
        if not os.path.exists(metadata_path):
            with open(metadata_path, "w") as f:
                f.write("Test metadata file\n")


if __name__ == "__main__":
    unittest.main()
