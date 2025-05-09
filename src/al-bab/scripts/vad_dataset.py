import tensorflow as tf
import numpy as np
import os
from scripts.mix_and_label import generate_labeled_example
from scripts.extract_log_mel import extract_log_mel_spectrogram


def create_tf_dataset(speech_dir, noise_dir, batch_size=32, seed=None, snr_range=(5, 15), num_examples=16000):
    """
    Create a TensorFlow dataset for voice activity detection.

    Args:
        speech_dir: Directory containing speech files
        noise_dir: Directory containing noise files
        batch_size: Number of samples per training batch
        seed: Random seed for reproducibility
        snr_range: Tuple of SNR values (min_dB, max_dB)
        num_examples: Number of total examples to generate (usually batch_size * steps_per_epoch)

    Returns:
        A batched, shuffled TensorFlow dataset yielding (mel_spec, vad_labels)
    """
    print(
        f"Creating dataset with {num_examples} examples (batch size = {batch_size})")
    print(f"Speech dir: {speech_dir}")
    print(f"Noise dir: {noise_dir}")
    print(f"SNR range: {snr_range} dB")

    if not os.path.exists(speech_dir):
        raise FileNotFoundError(f"Speech directory not found: {speech_dir}")
    if not os.path.exists(noise_dir):
        raise FileNotFoundError(f"Noise directory not found: {noise_dir}")

    def generator():
        rng = np.random.RandomState(seed)
        for _ in range(num_examples):
            try:
                snr_db = rng.uniform(snr_range[0], snr_range[1])
                mixed_audio, vad_labels, _ = generate_labeled_example(
                    speech_dir=speech_dir,
                    noise_dir=noise_dir,
                    sample_rate=16000,
                    duration_sec=1.0,
                    snr_db=snr_db,
                    seed=rng.randint(0, 1_000_000)
                )
                mel_spec = extract_log_mel_spectrogram(mixed_audio, 16000)
                mel_spec = mel_spec[..., np.newaxis]  # Add channel dim
                yield mel_spec, vad_labels
            except Exception as e:
                print(f"Skipping bad example: {e}")
                continue

    output_signature = (
        tf.TensorSpec(shape=(99, 40, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(99,), dtype=tf.float32),
    )

    dataset = tf.data.Dataset.from_generator(
        generator, output_signature=output_signature)
    dataset = dataset.shuffle(buffer_size=min(
        1000, num_examples), seed=seed, reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset
