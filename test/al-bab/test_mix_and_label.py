from vad.scripts.mix_and_label import generate_mixed_sample, generate_noise_only_sample, data_generator
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def test_determinism():
    # Test parameters
    speech_dir = "data/clean_speech/train-clean-100"
    noise_dir = "data/noise/musan/noise/sound-bible"
    rir_dir = "data/rir"

    # Dynamically get the first .wav file from each directory
    speech_files = [os.path.join(speech_dir, f)
                    for f in os.listdir(speech_dir) if f.endswith(".wav")]
    if not speech_files:
        for root, _, files in os.walk(speech_dir):
            speech_files = [os.path.join(root, f)
                            for f in files if f.endswith(".wav")]
            if speech_files:
                break
    speech_files = speech_files[:1]  # Use the first file

    noise_files = [os.path.join(noise_dir, f)
                   for f in os.listdir(noise_dir) if f.endswith(".wav")]
    noise_files = noise_files[:1]  # Use the first file

    rir_files = [os.path.join(rir_dir, f) for f in os.listdir(
        rir_dir) if f.endswith(".wav")] if os.path.exists(rir_dir) else []

    if not speech_files or not noise_files:
        raise FileNotFoundError(
            "No .wav files found in speech or noise directories. Please ensure data/ contains the dataset.")

    seed = 42
    sample_rate = 16000
    duration_sec = 3

    # Test generate_mixed_sample determinism
    log_mel1, labels1 = generate_mixed_sample(
        speech_files, noise_files, rir_files, seed=seed, sample_rate=sample_rate, duration_sec=duration_sec
    )
    log_mel2, labels2 = generate_mixed_sample(
        speech_files, noise_files, rir_files, seed=seed, sample_rate=sample_rate, duration_sec=duration_sec
    )
    assert np.array_equal(
        log_mel1, log_mel2), "Mixed sample spectrograms differ with same seed"
    assert np.array_equal(
        labels1, labels2), "Mixed sample labels differ with same seed"

    # Test generate_noise_only_sample determinism
    log_mel1, labels1 = generate_noise_only_sample(
        noise_files, sample_rate, duration_sec, seed=seed)
    log_mel2, labels2 = generate_noise_only_sample(
        noise_files, sample_rate, duration_sec, seed=seed)
    assert np.array_equal(
        log_mel1, log_mel2), "Noise-only spectrograms differ with same seed"
    assert np.array_equal(
        labels1, labels2), "Noise-only labels differ with same seed"

    # Test data_generator determinism by reinitializing with the same seed and sample_idx
    def get_sample(gen, index):
        gen_instance = data_generator(
            speech_dir=speech_dir,
            noise_dir=noise_dir,
            rir_dir=rir_dir,
            sample_rate=sample_rate,
            duration_sec=duration_sec,
            seed=seed
        )
        for _ in range(index):
            next(gen_instance)
        return next(gen_instance)

    for i in range(3):  # Test first 3 samples
        log_mel1, labels1 = get_sample(data_generator, i)
        log_mel2, labels2 = get_sample(data_generator, i)
        assert np.array_equal(
            log_mel1, log_mel2), f"Generator sample {i+1} spectrograms differ"
        assert np.array_equal(
            labels1, labels2), f"Generator sample {i+1} labels differ"


if __name__ == "__main__":
    test_determinism()
    print("All determinism tests passed!")
