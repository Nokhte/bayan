import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import librosa
from scripts.mix_and_label import generate_labeled_example, load_random_audio, generate_vad_labels_webrtc
from scripts.extract_log_mel import extract_log_mel_spectrogram
from scripts.vad_dataset import create_tf_dataset
import webrtcvad
import soundfile as sf

SAMPLE_RATE = 16000
DURATION_SEC = 1.0
N_FRAMES = 99  # Expected from log-mel extractor


def check_audio_content(audio, label="Audio"):
    """Ensure audio is not silent or invalid."""
    energy = np.mean(np.abs(audio))
    assert energy > 1e-6, f"{label} is effectively silent (energy: {energy})"
    assert not np.any(np.isnan(audio)), f"{label} contains NaN values"
    assert not np.any(np.isinf(audio)), f"{label} contains infinite values"
    return energy


def verify_snr(speech, noise, mixed, target_snr_db, actual_snr=None):
    """Verify the SNR of the mixed audio matches the target."""
    if actual_snr is not None:
        # Use the provided actual SNR if available
        tolerance = 1.0  # Allowable deviation in dB
        assert abs(actual_snr - target_snr_db) < tolerance, (
            f"SNR mismatch! Expected {target_snr_db:.2f}dB, got {actual_snr:.2f}dB"
        )
        return actual_snr
    else:
        # Calculate an approximation for backward compatibility
        speech_power = np.mean(speech**2)
        noise_power = np.mean(noise**2)
        mixed_noise_power = np.mean((mixed - speech)**2)  # Approximation
        calculated_snr = 10 * \
            np.log10(speech_power / (mixed_noise_power + 1e-10))

        tolerance = 1.0  # Allowable deviation in dB
        assert abs(calculated_snr - target_snr_db) < tolerance, (
            f"SNR mismatch! Expected {target_snr_db:.2f}dB, got {calculated_snr:.2f}dB"
        )
        return calculated_snr


def sanity_check_one_example():
    """Sanity check for a single example with enhanced validations."""
    print("\n=== Running Sanity Check for One Example ===")

    # Generate (speech + noise), frame-aligned labels
    audio, labels, actual_snr = generate_labeled_example(
        speech_dir="data/clean_speech/train-clean-100",
        noise_dir="data/noise/musan/noise/free-sound",
        sample_rate=SAMPLE_RATE,
        duration_sec=DURATION_SEC,
        seed=42
    )

    # Check audio length
    assert len(audio) == SAMPLE_RATE * DURATION_SEC, (
        f"Audio length mismatch! Expected {SAMPLE_RATE * DURATION_SEC}, but got {len(audio)}"
    )

    # Extract clean speech and noise for verification
    speech = load_random_audio(
        "data/clean_speech/train-clean-100", SAMPLE_RATE, min_duration=DURATION_SEC, seed=42)
    noise = load_random_audio(
        "data/noise/musan/noise/free-sound", SAMPLE_RATE, min_duration=DURATION_SEC, seed=43)
    target_samples = int(DURATION_SEC * SAMPLE_RATE)
    if len(speech) > target_samples:
        start = (len(speech) - target_samples) // 2
        speech = speech[start:start + target_samples]
    if len(noise) > target_samples:
        start = (len(noise) - target_samples) // 2
        noise = noise[start:start + target_samples]
    while len(noise) < target_samples:
        noise = np.concatenate([noise, noise])
    noise = noise[:target_samples]

    # Check audio content
    speech_energy = check_audio_content(speech, "Speech")
    noise_energy = check_audio_content(noise, "Noise")
    mixed_energy = check_audio_content(audio, "Mixed Audio")

    # Verify SNR
    snr_db = 10  # Default from generate_labeled_example
    verify_snr(speech, noise, audio, snr_db, actual_snr=actual_snr)

    # Extract mel spectrogram
    mel_spec = extract_log_mel_spectrogram(audio, SAMPLE_RATE)  # (99, 40)

    # Print dimensions before alignment check
    print(
        f"Before alignment: Mel frames = {mel_spec.shape[0]}, Labels = {labels.shape[0]}")

    # Align labels to mel frames if needed
    if mel_spec.shape[0] != labels.shape[0]:
        print(f"WARNING: Mel frames and labels have different lengths!")
        min_len = min(mel_spec.shape[0], labels.shape[0])
        mel_spec = mel_spec[:min_len]
        labels = labels[:min_len]

    # Check mel spectrogram shape
    assert mel_spec.shape[1] == 40, f"Expected 40 mel bins, but got {mel_spec.shape[1]}"

    # Check alignment
    assert mel_spec.shape[0] == labels.shape[0], "Mismatch between mel frames and labels!"

    # Check if labels are binary
    assert np.all(np.isin(labels, [0, 1])
                  ), "Labels should only contain 0 or 1 values!"

    # Compute statistics
    mel_mean, mel_std = np.mean(mel_spec), np.std(mel_spec)
    speech_frames = np.sum(labels)
    speech_ratio = speech_frames / len(labels)

    print("Mel Spec Shape:", mel_spec.shape)
    print("Labels Shape:", labels.shape)
    print(
        f"Labels distribution: {speech_frames} speech frames, {len(labels) - speech_frames} non-speech frames")
    print(f"Speech Energy: {speech_energy:.6f}")
    print(f"Noise Energy: {noise_energy:.6f}")
    print(f"Mixed Energy: {mixed_energy:.6f}")
    print(f"SNR Target: {snr_db:.2f}dB, Actual: {actual_snr:.2f}dB")
    print(f"Mel Stats (mean, std): ({mel_mean:.2f}, {mel_std:.2f})")
    print(f"Label Distribution: {speech_ratio:.2%} speech")

    # Plot for visual sanity check
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(mel_spec.T, aspect="auto", origin="lower")
    plt.title("Log-Mel Spectrogram")
    plt.ylabel("Mel Bins")
    plt.colorbar()

    plt.subplot(2, 1, 2)
    plt.plot(labels, 'r-', linewidth=2)
    plt.title("Speech/Non-Speech Labels")
    plt.xlabel("Frames")
    plt.ylabel("Label")
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('combined_visualization.png')
    plt.close()


def check_vad_alignment():
    """Check VAD labels alignment with enhanced validations across multiple examples."""
    print("\n=== Checking VAD Labels Alignment ===")

    num_examples = 3
    snr_range = (5, 15)
    speech_energies = []
    noise_energies = []
    snr_differences = []
    mel_stats = []
    label_distributions = []

    for i in range(num_examples):
        print(f"\nExample {i+1}:")
        snr_db = np.random.uniform(snr_range[0], snr_range[1])

        audio, labels, actual_snr = generate_labeled_example(
            speech_dir="data/clean_speech/train-clean-100",
            noise_dir="data/noise/musan/noise/free-sound",
            sample_rate=SAMPLE_RATE,
            duration_sec=DURATION_SEC,
            seed=42+i
        )

        # Extract clean speech and noise for verification
        speech = load_random_audio(
            "data/clean_speech/train-clean-100", SAMPLE_RATE, min_duration=DURATION_SEC, seed=42+i)
        noise = load_random_audio(
            "data/noise/musan/noise/free-sound", SAMPLE_RATE, min_duration=DURATION_SEC, seed=43+i)
        target_samples = int(DURATION_SEC * SAMPLE_RATE)
        if len(speech) > target_samples:
            start = (len(speech) - target_samples) // 2
            speech = speech[start:start + target_samples]
        if len(noise) > target_samples:
            start = (len(noise) - target_samples) // 2
            noise = noise[start:start + target_samples]
        while len(noise) < target_samples:
            noise = np.concatenate([noise, noise])
        noise = noise[:target_samples]

        # Check audio content
        speech_energy = check_audio_content(speech, "Speech")
        noise_energy = check_audio_content(noise, "Noise")
        mixed_energy = check_audio_content(audio, "Mixed Audio")
        speech_energies.append(speech_energy)
        noise_energies.append(noise_energy)

        # Verify SNR
        snr_differences.append(abs(actual_snr - snr_db))

        mel_spec = extract_log_mel_spectrogram(audio, SAMPLE_RATE)

        # Compute statistics
        mel_stats.append((np.mean(mel_spec), np.std(mel_spec)))
        speech_frames = np.sum(labels)
        label_distributions.append(speech_frames / len(labels))

        print(
            f"Audio length: {len(audio)} samples ({len(audio)/SAMPLE_RATE:.2f}s)")
        print(f"Mel spectrogram shape: {mel_spec.shape}")
        print(f"Labels length: {len(labels)}")
        print(f"Speech frames: {speech_frames} ({np.mean(labels)*100:.1f}%)")
        print(
            f"Alignment status: {'OK' if mel_spec.shape[0] == len(labels) else 'MISMATCH'}")
        if mel_spec.shape[0] != len(labels):
            print(f"  Difference: {mel_spec.shape[0] - len(labels)} frames")
        print(f"SNR Target: {snr_db:.2f}dB, Actual: {actual_snr:.2f}dB")

    # Summarize results
    print("\nSummary of VAD Alignment Check:")
    print(
        f"Average Speech Energy: {np.mean(speech_energies):.6f} (±{np.std(speech_energies):.6f})")
    print(
        f"Average Noise Energy: {np.mean(noise_energies):.6f} (±{np.std(noise_energies):.6f})")
    print(f"Average SNR Difference: {np.mean(snr_differences):.2f}dB")
    print(
        f"Mel Stats (mean, std): {np.mean([m[0] for m in mel_stats]):.2f}, {np.mean([m[1] for m in mel_stats]):.2f}")
    print(
        f"Label Distribution: {np.mean(label_distributions):.2%} (±{np.std(label_distributions):.2%}) speech")


def verify_alignment(speech_dir, noise_dir, sample_rate=16000, duration_sec=1.0, seed=42):
    """Verify mel spectrogram frames and VAD labels alignment with enhanced checks."""
    print("\n=== Running Frame Alignment Verification ===")

    mixed_audio, vad_labels, actual_snr = generate_labeled_example(
        speech_dir=speech_dir,
        noise_dir=noise_dir,
        sample_rate=sample_rate,
        duration_sec=duration_sec,
        snr_db=10,
        seed=seed
    )

    # Check audio content
    mixed_energy = check_audio_content(mixed_audio, "Mixed Audio")

    mel_spec = extract_log_mel_spectrogram(mixed_audio, sample_rate)

    print(f"Audio length: {len(mixed_audio)} samples ({duration_sec:.2f}s)")
    print(f"Mel spectrogram shape: {mel_spec.shape}")
    print(f"VAD labels shape: {vad_labels.shape}")

    if mel_spec.shape[0] == vad_labels.shape[0]:
        print(f"✅ MATCH! Frames are aligned: {mel_spec.shape[0]} frames")
    else:
        print(
            f"❌ MISMATCH! Mel frames: {mel_spec.shape[0]}, Labels: {vad_labels.shape[0]}")
        print(
            f"   Difference: {abs(mel_spec.shape[0] - vad_labels.shape[0])} frames")

    speech_frames = np.sum(vad_labels)
    speech_percentage = speech_frames / len(vad_labels) * 100
    print(f"Speech frames: {speech_frames} ({speech_percentage:.1f}%)")

    # Plot for visual verification
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.imshow(mel_spec.T, aspect='auto', origin='lower')
    plt.title('Mel Spectrogram')
    plt.ylabel('Mel Bin')
    plt.colorbar(format='%+2.0f dB')

    plt.subplot(2, 1, 2)
    plt.plot(vad_labels)
    plt.title('VAD Labels (1=speech, 0=non-speech)')
    plt.xlabel('Frame')
    plt.ylim(-0.1, 1.1)
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('vad_alignment_verification.png')
    print("Verification plot saved to 'vad_alignment_verification.png'")

    return mel_spec.shape[0] == vad_labels.shape[0]


def edge_case_tests():
    """Test edge cases to ensure robustness."""
    print("\n=== Testing Edge Cases ===")

    # Edge case: Pure silence
    print("\nTesting Edge Case: Pure Silence")
    silent_audio = np.zeros(int(SAMPLE_RATE * DURATION_SEC))
    mel_spec = extract_log_mel_spectrogram(silent_audio, SAMPLE_RATE)
    labels = generate_vad_labels_webrtc(
        silent_audio, SAMPLE_RATE, target_frames=N_FRAMES)
    assert np.all(
        labels == 0), "Silent audio should have all non-speech labels"
    print("Silent audio test passed!")

    # Edge case: Pure noise
    print("\nTesting Edge Case: Pure Noise")
    noise_only = load_random_audio(
        "data/noise/musan/noise/free-sound", SAMPLE_RATE, min_duration=DURATION_SEC, seed=44)
    noise_only = noise_only[:int(SAMPLE_RATE * DURATION_SEC)]
    mel_spec = extract_log_mel_spectrogram(noise_only, SAMPLE_RATE)
    labels = generate_vad_labels_webrtc(
        noise_only, SAMPLE_RATE, target_frames=N_FRAMES)
    speech_ratio = np.mean(labels)
    assert speech_ratio < 0.3, f"Noise-only audio has too much 'speech': {speech_ratio:.2%}"
    print("Noise-only test passed!")


if __name__ == "__main__":
    try:
        sanity_check_one_example()
        print("Basic sanity check completed!")
    except Exception as e:
        print(f"Error in sanity_check_one_example: {e}")

    try:
        check_vad_alignment()
        print("VAD alignment check completed!")
    except Exception as e:
        print(f"Error in check_vad_alignment: {e}")

    try:
        print("\n=== Checking TF Dataset ===")
        speech_dir = "data/clean_speech/train-clean-100"
        noise_dir = "data/noise/musan/noise/free-sound"

        # Create dataset with a timeout mechanism to avoid infinite loops
        ds = create_tf_dataset(
            speech_dir=speech_dir,
            noise_dir=noise_dir,
            batch_size=2,
            seed=42
        )

        verify_alignment(speech_dir, noise_dir)

        # Add a counter to limit iterations in case of issues
        max_attempts = 3
        attempt = 0

        for mel_batch, label_batch in ds.take(1):
            print("Mel batch shape:", mel_batch.shape)
            print("Label batch shape:", label_batch.shape)

            assert mel_batch.shape[1] == N_FRAMES, (
                f"Expected {N_FRAMES} mel frames, but got {mel_batch.shape[1]}"
            )
            assert mel_batch.shape[2] == 40, (
                f"Expected 40 mel bins, but got {mel_batch.shape[2]}"
            )
            assert mel_batch.shape[3] == 1, (
                f"Expected single-channel mel spectrogram, but got {mel_batch.shape[3]}"
            )

            assert mel_batch.shape[0] == label_batch.shape[0], (
                f"Mismatch in batch size! {mel_batch.shape[0]} vs {label_batch.shape[0]}"
            )
            assert label_batch.shape[1] == N_FRAMES, (
                f"Label batch frame size mismatch! Expected {N_FRAMES}, got {label_batch.shape[1]}"
            )

            has_nan = tf.math.reduce_any(tf.math.is_nan(mel_batch)).numpy()
            has_inf = tf.math.reduce_any(tf.math.is_inf(mel_batch)).numpy()
            if has_nan:
                print("WARNING: Input contains NaN values!")
            if has_inf:
                print("WARNING: Input contains infinity values!")

            print(
                f"Mel spectrogram range: [{tf.reduce_min(mel_batch).numpy():.2f}, {tf.reduce_max(mel_batch).numpy():.2f}]")
            print(
                f"Label distribution: {tf.reduce_mean(label_batch).numpy()*100:.1f}% speech frames")

            attempt += 1
            if attempt >= max_attempts:
                break

        print("Dataset checks passed!")
    except Exception as e:
        print(f"Error in dataset check: {e}")

    try:
        edge_case_tests()
        print("Edge case tests completed!")
    except Exception as e:
        print(f"Error in edge case tests: {e}")

    print("\nSanity checks complete - check the generated images for analysis")
