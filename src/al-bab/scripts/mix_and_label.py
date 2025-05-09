import os
import numpy as np
import random
import soundfile as sf
import glob
from scipy import signal
import webrtcvad
import struct


def find_audio_files(directory, extensions=['.wav', '.flac']):
    """Find all audio files with specified extensions in a directory (recursively)"""
    audio_files = []
    for ext in extensions:
        audio_files.extend(glob.glob(os.path.join(
            directory, f"**/*{ext}"), recursive=True))
    return audio_files


def load_random_audio(directory, sample_rate, min_duration=1.0, extensions=['.wav', '.flac'], seed=None):
    """Load a random audio file from the specified directory"""
    if seed is not None:
        random.seed(seed)

    audio_files = find_audio_files(directory, extensions)

    if not audio_files:
        raise ValueError(f"No audio files found in {directory}")

    random.shuffle(audio_files)

    for file_path in audio_files:
        try:
            audio, file_sr = sf.read(file_path)

            # Convert to mono if stereo
            if len(audio.shape) > 1 and audio.shape[1] > 1:
                audio = np.mean(audio, axis=1)

            # Resample if needed
            if file_sr != sample_rate:
                new_length = int(len(audio) * sample_rate / file_sr)
                audio = signal.resample(audio, new_length)

            # Check if duration is sufficient
            duration = len(audio) / sample_rate
            if duration >= min_duration:
                return audio

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    raise ValueError(
        f"Could not find audio file with duration >= {min_duration}s in {directory}")


def generate_vad_labels_webrtc(audio, sample_rate, frame_duration_ms=30, aggressiveness=3, target_frames=99):
    """
    Generate VAD labels using WebRTC VAD
    """
    vad = webrtcvad.Vad()
    vad.set_mode(aggressiveness)

    if np.max(np.abs(audio)) > 1.0:
        audio = audio / np.max(np.abs(audio))

    audio_pcm = (audio * 32767).astype(np.int16)

    frame_size = int(sample_rate * frame_duration_ms / 1000)

    webrtc_frame_durations = [10, 20, 30]
    if frame_duration_ms not in webrtc_frame_durations:
        raise ValueError(
            f"frame_duration_ms must be one of {webrtc_frame_durations}")

    if sample_rate not in [8000, 16000, 32000, 48000]:
        raise ValueError(
            "WebRTC VAD only accepts 8kHz, 16kHz, 32kHz or 48kHz audio")

    num_frames = len(audio_pcm) // frame_size

    webrtc_labels = np.zeros(num_frames, dtype=np.float32)

    for i in range(num_frames):
        start_idx = i * frame_size
        frame = audio_pcm[start_idx:start_idx + frame_size]
        frame_bytes = struct.pack("%dh" % len(frame), *frame)
        try:
            is_speech = vad.is_speech(frame_bytes, sample_rate)
            webrtc_labels[i] = float(is_speech)
        except Exception as e:
            print(f"WebRTC VAD error: {e}")
            webrtc_labels[i] = 0.0

    x_webrtc = np.linspace(0, 1, len(webrtc_labels))
    x_target = np.linspace(0, 1, target_frames)
    interp_labels = np.interp(x_target, x_webrtc, webrtc_labels)
    final_labels = (interp_labels > 0.5).astype(np.float32)

    return final_labels


def mix_audio_with_snr(speech, noise, snr_db):
    """
    Mix speech and noise at the specified SNR with corrected power and verification
    calculations. Returns mixed audio and the achieved SNR.
    """
    # Make copies to avoid modifying originals
    speech = speech.copy()
    noise = noise.copy()

    # Calculate power for both signals
    speech_power = np.mean(speech**2)
    noise_power = np.mean(noise**2)

    # Calculate RMS values for logging
    speech_rms = np.sqrt(speech_power)
    noise_rms = np.sqrt(noise_power)

    # Debug prints
    # print(f"Speech RMS: {speech_rms:.4f}")
    # print(f"Noise RMS: {noise_rms:.4f}")

    # Avoid division by zero
    if speech_power == 0:
        raise ValueError("Speech power is zero")

    # Calculate the scaling factor for noise based on the desired SNR
    # SNR = 10 * log10(speech_power / noise_power)
    # We want: noise_power_scaled = speech_power / (10^(SNR/10))
    desired_noise_power = speech_power / (10 ** (snr_db / 10))
    scaling_factor = np.sqrt(desired_noise_power / (noise_power + 1e-10))

    # Scale the noise
    scaled_noise = noise * scaling_factor
    scaled_noise_power = np.mean(scaled_noise**2)

    # Debug prints
    # print(f"Scaling Factor: {scaling_factor:.4f}")
    # print(f"Target Noise RMS: {np.sqrt(desired_noise_power):.4f}")
    # print(f"Scaled Noise RMS: {np.sqrt(scaled_noise_power):.4f}")

    # Mix the signals
    mixed = speech + scaled_noise

    # Normalize to prevent clipping
    max_abs_value = np.max(np.abs(mixed))
    if max_abs_value > 1.0:
        mixed = mixed / max_abs_value * 0.9  # Leave some headroom

    # Check the actual SNR achieved
    achieved_snr = 10 * np.log10(speech_power / (scaled_noise_power + 1e-10))
    # print(f"Achieved SNR: {achieved_snr:.2f}dB")

    # Return both the mixed audio and the achieved SNR
    return mixed, achieved_snr


def generate_labeled_example(speech_dir, noise_dir, sample_rate, duration_sec, snr_db=10, seed=None):
    """
    Generate a mixed audio example with frame-level VAD labels using WebRTC VAD.
    Returns a tuple of (mixed_audio, vad_labels, actual_snr)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)

    speech = load_random_audio(
        speech_dir, sample_rate, min_duration=duration_sec, seed=seed)
    noise = load_random_audio(
        noise_dir, sample_rate, min_duration=duration_sec, seed=seed+1 if seed else None)

    target_samples = int(duration_sec * sample_rate)

    if len(speech) > target_samples:
        start = random.randint(0, len(speech) - target_samples)
        speech = speech[start:start + target_samples]
    else:
        speech = np.pad(speech, (0, max(0, target_samples - len(speech))))

    if len(noise) > target_samples:
        start = random.randint(0, len(noise) - target_samples)
        noise = noise[start:start + target_samples]
    else:
        while len(noise) < target_samples:
            noise = np.concatenate([noise, noise])
        noise = noise[:target_samples]

    EXPECTED_MEL_FRAMES = 99

    vad_labels = generate_vad_labels_webrtc(
        speech, sample_rate, frame_duration_ms=30, aggressiveness=3, target_frames=EXPECTED_MEL_FRAMES)

    mixed_audio, actual_snr = mix_audio_with_snr(speech, noise, snr_db)

    return mixed_audio, vad_labels, actual_snr
