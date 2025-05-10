import numpy as np
import librosa


def extract_log_mel_spectrogram(audio, sample_rate=16000, n_mels=40, n_fft=400, hop_length=160, duration_sec=1.0):
    """
    Extract log-mel spectrogram from audio with fixed frame count.

    Args:
        audio (np.ndarray): Input audio signal
        sample_rate (int): Sample rate (Hz)
        n_mels (int): Number of mel bands
        n_fft (int): FFT window size
        hop_length (int): Hop size for STFT
        duration_sec (float): Expected duration of audio (seconds)

    Returns:
        np.ndarray: Log-mel spectrogram [frames, n_mels]
    """
    # Ensure audio is the correct length
    target_samples = int(sample_rate * duration_sec)
    if len(audio) < target_samples:
        audio = np.pad(audio, (0, target_samples -
                       len(audio)), mode='constant')
    elif len(audio) > target_samples:
        audio = audio[:target_samples]

    # Compute STFT
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitudes = np.abs(stft) ** 2

    # Compute mel spectrogram
    mel_filterbank = librosa.filters.mel(
        sr=sample_rate, n_fft=n_fft, n_mels=n_mels)
    mel_spect = np.dot(mel_filterbank, magnitudes)

    # Compute log-mel
    log_mel = librosa.power_to_db(mel_spect, ref=np.max)

    # Normalize
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)

    # Ensure exact frame count
    expected_frames = int(
        np.ceil((target_samples - n_fft + hop_length) / hop_length))
    if log_mel.shape[1] < expected_frames:
        log_mel = np.pad(
            log_mel, ((0, 0), (0, expected_frames - log_mel.shape[1])), mode='constant')
    elif log_mel.shape[1] > expected_frames:
        log_mel = log_mel[:, :expected_frames]

    return log_mel.T.astype(np.float32)  # [frames, n_mels]
