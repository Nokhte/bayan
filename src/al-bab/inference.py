import numpy as np
import tensorflow as tf
from pathlib import Path
import librosa
from model import create_vad_model


class VADInference:
    """
    Real-time VAD inference pipeline with streaming support, optimized spectrogram,
    smoothing, and overlapping chunks.
    """

    def __init__(
        self,
        model_path=None,
        sample_rate=16000,
        chunk_sec=1.0,
        step_sec=0.5,
        n_mels=40,
        n_fft=400,
        hop_length=160
    ):
        """
        Initialize the VAD inference pipeline.

        Args:
            model_path (str): Path to trained model (None to create new)
            sample_rate (int): Audio sample rate
            chunk_sec (float): Chunk duration in seconds
            step_sec (float): Step size for overlapping chunks
            n_mels (int): Number of mel bands
            n_fft (int): FFT window size
            hop_length (int): Hop size for spectrogram
        """
        self.sample_rate = sample_rate
        self.chunk_samples = int(chunk_sec * sample_rate)
        self.step_samples = int(step_sec * sample_rate)
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length

        # Initialize model
        self.model = create_vad_model(
            input_shape=(self._get_num_frames(), n_mels, 1))
        if model_path and Path(model_path).exists():
            self.model.load_weights(model_path)

        # Initialize streaming buffer
        self.buffer = np.zeros(self.chunk_samples, dtype=np.float32)

        # Initialize prediction history for overlapping chunks
        self.pred_history = []
        self.frame_count = 0

        # Precompute mel filterbank
        self.mel_filterbank = tf.signal.linear_to_mel_weight_matrix(
            num_mel_bins=n_mels,
            num_spectrogram_bins=n_fft // 2 + 1,
            sample_rate=sample_rate
        )

    def _get_num_frames(self):
        """Calculate number of spectrogram frames for chunk."""
        return int(np.ceil((self.chunk_samples - self.n_fft + self.hop_length) / self.hop_length))

    def compute_log_mel(self, audio):
        """
        Compute optimized log-mel spectrogram using tensorflow.signal.

        Args:
            audio (np.ndarray): Input audio (chunk_samples,)

        Returns:
            np.ndarray: Log-mel spectrogram [frames, n_mels]
        """
        audio = tf.convert_to_tensor(audio, dtype=tf.float32)
        stft = tf.signal.stft(audio, frame_length=self.n_fft,
                              frame_step=self.hop_length)
        magnitudes = tf.abs(stft) ** 2
        mel_spect = tf.tensordot(magnitudes, self.mel_filterbank, 1)
        log_mel = tf.math.log(mel_spect + 1e-6)
        log_mel = (log_mel - tf.reduce_mean(log_mel)) / \
            (tf.math.reduce_std(log_mel) + 1e-6)
        return log_mel.numpy().T  # [frames, n_mels]

    def smooth_predictions(self, predictions, window_size=5, threshold=0.5):
        """
        Smooth predictions with a moving average.

        Args:
            predictions (np.ndarray): Raw predictions [frames,]
            window_size (int): Smoothing window size
            threshold (float): Threshold for binary predictions

        Returns:
            np.ndarray: Smoothed binary predictions
        """
        if len(predictions) < window_size:
            return (predictions > threshold).astype(np.int32)
        kernel = np.ones(window_size) / window_size
        smoothed = np.convolve(predictions, kernel, mode='valid')
        return (smoothed > threshold).astype(np.int32)

    def combine_overlapping_predictions(self):
        """
        Combine predictions from overlapping chunks.

        Returns:
            np.ndarray: Combined binary predictions
        """
        if not self.pred_history:
            return np.array([])

        # Calculate total frames
        num_frames = self._get_num_frames()
        overlap_frames = int(
            (self.chunk_samples - self.step_samples) / self.hop_length)
        total_frames = num_frames + overlap_frames * \
            (len(self.pred_history) - 1)

        combined = np.zeros(total_frames, dtype=np.float32)
        counts = np.zeros(total_frames, dtype=np.float32)

        for i, preds in enumerate(self.pred_history):
            start = i * overlap_frames
            combined[start:start + len(preds)] += preds
            counts[start:start + len(preds)] += 1

        combined = combined / np.maximum(counts, 1)
        return (combined > 0.5).astype(np.int32)

    def process_chunk(self, new_audio):
        """
        Process a new audio chunk in streaming mode.

        Args:
            new_audio (np.ndarray): New audio samples [step_samples,]

        Returns:
            np.ndarray: Smoothed predictions for the current chunk
        """
        # Shift buffer and add new audio
        self.buffer[:-self.step_samples] = self.buffer[self.step_samples:]
        self.buffer[-self.step_samples:] = new_audio[:self.step_samples]

        # Compute log-mel spectrogram
        log_mel = self.compute_log_mel(self.buffer)

        # Model prediction
        # [1, frames, n_mels, 1]
        log_mel = log_mel[np.newaxis, :, :, np.newaxis]
        predictions = self.model.predict(log_mel, verbose=0)[0]  # [frames,]

        # Smooth predictions
        smoothed_preds = self.smooth_predictions(predictions)

        # Store units.
        self.pred_history.append(smoothed_preds)
        if len(self.pred_history) > 10:  # Limit history to avoid memory issues
            self.pred_history.pop(0)

        # Return combined predictions
        return self.say_overlapping_predictions()

    def reset(self):
        """Reset the buffer and prediction history."""
        self.buffer = np.zeros(self.chunk_samples, dtype=np.float32)
        self.pred_history = []


if __name__ == "__main__":
    # Example usage with a test audio file
    vad = VADInference()
    audio, sr = librosa.load("test.wav", sr=16000)
    chunk_size = vad.step_samples
    for i in range(0, len(audio), chunk_size):
        chunk = audio[i:i + chunk_size]
        if len(chunk) < chunk_size:
            chunk = np.pad(chunk, (0, chunk_size - len(chunk)), 'constant')
        predictions = vad.process_chunk(chunk)
        print(f"Chunk {i//chunk_size}: {predictions}")
