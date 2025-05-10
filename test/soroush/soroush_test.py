import os
import unittest
import numpy as np
import tensorflow as tf
import librosa
import subprocess
from uuid import uuid4

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Constants
FIXED_LENGTH = 48000
TARGET_RATE = 48000
THRESHOLD = 0.83

def pad_or_truncate(audio, target_length):
    """Pad or truncate audio to fixed length."""
    if len(audio) > target_length:
        return audio[:target_length]
    elif len(audio) < target_length:
        return np.pad(audio, (0, target_length - len(audio)), "constant")
    return audio

class SoroushVoiceModelTestSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures, loading models and audio files."""
        cls.audio_dir = "audio/voice"
        cls.audio_files = [x for x in os.listdir(cls.audio_dir) if x.endswith(".wav")]
        
        # Expected results: (file1, file2): (expected_conclusion, expected_distance)
        cls.expected_results = {
            ("mom_voice.wav", "sonny_voice_2.wav"): ("Different people", 1.269),
            ("mom_voice.wav", "brother_voice.wav"): ("Different people", 1.492),
            ("mom_voice.wav", "sonny_voice.wav"): ("Different people", 1.219),
            ("sonny_voice_2.wav", "brother_voice.wav"): ("Different people", 1.348),
            ("sonny_voice_2.wav", "sonny_voice.wav"): ("Same person", 0.755),
            ("brother_voice.wav", "sonny_voice.wav"): ("Different people", 1.428)
        }
        
        # Load TFLite model
        cls.interpreter = tf.lite.Interpreter(model_path="out/soroush.tflite")
        cls.interpreter.allocate_tensors()
        cls.input_details = cls.interpreter.get_input_details()
        cls.output_details = cls.interpreter.get_output_details()
        
        # Load SavedModel
        cls.model = tf.saved_model.load("soroush-model")
        
        # Pre-process audio files
        cls.audio_data = {}
        for file in cls.audio_files:
            file_path = os.path.join(cls.audio_dir, file)
            audio, _ = librosa.load(file_path, sr=TARGET_RATE)
            audio = pad_or_truncate(audio, FIXED_LENGTH)
            cls.audio_data[file] = tf.convert_to_tensor(audio, dtype=tf.float32)

    def get_embedding(self, audio_data, use_lite=False):
        """Get embedding from audio data using specified model."""
        audio_tensor = tf.expand_dims(audio_data, axis=0)
        
        if use_lite:
            audio_np = audio_tensor.numpy().astype(np.float32)
            self.interpreter.set_tensor(self.input_details[0]["index"], audio_np)
            self.interpreter.invoke()
            return self.interpreter.get_tensor(self.output_details[0]["index"])[0]
        return self.model(audio_tensor)[0]

    def compare_voices(self, file1, file2, use_lite=False):
        """Compare two voice files and return distance and conclusion."""
        emb1 = self.get_embedding(self.audio_data[file1], use_lite)
        emb2 = self.get_embedding(self.audio_data[file2], use_lite)
        
        distance = np.linalg.norm(emb1 - emb2)
        conclusion = "Same person" if distance < THRESHOLD else "Different people"
        
        return conclusion, distance

    def test_savedmodel_comparisons(self):
        """Test voice comparisons using SavedModel."""
        for (file1, file2), (expected_conclusion, expected_distance) in self.expected_results.items():
            with self.subTest(file1=file1, file2=file2):
                conclusion, distance = self.compare_voices(file1, file2, use_lite=False)
                
                self.assertEqual(
                    conclusion, 
                    expected_conclusion,
                    f"SavedModel comparison of {file1} and {file2} failed: got '{conclusion}', expected '{expected_conclusion}'"
                )
                
                self.assertAlmostEqual(
                    distance, 
                    expected_distance, 
                    delta=0.01,
                    msg=f"SavedModel distance for {file1} and {file2} was {distance:.4f}, expected ~{expected_distance:.4f}"
                )

    def test_tflite_comparisons(self):
        """Test voice comparisons using TFLite model."""
        for (file1, file2), (expected_conclusion, expected_distance) in self.expected_results.items():
            with self.subTest(file1=file1, file2=file2):
                conclusion, distance = self.compare_voices(file1, file2, use_lite=True)
                
                self.assertEqual(
                    conclusion, 
                    expected_conclusion,
                    f"TFLite comparison of {file1} and {file2} failed: got '{conclusion}', expected '{expected_conclusion}'"
                )
                
                self.assertAlmostEqual(
                    distance, 
                    expected_distance, 
                    delta=0.01,
                    msg=f"TFLite distance for {file1} and {file2} was {distance:.4f}, expected ~{expected_distance:.4f}"
                )

    def test_model_consistency(self):
        """Verify SavedModel and TFLite produce consistent results."""
        for file1 in self.audio_files:
            for file2 in self.audio_files:
                if file1 == file2:
                    continue
                with self.subTest(file1=file1, file2=file2):
                    sm_conclusion, sm_distance = self.compare_voices(file1, file2, use_lite=False)
                    tfl_conclusion, tfl_distance = self.compare_voices(file1, file2, use_lite=True)
                    
                    self.assertEqual(
                        sm_conclusion, 
                        tfl_conclusion,
                        f"Models disagree on {file1} and {file2}: SavedModel='{sm_conclusion}', TFLite='{tfl_conclusion}'"
                    )
                    
                    self.assertAlmostEqual(
                        sm_distance, 
                        tfl_distance, 
                        delta=0.01,
                        msg=f"Distance inconsistency for {file1} and {file2}: SavedModel={sm_distance:.4f}, TFLite={tfl_distance:.4f}"
                    )

    def test_cli_commands(self):
        """Test CLI commands for both models."""
        # Test SavedModel CLI
        regular_output = subprocess.check_output(
            ["python", "test/soroush_test.py"],
            universal_newlines=True,
            stderr=subprocess.STDOUT
        )
        
        # Test TFLite CLI
        lite_output = subprocess.check_output(
            ["python", "test/soroush_test.py", "--use-lite"],
            universal_newlines=True,
            stderr=subprocess.STDOUT
        )
        
        # Verify outputs contain expected results
        for (file1, file2), (expected_conclusion, _) in self.expected_results.items():
            with self.subTest(file1=file1, file2=file2):
                expected_text = f"{file1} and {file2}: {expected_conclusion}:"
                self.assertIn(
                    expected_text, 
                    regular_output,
                    f"SavedModel CLI output missing result for {file1} and {file2}"
                )
                self.assertIn(
                    expected_text, 
                    lite_output,
                    f"TFLite CLI output missing result for {file1} and {file2}"
                )

if __name__ == "__main__":
    unittest.main()