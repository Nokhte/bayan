import numpy as np
from scipy.io import wavfile
from pathlib import Path

def create_synthetic_test_data(output_dir: str = "synthetic_test_data"):
    """
    Creates synthetic test data that mimics the structure of LibriTTS and MUSAN
    without downloading any external files.
    
    This generates:
    - Small audio files with synthetic speech-like sine waves
    - Directory structure matching LibriTTS (speaker/chapter/utterance)
    - Noise samples in MUSAN structure
    
    Args:
        output_dir: Directory to save the generated files
    """
    base_out = Path(output_dir)
    
    # Create synthetic dataset structure mimicking LibriTTS format
    datasets = {
        "libri-tts-train-clean-100": (3, 2, 3),  # 3 speakers, 2 chapters each, 3 utterances per chapter
        "libri-tts-train-clean-360": (4, 2, 3),  # 4 speakers, 2 chapters each, 3 utterances per chapter
    }
    
    print("Generating synthetic test data...")
    
    for dataset_name, (num_speakers, num_chapters, num_utterances) in datasets.items():
        dataset_dir = base_out / dataset_name.replace("/", "__")
        print(f"Creating {dataset_name} structure...")
        
        # Create speakers with chapters
        for speaker_id in range(101, 101 + num_speakers):
            for chapter_id in range(1001, 1001 + num_chapters):
                chapter_dir = dataset_dir / f"{speaker_id}" / f"{chapter_id}"
                chapter_dir.mkdir(parents=True, exist_ok=True)
                
                # Create utterances per chapter
                for i in range(1, num_utterances + 1):
                    filename = f"{speaker_id}_{chapter_id}_{i:03d}.wav"
                    file_path = chapter_dir / filename
                    
                    # Generate a simple sine wave (1 second, 16kHz)
                    sample_rate = 16000
                    duration = 1.0
                    t = np.linspace(0, duration, int(sample_rate * duration))
                    frequency = 440.0 * (1 + (speaker_id % 3) * 0.1)  # Slightly different tone per speaker
                    audio = np.sin(2 * np.pi * frequency * t) * 0.5
                    wavfile.write(file_path, sample_rate, (audio * 32767).astype(np.int16))
    
    # Create noise dataset structure mimicking MUSAN format
    noise_dir = base_out / "nhattruongdev__musan-noise" / "noise"
    noise_dir.mkdir(parents=True, exist_ok=True)
    print("Creating MUSAN noise structure...")
    
    # Generate different types of noise
    noise_types = ["white", "pink", "brown"]
    for i, noise_type in enumerate(noise_types):
        filename = f"{noise_type}_noise_{i+1:03d}.wav"
        file_path = noise_dir / filename
        
        # Generate appropriate noise
        sample_rate = 16000
        duration = 1.0
        samples = int(sample_rate * duration)
        
        if noise_type == "white":
            audio = np.random.normal(0, 0.1, samples)
        elif noise_type == "pink":
            # Simple approximation of pink noise
            audio = np.random.normal(0, 0.1, samples)
            # Apply basic low-pass filter
            for i in range(1, len(audio)):
                audio[i] = 0.8 * audio[i] + 0.2 * audio[i-1]
        else:  # brown noise
            # Simple approximation of brown noise
            audio = np.random.normal(0, 0.1, samples)
            # Apply stronger low-pass filter
            for i in range(1, len(audio)):
                audio[i] = 0.95 * audio[i-1] + 0.05 * audio[i]
        
        wavfile.write(file_path, sample_rate, (audio * 32767).astype(np.int16))
    
    # Create a metadata file
    metadata_file = base_out / "metadata.txt"
    with open(metadata_file, "w") as f:
        f.write("Synthetic test data structure:\n")
        f.write(f"- LibriTTS train-clean-100: {datasets['libri-tts-train-clean-100'][0]} speakers\n")
        f.write(f"- LibriTTS train-clean-360: {datasets['libri-tts-train-clean-360'][0]} speakers\n")
        f.write(f"- MUSAN noise: {len(noise_types)} noise types\n")
    
    print(f"Synthetic test data created at: {base_out}")
    print(f"Total files created: {sum([s*c*u for s,c,u in datasets.values()]) + len(noise_types)}")
    return base_out

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create synthetic test data for audio processing")
    parser.add_argument("--output-dir", default="synthetic_test_data", help="Output directory for test data")
    
    args = parser.parse_args()
    create_synthetic_test_data(args.output_dir)