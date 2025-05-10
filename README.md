# Bayan

A machine learning project focused on audio processing and analysis, built with TensorFlow and related libraries.

## Overview

This project appears to be a machine learning system for audio processing, utilizing TensorFlow and various audio processing libraries. The project consists of two main components:

- **Soroush Module**: A voice activity detection and speaker identification system
- **Al-Bab Module**: A comprehensive voice activity detection (VAD) system that includes:
  - Real-time audio stream processing
  - Multiple VAD model implementations
  - Audio preprocessing and feature extraction
  - Integration with various audio input sources
  - Performance optimization for different hardware configurations

## Prerequisites

- Python 3.10
- pip (Python package installer)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/nokhte/bayan.git
cd bayan
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Project Structure

```
bayan/
├── src/
│   ├── soroush/     # Soroush module
│   └── al_bab/      # Voice Activity Detection (VAD) 
│       ├── models/  # VAD model implementations
│       ├── utils/   # Audio processing utilities
│       ├── config/  # Configuration files
│
├── test/ # Unit Tests
│
├── requirements.txt  # Project dependencies
└── LICENSE          # Project license
```

## Data Management

### Kaggle Setup

Before downloading data from Kaggle, you need to set up your credentials:

1. Create a Kaggle account if you haven't already at [kaggle.com](https://www.kaggle.com)

2. Get your API credentials:
   - Go to your Kaggle account settings
   - Scroll to the "API" section
   - Click "Create New API Token"
   - This will download a `kaggle.json` file

3. Set up your credentials:
   ```bash
   # Create .kaggle directory in your home folder
   mkdir ~/.kaggle
   
   # Move the downloaded kaggle.json to the .kaggle directory
   mv ~/Downloads/kaggle.json ~/.kaggle/
   
   # Set appropriate permissions
   chmod 600 ~/.kaggle/kaggle.json
   ```

4. Install the Kaggle API:
   ```bash
   pip install kaggle
   ```

### Downloading Data

The project includes functionality to download and process audio data. Data can be downloaded using the following methods:

1. Using the command-line interface:
```bash
python -m src.soroush.data.downloader --source [SOURCE] --output [OUTPUT_DIR]
```

2. Programmatically:
```python
from src.soroush.data.downloader import download_data

# Download data from a specific source
download_data(source="your_source", output_dir="path/to/output")
```

### Data Sources

The system supports multiple data sources:
- Local audio files
- Remote audio repositories
- Streaming audio sources

### Data Processing

After downloading, data can be processed using the built-in processing pipeline:
```python
from src.soroush.data.processor import process_audio

# Process downloaded audio files
process_audio(input_dir="path/to/input", output_dir="path/to/output")
```

### Data Storage

- Downloaded data is stored in the `data/` directory by default
- Processed data is organized in subdirectories based on the source and processing parameters
- The system maintains metadata about downloaded and processed files

## Development

[Add development setup and contribution guidelines here]

## Testing

The project uses pytest for testing. To run tests:

```bash
pytest
```

## Acknowledgments

The Soroush model is built on top of the work done by Douglas Coimbra de Andrade's [SpeechIdentity](https://github.com/douglas125/SpeechIdentity).
