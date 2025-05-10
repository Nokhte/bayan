# Bayan

A family of voice-driven models for audio processing, built with TensorFlow and related libraries. Developed for use within the Nokhte ecosystem, including the Nokhte mobile app.

## Overview

Bayan (بیان) means "speech" or "expression" in Persian, reflecting the project's focus on understanding and interpreting human voice.

This repository houses a collection of machine learning systems for real-time audio analysis and voice understanding. It is composed of two main modules:

Soroush (سروش): A voice activity detection (VAD) and speaker identification system. In Persian mythology, Soroush is an angel of communication—aptly named for a module that listens and distinguishes who is speaking.

Al-Bab (الباب): A modular, high-performance VAD engine. Al-Bab, meaning "the gate" in Arabic, symbolizes its function as the gateway to downstream voice intelligence by deciding when speech occurs.

Together, these models form the audio intelligence layer for the Nokhte mobile app, enabling voice-triggered interaction and context-aware voice features. While tightly integrated with the app, the models are modular and can be reused across other systems.


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

## Development

[Add development setup and contribution guidelines here]

## Testing

The project uses pytest for testing. To run tests:

```bash
pytest
```

## Acknowledgments

The Soroush model is built on top of the work done by Douglas Coimbra de Andrade's [SpeechIdentity](https://github.com/douglas125/SpeechIdentity).