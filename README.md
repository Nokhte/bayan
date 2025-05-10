# Bayan

A machine learning project focused on audio processing and analysis, built with TensorFlow and related libraries.

## Overview

This project appears to be a machine learning system for audio processing, utilizing TensorFlow and various audio processing libraries. The project is organized into multiple components, including modules for Soroush and Al-Bab.

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
pip install -r requirements.txt
```

## Project Structure

```
bayan/
├── src/
│   ├── soroush/     # Soroush module
│   └── al-bab/      # Al-Bab module
├── requirements.txt  # Project dependencies
└── LICENSE          # Project license
```

## Dependencies

The project relies on the following main dependencies:
- TensorFlow (2.19.0)
- NumPy (1.26.4)
- Keras (3.5.0)
- TensorBoard (2.19.0)
- Pandas (2.0.3)
- Librosa (0.11.0)
- TensorFlow Addons (0.21.0)
- Matplotlib
- SciPy
- PyTest
- TensorFlow-Metal (for Apple Silicon support)
- WebRTC VAD

## Usage

[Add specific usage instructions here]

## Development

[Add development setup and contribution guidelines here]

## Testing

The project uses pytest for testing. To run tests:

```bash
pytest
```

## Contributing

[Add contribution guidelines here]

## Contact

[Add contact information here]

## Acknowledgments

The Soroush model is built on top of the work done by Douglas Coimbra de Andrade's [SpeechIdentity](https://github.com/douglas125/SpeechIdentity).