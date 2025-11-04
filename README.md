# SoloMuse - AI Music Companion

An AI system that listens to live audio, detects chords, and generates musical solos in real-time.

## What it does

SoloMuse combines real-time chord detection with a trained transformer model to generate contextually appropriate musical solos. Play some chords, and it'll create a solo that fits harmonically.

## Features

- Real-time chord detection using Vamp Chordino plugin
- Key and tempo detection with stabilization
- GPT-2 based model trained on MIDI data
- Chord-conditioned solo generation
- Works with audio interfaces and loopback devices

## Setup

### Requirements
- Python 3.8+
- Audio loopback device (BlackHole recommended for macOS)
- Vamp Plugin Pack

### Installation

```bash
git clone https://github.com/yourusername/SoloMuse.git
cd SoloMuse
pip install -r requirements.txt
```

### Audio Setup (macOS)
```bash
brew install blackhole-2ch
```

Download and install the [Vamp Plugin Pack](https://www.vamp-plugins.org/download.html) to `~/Library/Audio/Plug-Ins/Vamp/`.

### Configuration
```bash
cp .env.example .env
# Edit .env with your audio device settings
```

## Usage

Start the chord listener:
```bash
cd Chords
python live_listener.py
```

Output shows detected chords, key, tempo, and audio features:
```
Now playing: Cmaj7   | Key: C  major | Tempo: 120.0 BPM | Centroid:  2451 | RMS: 0.045
```

## Project Structure

```
SoloMuse/
├── Chords/               # Real-time audio analysis
│   └── live_listener.py  # Main detection script
└── Training/             # Model training pipeline
    ├── 01.data/          # MIDI datasets
    ├── 02.preprocessing/ # Tokenization
    └── 03.finetuning/    # Model training
```

## Configuration

Key settings in `.env`:

```env
INPUT_DEVICE_NAME=BlackHole
SAMPLE_RATE=44100
DURATION=2.0
VAMP_PLUGIN_DIR=/Users/username/Library/Audio/Plug-Ins/Vamp
```

## Technical Details

### Audio Pipeline
1. Capture audio via sounddevice
2. Analyze with librosa and Vamp Chordino
3. Stabilize tempo/key/chord detection
4. Extract spectral features

### AI Pipeline
1. Tokenize MIDI files with custom REMI vocabulary
2. Train GPT-2 on chord-melody sequences
3. Generate solos conditioned on chord progressions

See `Training/README.md` for model training details.
