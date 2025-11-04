# Dataset Information

This directory contains datasets used for training SoloMuse's AI music generation models.

## Slakh2100 Dataset

I primarily use the Synthesized Lakh (Slakh) Dataset for training. Slakh2100 is a dataset of multi-track audio and aligned MIDI designed for music source separation and multi-instrument automatic transcription.

### Why Slakh2100?

Slakh2100 is ideal for SoloMuse because:

- **Multi-track MIDI**: Contains separated instrument tracks, making it easy to isolate melody lines for solo generation
- **Professional synthesis**: Uses high-quality sample-based virtual instruments rather than basic MIDI playback
- **Harmonic diversity**: 2100 tracks with varied chord progressions and musical styles
- **Clean data**: MIDI files are properly aligned with audio and contain accurate timing information
- **Instrument variety**: 187 instrument patches across 34 classes provide diverse timbral training data

### Dataset Details

- **Size**: 2100 tracks, 145 hours of music, ~105GB download
- **Format**: FLAC audio (44.1kHz, 16-bit) with aligned MIDI files
- **Structure**: Each track contains mixture audio plus individual source stems
- **Guaranteed instruments**: Every mix includes Piano, Guitar, Drums, and Bass
- **Sources per mix**: Minimum 4 instruments, variable maximum

### Directory Structure

```
TrackXXXXX/
├── mix.flac              # Full mixture
├── stems/
│   ├── S00.flac         # Individual instrument stems
│   ├── S01.flac
│   └── ...
├── MIDI/
│   ├── S00.mid          # MIDI for each stem
│   ├── S01.mid
│   └── ...
└── metadata.yaml        # Track information
```

### Data Splits

We use Slakh2100-redux which removes duplicate MIDI files found in the original release. The `omitted` directory contains tracks with duplicate MIDI that should be avoided for training to prevent data leakage.

## Usage in SoloMuse

1. **Melody extraction**: We identify the highest-register monophonic instrument as the melody line
2. **Harmony analysis**: Other tracks are analyzed for chord progressions using music21
3. **Tokenization**: MIDI data is converted to my custom token vocabulary
4. **Training**: Chord progressions condition the model to generate appropriate melody lines

## Getting the Dataset

Download Slakh2100-redux from [Zenodo](https://zenodo.org/record/4603810). Place extracted tracks in `Remote/slakh2100/` or link individual MIDI files to `Local/midi_files/` for processing.

## Other Datasets

You can also use:

- **Lakh MIDI Dataset**: Original MIDI collection (larger but lower quality)
- **MAESTRO**: Classical piano performances (high quality but limited scope)
- **Personal MIDI collections**: Your own MIDI files work fine

Place any MIDI files in `Local/midi_files/` and the preprocessing pipeline will handle them.

## Citation

If you use Slakh2100 in your work:

```bibtex
@inproceedings{manilow2019cutting,
  title={Cutting Music Source Separation Some {Slakh}: A Dataset to Study the Impact of Training Data Quality and Quantity},
  author={Manilow, Ethan and Wichern, Gordon and Seetharaman, Prem and Le Roux, Jonathan},
  booktitle={Proc. IEEE Workshop on Applications of Signal Processing to Audio and Acoustics (WASPAA)},
  year={2019},
  organization={IEEE}
}
```