# SoloMuse Training Pipeline

Train transformer models to generate musical solos conditioned on chord progressions.

## Overview

This pipeline converts MIDI files into tokenized sequences representing chords and melodies, then trains a GPT-2 model to generate new solos that follow given chord progressions.

## Dataset

### Quick Start with Examples
The `01.data/Remote/` directory contains a few example MIDI files to get started quickly.

### Full Slakh2100 Dataset
For full training, download the complete Slakh2100 dataset:
- **Download**: [Slakh2100 on Zenodo](https://zenodo.org/records/4599666) (~105GB)
- **Extract to**: `01.data/Remote/slakh2100/`
- **Contents**: 2100 multi-track songs with separated MIDI files

See `01.data/dataset.md` for detailed information about the Slakh2100 dataset.

## Directory Structure

```
Training/
├── 01.data/                     # MIDI datasets
│   ├── Local/                   # Local MIDI files
│   └── Remote/                  # Downloaded datasets
├── 02.preprocessing/            # Data preparation
│   └── prepare_dataset.py       # MIDI tokenization
├── 03.finetuning/              # Model training
│   └── finetune_midi_gpt.py     # Training script
└── configs/                     # Configuration files
```

## Quick Start

### 1. Prepare Data
```bash
mkdir -p 01.data/Local/midi_files
# Add your .mid files to Local/midi_files/
```

### 2. Tokenize MIDI
```bash
cd 02.preprocessing
python prepare_dataset.py
```

### 3. Train Model
```bash
cd 03.finetuning
python finetune_midi_gpt.py \
    --midi_root ../01.data/Local/midi_files \
    --out_dir ./runs/my_model \
    --epochs 10
```

### 4. Generate Demo
```bash
python finetune_midi_gpt.py \
    --midi_root ../01.data/Local/midi_files \
    --generate_demo \
    --out_dir ./runs/my_model
```

## Tokenization

Creates a vocabulary combining musical elements and harmonic context:

- **Notes**: `NOTE_ON_60`, `NOTE_ON_64`, etc.
- **Durations**: `DUR_1`, `DUR_2`, `DUR_4` (16th note steps)
- **Timing**: `TIME_SHIFT_1`, `TIME_SHIFT_2`, etc.
- **Velocity**: `VEL_64`, `VEL_96`, etc.
- **Chords**: `CHORD_ROOT_C`, `CHORD_QUAL_maj`, etc.

Sequence format:
```
<BOS> CHORD_ROOT_C CHORD_QUAL_maj <SEP> VEL_80 NOTE_ON_60 DUR_2 <EOS>
```

## Configuration

### Data Processing
```python
PPQ = 480                    # Pulses per quarter note
GRID = 4                     # Steps per beat (16th notes)
SEQ_LEN = 512               # Training sequence length
```

### Model Architecture
```python
n_layer = 8                  # Transformer layers
n_head = 8                   # Attention heads
n_embd = 512                # Embedding dimension
block_size = 1024           # Context window
```

### Training
```python
batch_size = 8              # Training batch size
learning_rate = 3e-4        # Adam learning rate
epochs = 10                 # Training epochs
```

## Data Preparation

### MIDI Sources
- Place files in `01.data/Local/midi_files/`
- Supports `.mid` and `.midi` files
- Multi-track files automatically processed

### Track Selection
The preprocessing automatically picks melody tracks by:
1. Preferring higher-register instruments
2. Minimizing polyphony
3. Ensuring sufficient note density

## Training Process

The `finetune_midi_gpt.py` script handles:

1. **Dataset Loading**: Tokenized MIDI sequences
2. **Model Setup**: GPT-2 with custom vocabulary
3. **Training**: Autoregressive language modeling
4. **Checkpointing**: Regular model saves
5. **Demo Generation**: Sample output with chord conditioning

### Hardware Requirements
- **Minimum**: 8GB RAM, modern CPU
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM
- **Training Time**: 2-4 hours (1K files), 8-12 hours (10K files)

## Generation

Generate solos from chord progressions:

```python
chords = ["C:maj", "F:maj", "G:maj", "C:maj"]
generated_ids = generate_from_chords(
    model=model,
    chord_roots=chords,
    n_events=256,
    temperature=1.0
)
```

Parameters:
- **Temperature**: Controls randomness (0.5-1.5)
- **Length**: Number of events to generate
- **Conditioning**: Input chord progression

## Common Issues

**No MIDI files found**: Check file paths and extensions

**Tokenization fails**: Verify MIDI file compatibility, reduce sequence length

**Training crashes**: Reduce batch size, check GPU memory

**Poor quality**: Increase dataset size, train longer, adjust temperature

## Advanced Usage

### Multi-GPU Training
```bash
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    finetune_midi_gpt.py
```

### Resume Training
```bash
python finetune_midi_gpt.py \
    --resume_from ./runs/checkpoint-1000
```