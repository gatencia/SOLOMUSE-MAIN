
Real-time **chord/key/tempo listener** + an offline pipeline to **fine-tune a small transformer on MIDI** solos so it can generate solos conditioned on chord progressions.

## Repo Layout

SOLOMUSE-MAIN/
├── chords/
│   └── live_listener.py
│
├── training/
│   ├── data/
│   │   ├── local/        # put your own MIDIs here (untracked)
│   │   └── remote/       # datasets you download (untracked)
│   ├── preprocessing/
│   │   └── prepare_dataset.py
│   ├── finetuning/
│   │   └── finetune_midi_gpt.py
│   ├── configs/
│   │   └── example.env
│   └── README.md
│
├── requirements.txt
├── .gitignore
└── README.md

> If you still have older names like `Chord Detection/` or `data prep/`, rename to the above for consistency.

## Quickstart

### 1 Install Python deps

```bash
pip install -r requirements.txt
```

On Apple Silicon (M1/M2/M3), PyTorch will use MPS (Metal) by default when available.

2) Install the Vamp Plugin Pack (for Chordino)
	•	macOS installer: Vamp Plugin Pack (includes nnls-chroma / Chordino).
	•	After install, set VAMP_PATH to the folder containing the plugins:
	•	Typically one of:
	•	~/Library/Audio/Plug-Ins/Vamp
	•	/Library/Audio/Plug-Ins/Vamp

You can store this in a .env (see below).

3) .env config

Copy the template and edit:

cp training/configs/example.env .env

Open .env and set:
	•	VAMP_PATH → path to your Vamp plugins
	•	AUDIO_INPUT_DEVICE_NAME → your mic or loopback device (e.g., “MacBook Pro Microphone” or “BlackHole 2ch”)
	•	Optional: WINDOW_SECONDS, SERVER_URL

4) Run the live listener

python chords/live_listener.py

You should see a stream like:

🎶 Now playing: Em7 | Key: E minor | Tempo: 110.0 BPM

Tip: If you want to analyze system audio (e.g., a YouTube backing track), install a loopback device (e.g., BlackHole 2ch) and set AUDIO_INPUT_DEVICE_NAME=BlackHole 2ch in .env. To listen yourself while routing, create a macOS Audio MIDI Setup Multi-Output Device that includes both BlackHole and your headphones.

⸻

Training (overview)
	•	Put MIDI files into training/data/local/ (your own) and/or download a public dataset into training/data/remote/.
	•	Run training/preprocessing/prepare_dataset.py to extract chords and solo phrases and convert them to event tokens with miditok.
	•	Run training/finetuning/finetune_midi_gpt.py to LoRA fine-tune a compact GPT-style model on those tokens.
	•	Models and logs are written to training/runs/….

See the full guide in training/README.md￼.

License & Dataset Notes
	•	This repo is MIT for code (unless you change it).
	•	Verify license/usage for any datasets you download (e.g., Lakh MIDI Dataset, MIDIWorld). Use only content you’re permitted to process.

---

### `training/README.md`

```markdown
# Training: Building a Solo Generator from MIDI

This subproject prepares a dataset of **(chords → solo tokens)** pairs and fine-tunes a compact transformer with LoRA to generate solos conditioned on chord progressions.

## 0) Environment

Install repo-wide deps:

```bash
pip install -r ../requirements.txt
```

Create an env file:

cp configs/example.env ../../.env

Then update values in ../../.env as needed (paths, device name, etc.).

1) Data

Put MIDI files here:

training/data/
├── local/     # your MIDIs
└── remote/    # downloaded datasets (e.g., LMD subsets)

Suggested public datasets
	•	Lakh MIDI Dataset (LMD-matched or LMD-aligned): good quality; aligned subset is cleaner for timing.
	•	Start small (a few thousand files) to iterate faster.

⚠️ Check the dataset’s license/terms before use.

2) Prepare the dataset

This script:
	•	Parses MIDI,
	•	Splits into accompaniment vs. lead (heuristics),
	•	Extracts chord progressions and solo phrases,
	•	Tokenizes with miditok (REMI-like) → JSONL of sequence pairs.

Run:

python preprocessing/prepare_dataset.py \
  --in_dir data/local \
  --in_dir data/remote \
  --out_json data/processed/midi_events.jsonl \
  --min_tracks 2 \
  --max_len 2048

Key flags:
	•	--min_tracks 2: skip 1-track MIDIs (often monophonic or percussion-only).
	•	--max_len: truncation length for token sequences.

Output: data/processed/midi_events.jsonl with records like:

{
  "id": "song_000123",
  "chords_tokens": [ ... ],
  "solo_tokens": [ ... ],
  "meta": { "tempo": 110, "key": "E", "scale": "minor" }
}

3) Fine-tune the model (LoRA)

We fine-tune a compact GPT-style model on token sequences (no audio). Default: small decoder LM from 🤗 Transformers.

python finetuning/finetune_midi_gpt.py \
  --dataset_path data/processed/midi_events.jsonl \
  --output_dir runs/midi-gpt-lora \
  --base_model tiny-gpt \
  --epochs 3 \
  --batch_size 8 \
  --lr 2e-4 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05

Notes
	•	--base_model can be a small GPT-like architecture you define in the script (fastest to iterate). You can later swap to a HF model id if you prefer.
	•	On Apple Silicon, PyTorch uses MPS automatically when available.

Artifacts saved to training/runs/midi-gpt-lora/:
	•	adapter_config.json & adapter_model.safetensors (LoRA weights)
	•	tokenizer config & logs.

4) Generate / sanity check

After training, the script prints a few sample generations. You can also add a small generate.py to:
	•	Load LoRA adapters on the base model,
	•	Feed chord tokens,
	•	Decode a solo token sequence,
	•	Convert tokens → MIDI,
	•	Save generated_solo.mid for listening in a DAW.

Tips
	•	Start with small subsets (1–5k MIDI files). Clean your data: remove all-drums, very short, or broken files.
	•	Keep sequence lengths modest (max_len 1024–2048) to avoid OOM and speed up iterations.
	•	Use LoRA for rapid experiments; switch to full fine-tune only if you really need it.

Troubleshooting
	•	torch install: If you hit issues, try pip install --upgrade pip first. On macOS ARM, the default PyPI wheels generally work; MPS is used automatically when available.
	•	miditok tokenization errors: log problematic file paths; skip and continue.

---

### `chords/README.md`

```markdown
# Live Chord/Key/Tempo Listener

Continuously records short windows from your selected input (mic or loopback) and prints:
- **Chord** (Chordino via Vamp)
- **Key** (quick chroma template)
- **Tempo** (librosa beat tracker)
- A few “mood” descriptors (RMS, centroid, ZCR)

## Setup

1) Install deps:
```bash
pip install -r ../requirements.txt
```

	2.	Install Vamp Plugin Pack and set your VAMP_PATH in an .env at repo root (see training/configs/example.env).
	3.	Choose your input device:

	•	AUDIO_INPUT_DEVICE_NAME="MacBook Pro Microphone" → live mic
	•	or install BlackHole 2ch and use AUDIO_INPUT_DEVICE_NAME="BlackHole 2ch" to capture system audio (YouTube/Spotify/etc.). Use a macOS Multi-Output Device if you also want to hear it.

Run

python live_listener.py

You’ll see a live line updating per window like:

🎶 Now playing: Em7 | Key: E minor | Tempo: 110.0 BPM

If you see “No plugin found: nnls-chroma:chordino”, your VAMP_PATH isn’t pointing at the right folder. Typical macOS paths:
	•	~/Library/Audio/Plug-Ins/Vamp
	•	/Library/Audio/Plug-Ins/Vamp

