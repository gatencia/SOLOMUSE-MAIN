# prepare_dataset.py
# Minimal MIDI → tokens pipeline for chord-conditioned solo generation.
# pip install pretty_midi music21 miditoolkit tqdm

import os, json, math
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

import pretty_midi
import music21
import miditoolkit

# ====== Config ======
MIDI_DIR = "midi_files"         # put your .mid/.midi here
OUT_JSONL = "data/train.jsonl"  # tokenized sequences
PPQ = 480                       # target pulses per quarter
GRID = 4                        # 4 steps per beat (16th notes if 4/4)
SEQ_LEN = 512                   # training sequence length
MIN_NOTES_MELODY = 16

# ====== Vocab (tiny REMI-ish) ======
# Build programmatically so it's easy to extend
CHORD_ROOTS = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORD_QUALS = ['maj','min','dim','aug','7','m7','maj7','sus2','sus4','dim7','m7b5']
TOKENS = ["<PAD>","<BOS>","<EOS>","<BAR>","<SEP>"]

# Note/Duration/Shift ranges
for p in range(128): TOKENS.append(f"NOTE_ON_{p}")
for d in [1,2,3,4,6,8,12,16]: TOKENS.append(f"DUR_{d}")         # in 16th-note steps
for ts in [1,2,3,4,6,8,12,16]: TOKENS.append(f"TIME_SHIFT_{ts}")
for v in [16,32,48,64,80,96,112,127]: TOKENS.append(f"VEL_{v}")
for r in CHORD_ROOTS: TOKENS.append(f"CHORD_ROOT_{r}")
for q in CHORD_QUALS: TOKENS.append(f"CHORD_QUAL_{q}")

stoi = {t:i for i,t in enumerate(TOKENS)}
itos = {i:t for t,i in stoi.items()}

def quantize_time(ticks: int, ppq: int, grid: int) -> int:
    """Quantize ticks to grid sub-beats (e.g., 16th notes)."""
    step = ppq // grid
    return round(ticks / step) * step

def pick_melody_track(pm: pretty_midi.PrettyMIDI) -> int:
    """Choose a monophonic, high-register instrument as melody."""
    # Heuristic: maximize avg pitch, minimize chord density
    best_idx, best_score = 0, -1e9
    for i, inst in enumerate(pm.instruments):
        if inst.is_drum or len(inst.notes) < MIN_NOTES_MELODY:
            continue
        pitches = [n.pitch for n in inst.notes]
        if not pitches: 
            continue
        # “Polyphony” proxy: fraction of overlapping notes
        overlaps = 0
        sorted_notes = sorted(inst.notes, key=lambda n: (n.start, n.end))
        for a, b in zip(sorted_notes, sorted_notes[1:]):
            if b.start < a.end: overlaps += 1
        poly = overlaps / max(1, len(sorted_notes))
        score = (sum(pitches)/len(pitches)) - 60*poly
        if score > best_score:
            best_score, best_idx = score, i
    return best_idx

def chordify_harmony(midi_path: str, target_ppq: int, grid: int):
    """
    Use music21 to chordify non-melody parts and return a bar/beat grid of chords:
    [(step_index, root, quality), ...]
    """
    s = music21.converter.parse(midi_path)
    s.parts.stream()  # ensure parts
    # Chordify everything, then we’ll still use melody from pretty_midi
    ch = s.chordify()
    key = ch.analyze('key')
    # Reduce to a grid, 4 steps/beat
    ql = []
    # Use measures → beats
    for m in ch.makeMeasures():
        for offset in range(0, int(m.highestTime*grid+0.5)):
            t = (m.offset + offset/grid)
            elems = ch.flat.getElementsByOffset(t, mustBeginInSpan=False, includeEndBoundary=False)
            chords = [e for e in elems if isinstance(e, music21.chord.Chord)]
            if not chords:
                continue
            c = chords[0]
            root = c.root().name if c.root() else key.tonic.name
            qual = c.quality if hasattr(c, 'quality') else 'maj'
            # normalize a few names
            qual = {'major':'maj','minor':'min','augmented':'aug','diminished':'dim',
                    'dominant':'7','half-diminished':'m7b5','major-seventh':'maj7',
                    'minor-seventh':'m7'}.get(qual, qual)
            if root in CHORD_ROOTS and qual in CHORD_QUALS:
                ql.append((int(round(t*grid)), root, qual))
    # deduplicate consecutive equal chords
    dedup = []
    last = None
    for step, r, q in ql:
        cur = (r,q)
        if cur != last:
            dedup.append((step, r, q))
            last = cur
    return dedup, key

def eventize_melody(inst: pretty_midi.Instrument, tempo, ppq: int, grid: int) -> List[str]:
    """Convert melody notes to NOTE/VEL/DUR/TIME_SHIFT tokens on a 16th grid."""
    step = ppq // grid
    # Build a step-indexed list of events
    events = []
    cur_step = 0
    for n in sorted(inst.notes, key=lambda x: x.start):
        start = int(round(n.start * ppq * (tempo/60.0)))  # map sec → ticks via tempo
        dur = max(step, int(round((n.end - n.start) * ppq * (tempo/60.0))))
        qstart = quantize_time(start, ppq, grid)
        qdur = max(1, round(dur / step))
        # time shift
        delta_steps = (qstart - (cur_step*step)) // step
        while delta_steps > 0:
            shift = min(16, delta_steps)
            events.append(f"TIME_SHIFT_{shift}")
            delta_steps -= shift
        # note events
        events.append(f"VEL_{min([16,32,48,64,80,96,112,127], key=lambda v: abs(v-n.velocity))}")
        events.append(f"NOTE_ON_{n.pitch}")
        events.append(f"DUR_{min([1,2,3,4,6,8,12,16], key=lambda d: abs(d-qdur))}")
        cur_step = qstart//step + qdur
    return events

def interleave_chords(chords: List[Tuple[int,str,str]], grid: int) -> List[str]:
    """Turn chord timeline into tokens."""
    tokens = []
    last_step = 0
    for step, root, qual in chords:
        delta = step - last_step
        while delta > 0:
            shift = min(16, delta)
            tokens.append(f"TIME_SHIFT_{shift}")
            delta -= shift
        tokens.append(f"CHORD_ROOT_{root}")
        tokens.append(f"CHORD_QUAL_{qual}")
        last_step = step
    return tokens

def process_file(path: Path) -> List[int]:
    pm = pretty_midi.PrettyMIDI(str(path))
    # use first tempo estimate
    tempi, _ = pm.get_tempo_changes()
    tempo = float(tempi[0]) if len(tempi) else 120.0

    # pick melody instrument
    m_idx = pick_melody_track(pm)
    melody_inst = pm.instruments[m_idx]

    # chordify harmony with music21
    chords, _ = chordify_harmony(str(path), PPQ, GRID)

    chord_tokens = interleave_chords(chords, GRID)
    mel_tokens = eventize_melody(melody_inst, tempo, PPQ, GRID)

    seq = ["<BOS>"] + chord_tokens + ["<SEP>"] + mel_tokens + ["<EOS>"]
    # map to ids & truncate into training windows
    ids = [stoi[t] for t in seq if t in stoi]
    return ids

def main():
    os.makedirs(Path(OUT_JSONL).parent, exist_ok=True)
    paths = [p for p in Path(MIDI_DIR).rglob("*.mid")] + [p for p in Path(MIDI_DIR).rglob("*.midi")]
    with open(OUT_JSONL, "w") as f:
        for p in tqdm(paths, desc="Tokenizing"):
            try:
                ids = process_file(p)
                # chop into fixed windows for LM
                for i in range(0, max(0, len(ids)-SEQ_LEN), SEQ_LEN):
                    window = ids[i:i+SEQ_LEN]
                    if len(window) > 32:
                        f.write(json.dumps({"tokens": window}) + "\n")
            except Exception as e:
                # skip bad files
                continue

if __name__ == "__main__":
    main()