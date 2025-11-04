#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

# ----------------------------
# NumPy compatibility shim
# ----------------------------
import numpy as np  # must be before libs that might import np.int, etc.
# Add back removed aliases if missing (some deps still use them)
if not hasattr(np, "int"):      np.int = int      # type: ignore[attr-defined]
if not hasattr(np, "float"):    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "bool"):     np.bool = bool    # type: ignore[attr-defined]
if not hasattr(np, "complex"):  np.complex = complex  # type: ignore[attr-defined]
if not hasattr(np, "object"):   np.object = object    # type: ignore[attr-defined]

# ----------------------------
# stdlib & third-party imports
# ----------------------------
import argparse
import glob
import json
import os
import random
import sys
import traceback
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import yaml
from tqdm import tqdm

# ✅ Correct miditoolkit imports
from miditoolkit import MidiFile
from miditoolkit.midi.containers import Note, Instrument  # noqa: F401  (type hints)


# ----------------------------
# Heuristics & vocab
# ----------------------------
LEAD_HINTS = (
    "lead", "solo", "guitar", "sax", "trumpet", "clarinet", "flute", "violin", "voice"
)
CHORD_HINTS = (
    "piano", "guitar", "pad", "strings", "organ", "rhodes", "keys", "synth", "choir"
)

PITCH_PAD = 0            # 0 = rest, 1..128 = MIDI pitch+1
NUM_PITCHES = 129

CHROMA_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
CHORDS = ["N"] + [f"{r}:{q}" for r in CHROMA_NAMES for q in ("maj", "min")]
CHORD_TO_ID = {c: i for i, c in enumerate(CHORDS)}
ID_TO_CHORD = {i: c for c, i in CHORD_TO_ID.items()}


def guess_triad(pcs_count: Counter, min_notes: int = 2) -> str:
    """Guess simple maj/min triad (or 'N') from a pitch-class histogram."""
    total = sum(pcs_count.values())
    if total < min_notes:
        return "N"
    best, bestscore = "N", 0.0
    for root in range(12):
        maj = {root, (root + 4) % 12, (root + 7) % 12}
        minr = {root, (root + 3) % 12, (root + 7) % 12}
        smaj = sum(pcs_count[p] for p in maj)
        smin = sum(pcs_count[p] for p in minr)
        if smaj > bestscore:
            best, bestscore = f"{CHROMA_NAMES[root]}:maj", smaj
        if smin > bestscore:
            best, bestscore = f"{CHROMA_NAMES[root]}:min", smin
    # require at least 50% of notes to fit the triad
    return best if bestscore >= 0.5 * total else "N"


def stem_name_ok(s: str, hints) -> bool:
    s = s.lower()
    return any(h in s for h in hints)


def monophony_ratio(notes: List[Note], step: int) -> float:
    """Proportion of timesteps having <= 1 active note (monophonic)."""
    if not notes:
        return 0.0
    tmax = max(n.end for n in notes)
    grid = [0] * ((tmax // step) + 2)
    for n in notes:
        a = max(0, n.start // step)
        b = max(a, n.end // step)
        for t in range(a, b + 1):
            grid[t] += 1
    mono = sum(1 for v in grid if v <= 1)
    return mono / len(grid)


def load_midi_notes(path: str) -> Tuple[List[Note], int]:
    """Load a single-stem MIDI and return (notes, ticks_per_beat)."""
    m = MidiFile(path)
    tpq = m.ticks_per_beat
    notes: List[Note] = []
    for inst in m.instruments:
        notes.extend(inst.notes)
    return notes, tpq


# ----------------------------
# Main data builder
# ----------------------------
def build_from_song(song_dir: str, steps_per_beat: int = 4) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Returns (X_chords, Y_melody, info) with a shared timeline.
      - X_chords: per-step chord id (0..24)
      - Y_melody: per-step pitch token (0=rest, 1..128=pitch+1)
    """
    import numpy as _np  # local alias ok

    meta_path = os.path.join(song_dir, "metadata.yaml")
    midi_dir = os.path.join(song_dir, "MIDI")

    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"metadata.yaml not found in {song_dir}")
    if not os.path.isdir(midi_dir):
        # fallback to any MIDI under the song folder
        cand = glob.glob(os.path.join(song_dir, "**", "*.mid"), recursive=True)
        if not cand:
            raise FileNotFoundError(f"No MIDI found in {song_dir}")
        midi_dir = os.path.dirname(cand[0])

    with open(meta_path, "r") as f:
        meta = yaml.safe_load(f) or {}

    # collect stems
    stems = sorted(glob.glob(os.path.join(midi_dir, "S*.mid")))
    if not stems:
        raise FileNotFoundError(f"No Sxx.mid files in {midi_dir}")

    # choose SOLO stem (heuristic)
    candidates = []
    for p in stems:
        stem_id = os.path.splitext(os.path.basename(p))[0]  # e.g., S02
        stem_meta = (meta.get("stems", {}) or {}).get(stem_id, {}) or {}
        inst_class = (stem_meta.get("inst_class") or stem_meta.get("midi_program_name") or "").lower()
        if stem_meta.get("is_drum"):
            continue
        score = 0.0
        if stem_name_ok(inst_class, LEAD_HINTS):
            score += 2
        if "lead" in inst_class:
            score += 2

        notes, tpq = load_midi_notes(p)
        step = max(1, tpq // steps_per_beat)
        mono = monophony_ratio(notes, step)
        mean_pitch = float(_np.mean([n.pitch for n in notes])) if notes else 0.0
        score += (1.0 if mono >= 0.9 else 0.0) + (mean_pitch / 128.0)

        candidates.append((score, p, notes, tpq, stem_id, inst_class))

    if not candidates:
        raise RuntimeError("No non-drum stems to choose as solo.")
    candidates.sort(reverse=True)
    _solo_score, solo_path, solo_notes, tpq, solo_id, solo_name = candidates[0]

    # accompaniment = everything harmonic (exclude drums + the solo)
    accomp_paths: List[str] = []
    for p in stems:
        if p == solo_path:
            continue
        stem_id = os.path.splitext(os.path.basename(p))[0]
        st = (meta.get("stems", {}) or {}).get(stem_id, {}) or {}
        if st.get("is_drum"):
            continue
        inst_class = (st.get("inst_class") or st.get("midi_program_name") or "").lower()
        if stem_name_ok(inst_class, CHORD_HINTS):
            accomp_paths.append(p)

    # read accompaniment notes
    accomp_notes: List[Note] = []
    for p in accomp_paths:
        nts, _ = load_midi_notes(p)
        accomp_notes.extend(nts)

    if not accomp_notes:
        # fallback: use all non-solo, non-drum stems
        for p in stems:
            if p == solo_path:
                continue
            stem_id = os.path.splitext(os.path.basename(p))[0]
            st = (meta.get("stems", {}) or {}).get(stem_id, {}) or {}
            if st.get("is_drum"):
                continue
            nts, _ = load_midi_notes(p)
            accomp_notes.extend(nts)

    step = max(1, tpq // steps_per_beat)

    # timeline
    tmax = 0
    if solo_notes:
        tmax = max(tmax, max(n.end for n in solo_notes))
    if accomp_notes:
        tmax = max(tmax, max(n.end for n in accomp_notes))
    T = (tmax // step) + 2

    # build chord tokens (input X)
    chords = np.zeros(T, dtype=np.int16)  # id in [0..24]
    pcs_by_t = [Counter() for _ in range(T)]  # histogram per step of active PCs
    for n in accomp_notes:
        a = max(0, n.start // step)
        b = max(a, n.end // step)
        for t in range(a, b + 1):
            pcs_by_t[t][n.pitch % 12] += 1
    for t in range(T):
        chords[t] = CHORD_TO_ID.get(guess_triad(pcs_by_t[t]), 0)

    # build melody tokens (target Y) - monophonic target = highest active note per step
    y = np.zeros(T, dtype=np.int16)  # 0=rest, else pitch+1
    active: Dict[int, List[int]] = defaultdict(list)
    for n in solo_notes:
        a = max(0, n.start // step)
        b = max(a, n.end // step)
        for t in range(a, b + 1):
            active[t].append(n.pitch)
    for t in range(T):
        if active.get(t):
            y[t] = max(active[t]) + 1  # highest pitch

    return chords, y, dict(solo_id=solo_id, solo_name=solo_name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Path containing multiple Slakh-micro song folders")
    ap.add_argument("--out", required=True, help="Output dir (will store pairs.npz + split.json + vocab.json)")
    # accept both styles for convenience
    ap.add_argument("--steps-per-beat", "--steps_per_beat", dest="steps_per_beat", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    song_dirs = sorted([d for d in glob.glob(os.path.join(args.root, "*")) if os.path.isdir(d)])

    X, Y, infos = [], [], []
    for sd in tqdm(song_dirs, desc="Songs"):
        try:
            x, y, info = build_from_song(sd, steps_per_beat=args.steps_per_beat)
            X.append(x)
            Y.append(y)
            infos.append({"song": os.path.basename(sd), **info})
        except Exception as e:
            print(f"[skip] {sd}: {e}")
            traceback.print_exc(limit=2)

    if not X:
        print("No pairs built. Check your --root.")
        sys.exit(1)

    # simple split
    n = len(X)
    idx = list(range(n))
    random.shuffle(idx)
    n_val = max(1, n // 5)
    val_idx = set(idx[:n_val])

    # Save dataset
    np.savez_compressed(
        os.path.join(args.out, "pairs.npz"),
        X=np.array(X, dtype=object),
        Y=np.array(Y, dtype=object),
        infos=np.array(infos, dtype=object),
    )
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({"val_idx": sorted(list(val_idx)), "num": n}, f, indent=2)

    # save vocab for chords
    with open(os.path.join(args.out, "vocab.json"), "w") as f:
        json.dump({"chords": CHORDS, "pitch_pad": PITCH_PAD}, f, indent=2)

    print(f"✔ Wrote dataset with {n} songs to {args.out}")


if __name__ == "__main__":
    main()