#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoloMuse sample generation with theory-aware priors.

- Loads a trained checkpoint (best.pt or last.pt).
- Takes a chord sequence and generates a monophonic melody token for each step.
- Applies Scaffold() prior as a log-bias during sampling.

Usage
-----
python sample_generate.py \
  --ckpt Training/03.finetuning/outputs/minitok/best.pt \
  --vocab Training/datasets/slakh_micro/vocab.json \
  --chords "C:maj,D:min,G:7,C:maj" \
  --steps-per-beat 4 \
  --tempo 120 \
  --bars 4 \
  --prior-lambda 1.2 \
  --out sample.mid
"""
from __future__ import annotations

import argparse, json, os, math, random
from typing import List

import numpy as np
import torch
import torch.nn as nn
from miditoolkit import MidiFile, Instrument, Note

# local import
try:
    from scaffold import Scaffold
except Exception:
    from Training03finetuning.scaffold import Scaffold  # if installed

# tiny encoder (must match train dims)
class TinyTransformer(nn.Module):
    def __init__(self, n_chords, n_pitches, d_model=128, nhead=4, nlayers=4):
        super().__init__()
        self.chord_emb = nn.Embedding(n_chords, d_model)
        self.pos_emb   = nn.Embedding(4096, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.encoder   = nn.TransformerEncoder(encoder_layer, nlayers)
        self.head      = nn.Linear(d_model, n_pitches)

    def forward(self, x):
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.chord_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h) # (B,T,C)

def softmax_top_p(logits: torch.Tensor, top_p: float, temperature: float) -> torch.Tensor:
    # logits: (C,)
    logits = logits / max(temperature, 1e-8)
    probs = torch.softmax(logits, dim=-1)
    if top_p <= 0 or top_p >= 1.0:
        return torch.multinomial(probs, 1).item()
    # nucleus truncation
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    keep = cumsum <= top_p
    # ensure at least 1 token
    if not torch.any(keep):
        keep[0] = True
    truncated_probs = sorted_probs[keep]
    truncated_probs = truncated_probs / truncated_probs.sum()
    choice = torch.multinomial(truncated_probs, 1).item()
    return int(sorted_idx[choice].item())

def chord_list_to_ids(chords: List[str], chord_to_id: dict) -> np.ndarray:
    ids = []
    for c in chords:
        c = c.strip()
        ids.append(int(chord_to_id.get(c, 0)))
    return np.array(ids, dtype=np.int64)

def write_midi_from_tokens(tokens: np.ndarray, out_path: str, tempo_bpm: float, steps_per_beat: int, program: int = 0):
    # tokens: length T, 0=rest else pitch+1
    mf = MidiFile()
    mf.ticks_per_beat = 480
    inst = Instrument(program=program, is_drum=False, name="SoloMuse")
    mf.instruments.append(inst)

    step_ticks = int(mf.ticks_per_beat / steps_per_beat)
    # accumulate durations of same pitch (non-zero)
    t = 0
    i = 0
    while i < len(tokens):
        tok = int(tokens[i])
        if tok <= 0:
            # rest advances time by one step
            t += step_ticks
            i += 1
            continue
        pitch = tok - 1
        # extend while same token holds
        j = i
        while j < len(tokens) and int(tokens[j]) == tok:
            j += 1
        start = t
        end   = t + (j - i) * step_ticks
        inst.notes.append(Note(velocity=90, pitch=pitch, start=start, end=end))
        t = end
        i = j

    # tempo
    from miditoolkit.midi.containers import TempoChange
    mf.tempo_changes.append(TempoChange(tempo=int(60_000_000 / max(tempo_bpm, 1e-3)), time=0))
    mf.dump(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out",   default="sample.mid")

    # chord input
    ap.add_argument("--chords", default="", help="Comma-separated chord labels like 'C:maj,D:min,G:7,C:maj'")
    ap.add_argument("--chords-file", default="", help="Text file with one chord label per line")
    ap.add_argument("--bars", type=int, default=4, help="If no chords given, repeat I-vi-ii-V for this many bars")
    ap.add_argument("--steps-per-beat", type=int, default=4)
    ap.add_argument("--tempo", type=float, default=120.0)

    # sampling
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--prior-lambda", type=float, default=1.0, help="scale for adding Scaffold log-bias")
    ap.add_argument("--pitch-min", type=int, default=48)
    ap.add_argument("--pitch-max", type=int, default=84)

    args = ap.parse_args()

    # load vocab
    with open(args.vocab, "r") as f:
        voc = json.load(f)
    chords_list = voc["chords"]
    id_to_chord = {i:c for i,c in enumerate(chords_list)}
    chord_to_id = {c:i for i,c in enumerate(chords_list)}
    n_chords = len(chords_list)
    n_pitches = 129

    # build chord ids
    if args.chords:
        chords = [c.strip() for c in args.chords.split(",") if c.strip()]
    elif args.chords_file and os.path.isfile(args.chords_file):
        with open(args.chords_file, "r") as f:
            chords = [ln.strip() for ln in f if ln.strip()]
    else:
        # default: 4-bar I-vi-ii-V in C major (one chord per beat)
        base = ["C:maj","A:min","D:min","G:7"] * args.bars
        chords = base
    X = torch.from_numpy(chord_list_to_ids(chords, chord_to_id)).unsqueeze(0).long()  # (1,T)

    # load model
    ck = torch.load(args.ckpt, map_location="cpu")
    h = ck.get("_hparams", {})
    d_model = h.get("d_model", 128); nhead = h.get("nhead", 4); nlayers = h.get("nlayers", 4)
    model = TinyTransformer(n_chords, n_pitches, d_model=d_model, nhead=nhead, nlayers=nlayers)
    model.load_state_dict(ck["model"]); model.eval()

    # theory prior
    scaf = Scaffold(steps_per_beat=args.steps_per_beat, pitch_min=args.pitch_min, pitch_max=args.pitch_max, y_rest_token=0)

    with torch.no_grad():
        logits = model(X)  # (1,T,129)
        T = logits.shape[1]
        out = []
        for t in range(T):
            raw = logits[0, t].clone()
            # add log-bias based on current chord label
            chord_label = id_to_chord[int(X[0, t].item())]
            bias = scaf.prior_logits(chord_label=chord_label, t=t)
            # pad bias if needed
            if bias.shape[0] < raw.shape[0]:
                b = torch.full_like(raw, 0.0)
                import numpy as _np
                b[:bias.shape[0]] = torch.from_numpy(bias).float()
                bias_t = b
            else:
                import numpy as _np
                bias_t = torch.from_numpy(bias).float()[: raw.shape[0]]
            raw = raw + float(args.prior_lambda) * bias_t
            tok = softmax_top_p(raw, top_p=args.top_p, temperature=args.temperature)
            out.append(tok)
        tokens = np.array(out, dtype=np.int64)

    write_midi_from_tokens(tokens, args.out, tempo_bpm=args.tempo, steps_per_beat=args.steps_per_beat)
    print(f"✔ Wrote {args.out}  (T={len(tokens)} steps, tempo={args.tempo} BPM)")

if __name__ == "__main__":
    main()