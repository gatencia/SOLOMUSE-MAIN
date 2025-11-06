#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoloMuse: tiny chord->melody Transformer (with optional theory-aware loss weighting).

Inputs
------
- --data  : path to pairs.npz (X chords, Y pitch tokens)
- --split : path to split.json ({"val_idx":[...], "num":N})
- --vocab : path to vocab.json ({"chords":[...], "pitch_pad":0}) [recommended]

Targets
-------
- Predicts pitch token per timestep (0=rest, 1..128=pitch+1).
- Pads labels with -100 (ignore_index). Chords pad with 0 ("N").

Features
--------
- Transposition augmentation (random semitone shift on-the-fly).
- Optional theory-weighted loss using Training/03.finetuning/scaffold.py
  (emphasizes chord/guide tones on strong beats).
- Simple accuracy metric + chord-tone hit rate on predictions.

Saves
-----
- {out}/best.pt  : best val loss
- {out}/last.pt  : most recent checkpoint

"""
from __future__ import annotations

import argparse, json, math, os, time, random
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Optional: theory prior for loss weighting and metrics
try:
    from scaffold import Scaffold, CHORD_PCS, _parse_chord_label  # sibling file
except Exception:
    Scaffold = None
    CHORD_PCS = None
    _parse_chord_label = None

# -----------------------------
# Model
# -----------------------------
class TinyTransformer(nn.Module):
    def __init__(self, n_chords: int, n_pitches: int, d_model=128, nhead=4, nlayers=4, max_len=4096):
        super().__init__()
        self.chord_emb = nn.Embedding(n_chords, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.head    = nn.Linear(d_model, n_pitches)

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        # x: (B, T) chord ids
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.chord_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h)  # (B, T, n_pitches)

# -----------------------------
# Data
# -----------------------------
class PairDataset(Dataset):
    def __init__(self, X: List[np.ndarray], Y: List[np.ndarray], indices: List[int]):
        self.X = X
        self.Y = Y
        self.idx = indices

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i: int) -> Tuple[np.ndarray, np.ndarray]:
        k = self.idx[i]
        return self.X[k].astype(np.int64, copy=False), self.Y[k].astype(np.int64, copy=False)

def collate_pad(batch):
    # Pads chord with 0 ("N"), labels with -100 (ignore)
    max_len = max(len(x) for x, _ in batch)
    Xb = np.full((len(batch), max_len), 0, dtype=np.int64)
    Yb = np.full((len(batch), max_len), -100, dtype=np.int64)
    for i, (x, y) in enumerate(batch):
        L = min(len(x), max_len)
        Xb[i, :L] = x[:L]
        Yb[i, :L] = y[:L]
    return torch.from_numpy(Xb), torch.from_numpy(Yb)

# -----------------------------
# Utils
# -----------------------------
def load_npz(path: str):
    arr = np.load(path, allow_pickle=True)
    X = list(arr["X"])
    Y = list(arr["Y"])
    return X, Y

def load_split(path: str, N: int):
    with open(path, "r") as f:
        js = json.load(f)
    va = set(js.get("val_idx", []))
    all_idx = list(range(N))
    tr_idx = [i for i in all_idx if i not in va]
    va_idx = sorted(list(va))
    return tr_idx, va_idx

def infer_n_chords(X: List[np.ndarray]) -> int:
    mx = 0
    for x in X:
        if len(x):
            mx = max(mx, int(np.max(x)))
    return int(mx + 1)

def choose_device(name: str):
    name = (name or "cpu").lower()
    if name == "cuda" and torch.cuda.is_available(): return torch.device("cuda")
    if name == "mps" and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits: (B,T,C), labels: (B,T) with -100 as ignore
    with torch.no_grad():
        pred = torch.argmax(logits, dim=-1)
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        correct = (pred[mask] == labels[mask]).float().mean().item()
        return float(correct)

def build_vocab_maps(chords_list: List[str]):
    id_to_chord = {i: c for i, c in enumerate(chords_list)}
    chord_to_id = {c: i for i, c in enumerate(chords_list)}
    return id_to_chord, chord_to_id

def transpose_chord_ids(x: torch.Tensor, id_to_chord: Dict[int,str], chord_to_id: Dict[str,int], shift: int) -> torch.Tensor:
    if shift % 12 == 0: 
        return x
    x_np = x.cpu().numpy()
    out = x_np.copy()
    for b in range(x_np.shape[0]):
        for t in range(x_np.shape[1]):
            cid = int(x_np[b, t])
            if cid == 0:  # "N"
                continue
            lbl = id_to_chord.get(cid, "N")
            if lbl == "N":
                continue
            root, qual = lbl.split(":")
            # map root to index 0..11 using same spelling as vocab
            CHROMA = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            try:
                ridx = CHROMA.index(root)
            except ValueError:
                # try to normalize flats
                flats = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
                ridx = CHROMA.index(flats.get(root, root))
            nr = CHROMA[(ridx + shift) % 12]
            nlbl = f"{nr}:{qual}"
            out[b, t] = chord_to_id.get(nlbl, cid)
    return torch.from_numpy(out).to(x.device)

def transpose_pitch_tokens(y: torch.Tensor, shift: int) -> torch.Tensor:
    if shift % 12 == 0:
        return y
    y_np = y.cpu().numpy()
    out = y_np.copy()
    # 0=rest, else pitch+1 in 1..128
    idx = y_np > 0
    out[idx] = np.clip(out[idx] + shift, 1, 128)
    return torch.from_numpy(out).to(y.device)

def chord_tone_hit_rate(logits: torch.Tensor, X: torch.Tensor, id_to_chord: Dict[int,str]) -> float:
    """Fraction of predicted non-rest notes that are chord tones (ignores beat position)."""
    if CHORD_PCS is None or _parse_chord_label is None:
        return 0.0
    pred = torch.argmax(logits, dim=-1)  # (B,T)
    B, T = pred.shape
    total, hits = 0, 0
    for b in range(B):
        for t in range(T):
            ytok = int(pred[b, t].item())
            if ytok <= 0:  # rest or pad
                continue
            pitch = ytok - 1  # raw MIDI pitch 0..127
            chord_label = id_to_chord.get(int(X[b, t].item()), "N")
            root_pc, qual = _parse_chord_label(chord_label) if chord_label != "N" else (-1, "N")
            if root_pc < 0:
                continue
            pcs = set((root_pc + p) % 12 for p in CHORD_PCS.get(qual, CHORD_PCS["maj"]))
            if (pitch % 12) in pcs:
                hits += 1
            total += 1
    return float(hits) / max(total, 1)

# -----------------------------
# Train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",  required=True, help="pairs.npz path")
    ap.add_argument("--split", required=True, help="split.json path")
    ap.add_argument("--vocab", default="", help="vocab.json path (for chords list)")
    ap.add_argument("--out",   required=True, help="output dir for checkpoints")

    # model/opt
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead",   type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--epochs",  type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--device", default="cpu", choices=["cpu","cuda","mps"])

    # augmentation
    ap.add_argument("--transpose_aug", type=int, default=0, help="max semitone shift for random transposition (0=off, 12=recommend)")
    ap.add_argument("--transpose_prob", type=float, default=0.5, help="probability to apply transposition to a batch")

    # scaffold / theory weighting
    ap.add_argument("--use_scaffold_loss", action="store_true", help="multiply token loss by theory weights")
    ap.add_argument("--steps_per_beat", type=int, default=4)
    ap.add_argument("--pitch_min", type=int, default=0, help="dataset token mapping uses absolute MIDI, so 0..127 here")
    ap.add_argument("--pitch_max", type=int, default=127)

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Load
    X, Y = load_npz(args.data)
    N = len(X)
    tr_idx, va_idx = load_split(args.split, N)

    # chord vocab
    if args.vocab and os.path.isfile(args.vocab):
        with open(args.vocab, "r") as f:
            voc = json.load(f)
        chords_list = voc.get("chords", [])
    else:
        # fallback to preprocessing defaults
        CHROMA_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
        chords_list = ["N"] + [f"{r}:{q}" for r in CHROMA_NAMES for q in ("maj","min")]
    id_to_chord, chord_to_id = build_vocab_maps(chords_list)

    n_chords = len(chords_list)
    n_pitches = 129  # 0..128

    # Datasets
    ds_tr = PairDataset(X, Y, tr_idx)
    ds_va = PairDataset(X, Y, va_idx)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_pad, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=0)

    device = choose_device(args.device)
    print(f"🧠 Device: {device} | n_chords={n_chords} | n_pitches={n_pitches} | train={len(tr_idx)} | val={len(va_idx)}")

    # Model / opt
    model = TinyTransformer(n_chords=n_chords, n_pitches=n_pitches, d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    ce_loss = nn.CrossEntropyLoss(ignore_index=-100)

    # Optional scaffold instance for loss weighting
    scaf = None
    if args.use_scaffold_loss and Scaffold is not None:
        scaf = Scaffold(
            steps_per_beat=args.steps_per_beat,
            pitch_min=args.pitch_min,
            pitch_max=args.pitch_max,
            y_rest_token=0
        )
        print("🎼 Scaffold loss weighting enabled.")
    elif args.use_scaffold_loss and Scaffold is None:
        print("⚠️  --use_scaffold_loss requested but scaffold.py not importable; continuing without.")

    best_val = float("inf")
    best_path = os.path.join(args.out, "best.pt")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        tr_loss_sum = 0.0
        tr_acc_sum  = 0.0
        tr_ct_sum   = 0.0
        n_batches   = 0

        for Xb, Yb in dl_tr:
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            # optional transposition augmentation (same shift for chords and melody)
            if args.transpose_aug > 0 and random.random() < args.transpose_prob:
                shift = random.randint(-args.transpose_aug, args.transpose_aug)
                if shift != 0:
                    Xb = transpose_chord_ids(Xb, id_to_chord, chord_to_id, shift)
                    Yb = transpose_pitch_tokens(Yb, shift)

            logits = model(Xb)  # (B,T,129)

            # base loss
            loss = ce_loss(logits.reshape(-1, logits.size(-1)), Yb.view(-1))

            # optional theory-weighted loss (per-token scalar)
            if scaf is not None:
                with torch.no_grad():
                    # compute per-token weights
                    B, T = Yb.shape
                    w = torch.ones_like(Yb, dtype=torch.float32, device=device)
                    for b in range(B):
                        for t in range(T):
                            ytok = int(Yb[b, t].item())
                            if ytok == -100:
                                continue
                            chord_label = id_to_chord.get(int(Xb[b, t].item()), "N")
                            w[b, t] = scaf.loss_weight_for_token(chord_label=chord_label, t=t, y_token=ytok)
                # re-compute unreduced CE per token
                per_token = nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    Yb.view(-1),
                    reduction="none",
                    ignore_index=-100
                ).view_as(Yb).float()
                # apply weights only where not ignore
                mask = (Yb != -100).float()
                weighted = (per_token * w * mask).sum() / torch.clamp(mask.sum(), min=1.0)
                loss = weighted

            optim.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_((model.parameters(), 1.0)
            )
            optim.step()

            # metrics
            with torch.no_grad():
                tr_loss_sum += float(loss.item())
                tr_acc_sum  += masked_accuracy(logits, Yb)
                tr_ct_sum   += chord_tone_hit_rate(logits, Xb, id_to_chord)
                n_batches   += 1

        # ---- Validation
        model.eval()
        va_loss_sum = 0.0
        va_acc_sum  = 0.0
        va_ct_sum   = 0.0
        va_batches  = 0
        with torch.no_grad():
            for Xb, Yb in dl_va:
                Xb = Xb.to(device); Yb = Yb.to(device)
                logits = model(Xb)
                loss = ce_loss(logits.reshape(-1, logits.size(-1)), Yb.view(-1))
                va_loss_sum += float(loss.item())
                va_acc_sum  += masked_accuracy(logits, Yb)
                va_ct_sum   += chord_tone_hit_rate(logits, Xb, id_to_chord)
                va_batches  += 1

        tr_loss = tr_loss_sum / max(1, n_batches)
        tr_acc  = tr_acc_sum  / max(1, n_batches)
        tr_ct   = tr_ct_sum   / max(1, n_batches)

        va_loss = va_loss_sum / max(1, va_batches)
        va_acc  = va_acc_sum  / max(1, va_batches)
        va_ct   = va_ct_sum   / max(1, va_batches)

        dt = time.time() - t0
        print(f"Epoch {epoch:02d} | {dt:5.1f}s | train loss {tr_loss:.3f}, acc {tr_acc:.3f}, CT {tr_ct:.3f} || val loss {va_loss:.3f}, acc {va_acc:.3f}, CT {va_ct:.3f}")

        # Save last
        last_path = os.path.join(args.out, "last.pt")
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "_hparams": {
                "n_chords": n_chords, "n_pitches": n_pitches,
                "d_model": args.d_model, "nhead": args.nhead, "nlayers": args.nlayers
            },
            "_vocab": chords_list
        }, last_path)

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "_hparams": {
                    "n_chords": n_chords, "n_pitches": n_pitches,
                    "d_model": args.d_model, "nhead": args.nhead, "nlayers": args.nlayers
                },
                "_vocab": chords_list
            }, best_path)
            print(f"  ✅ saved best -> {best_path}")

if __name__ == "__main__":
    main()