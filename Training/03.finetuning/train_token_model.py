#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tiny chord->melody transformer trainer.

Inputs:
  - pairs.npz   (X: list[ndarray[int]] chord ids, Y: list[ndarray[int]] pitch tokens)
  - split.json  ({"val_idx":[...], "num":N})
  - vocab.json  ({"chords":[...], "pitch_pad":0})  [optional but recommended]

Targets:
  - predicts pitch token per timestep (0=rest, 1..128=pitch+1).
  - Pads labels with -100 so loss ignores padding; chords pad with 0 ('N').

Saves:
  - {out}/best.pt (lowest val loss)
  - {out}/last.pt (final epoch)
"""

from __future__ import annotations
import os, json, time, argparse, random
from typing import List, Tuple, Dict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
        j = self.idx[i]
        return self.X[j].astype(np.int64), self.Y[j].astype(np.int64)

def collate_pad(batch: List[Tuple[np.ndarray, np.ndarray]]):
    # Pad to max length in batch. Y pad with -100 (ignore_index), X pad with 0 ('N')
    xs, ys = zip(*batch)
    maxT = max(len(x) for x in xs)
    Xb = torch.zeros(len(xs), maxT, dtype=torch.long)         # chord pad = 0 (N)
    Yb = torch.full((len(xs), maxT), fill_value=-100, dtype=torch.long)  # label pad
    for i, (x, y) in enumerate(zip(xs, ys)):
        t = len(x)
        Xb[i, :t] = torch.from_numpy(x)
        Yb[i, :t] = torch.from_numpy(y)  # y is 0..128; we do NOT mask rests; only padding is -100
    return Xb, Yb

# -----------------------------
# Utils
# -----------------------------
def choose_device(name: str):
    name = (name or "cpu").lower()
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("⚠️  MPS requested but unavailable; falling back to CPU.")
    return torch.device("cpu")

def masked_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    # logits: (B,T,C), labels: (B,T) with -100 as ignore
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = labels != -100
        if mask.sum() == 0:
            return 0.0
        correct = (pred[mask] == labels[mask]).float().mean().item()
        return float(correct)

def load_npz(path: str):
    arr = np.load(path, allow_pickle=True)
    X = list(arr["X"])
    Y = list(arr["Y"])
    return X, Y

def infer_n_chords(X: List[np.ndarray]) -> int:
    mx = 0
    for x in X:
        if len(x):
            mx = max(mx, int(x.max()))
    return mx + 1

# -----------------------------
# Train
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="pairs.npz")
    ap.add_argument("--split", required=True, help="split.json with val_idx")
    ap.add_argument("--vocab", default=None, help="vocab.json (for n_chords); optional")
    ap.add_argument("--out", required=True, help="output dir for checkpoints")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", dest="batch_size", type=int, default=16)
    ap.add_argument("--batch", dest="batch_size", type=int, help="alias of --batch_size")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--nlayers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda","mps"])
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # Repro
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Load data
    X, Y = load_npz(args.data)
    with open(args.split, "r") as f:
        split = json.load(f)
    val_idx = set(split.get("val_idx", []))
    all_idx = list(range(len(X)))
    tr_idx = [i for i in all_idx if i not in val_idx]
    va_idx = sorted(list(val_idx))
    if not tr_idx or not va_idx:
        # fallback: 80/20 split
        random.shuffle(all_idx)
        k = max(1, int(0.2 * len(all_idx)))
        va_idx = all_idx[:k]
        tr_idx = all_idx[k:]
        print(f"⚠️  split.json missing or empty; using fallback split ({len(tr_idx)} train / {len(va_idx)} val).")

    # Determine vocab sizes
    n_pitches = 129  # 0..128 (0=rest)
    if args.vocab and os.path.isfile(args.vocab):
        with open(args.vocab, "r") as f:
            vocab = json.load(f)
        n_chords = len(vocab.get("chords", [])) or infer_n_chords(X)
    else:
        n_chords = infer_n_chords(X)

    # Datasets & loaders
    ds_tr = PairDataset(X, Y, tr_idx)
    ds_va = PairDataset(X, Y, va_idx)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  collate_fn=collate_pad, num_workers=0)
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, collate_fn=collate_pad, num_workers=0)

    # Device
    device = choose_device(args.device)
    print(f"🧠 Device: {device} | n_chords={n_chords} | n_pitches={n_pitches}")

    # Model / opt / loss
    model = TinyTransformer(
        n_chords=n_chords, n_pitches=n_pitches,
        d_model=args.d_model, nhead=args.nhead, nlayers=args.nlayers
    ).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    best_val = float("inf")
    hparams = {"d_model": args.d_model, "nhead": args.nhead, "nlayers": args.nlayers}

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # ---- Train
        model.train()
        tr_loss_sum, tr_acc_sum, tr_count = 0.0, 0.0, 0
        for Xb, Yb in dl_tr:
            Xb = Xb.to(device)
            Yb = Yb.to(device)

            logits = model(Xb)  # (B,T,C)
            loss = loss_fn(logits.view(-1, logits.size(-1)), Yb.view(-1))

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            tr_loss_sum += float(loss.item()) * Xb.size(0)
            tr_acc_sum  += masked_accuracy(logits, Yb) * Xb.size(0)
            tr_count    += Xb.size(0)

        tr_loss = tr_loss_sum / max(1, tr_count)
        tr_acc  = tr_acc_sum  / max(1, tr_count)

        # ---- Val
        model.eval()
        va_loss_sum, va_acc_sum, va_count = 0.0, 0.0, 0
        with torch.no_grad():
            for Xb, Yb in dl_va:
                Xb = Xb.to(device)
                Yb = Yb.to(device)
                logits = model(Xb)
                loss = loss_fn(logits.view(-1, logits.size(-1)), Yb.view(-1))
                va_loss_sum += float(loss.item()) * Xb.size(0)
                va_acc_sum  += masked_accuracy(logits, Yb) * Xb.size(0)
                va_count    += Xb.size(0)
        va_loss = va_loss_sum / max(1, va_count)
        va_acc  = va_acc_sum  / max(1, va_count)

        dt = time.time() - t0
        print(f"epoch {epoch:02d}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.3f} | {dt:.1f}s")

        # Save last
        last_path = os.path.join(args.out, "last.pt")
        torch.save({"model": model.state_dict(),
                    "optimizer": optim.state_dict(),
                    "epoch": epoch,
                    "_hparams": hparams}, last_path)

        # Save best
        if va_loss < best_val:
            best_val = va_loss
            best_path = os.path.join(args.out, "best.pt")
            torch.save({"model": model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "epoch": epoch,
                        "_hparams": hparams}, best_path)
            print(f"  ✅ saved best -> {best_path}")

if __name__ == "__main__":
    main()