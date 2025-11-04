#!/usr/bin/env python3
import argparse, json, os, numpy as np
import torch, torch.nn as nn
from miditoolkit import MidiFile, Instrument, Note

# --- tiny transformer (must match train script dims) ---
class TinyTransformer(nn.Module):
    def __init__(self, n_chords, n_pitches, d_model=128, nhead=4, nlayers=4):
        super().__init__()
        self.chord_emb = nn.Embedding(n_chords, d_model)
        self.pos_emb   = nn.Embedding(4096, d_model)
        encoder_layer  = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=4*d_model, batch_first=True)
        self.encoder   = nn.TransformerEncoder(encoder_layer, nlayers)
        self.head      = nn.Linear(d_model, n_pitches)

    def forward(self, x):
        # x: (B, T) chord ids
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.chord_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h)  # (B, T, n_pitches)

def tokens_to_melody_midi(pitches, out_path, ticks_per_beat=480, steps_per_beat=4, vel=80):
    # pitches: list[int], 0=rest, 1..128=pitch+1
    mf = MidiFile(ticks_per_beat=ticks_per_beat)
    inst = Instrument(program=0, is_drum=False, name="SoloMuse-Lead")
    mf.instruments.append(inst)
    step = ticks_per_beat // steps_per_beat
    t = 0
    i = 0
    while i < len(pitches):
        p = pitches[i]
        if p == 0:
            t += step
            i += 1
            continue
        # extend while same pitch continues
        j = i + 1
        while j < len(pitches) and pitches[j] == p:
            j += 1
        start = t
        end   = t + (j - i) * step
        note  = Note(velocity=vel, pitch=p-1, start=start, end=end)
        inst.notes.append(note)
        t = end
        i = j
    mf.dump(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--vocab", required=True)
    ap.add_argument("--out",   default="sample.mid")
    ap.add_argument("--device", default="mps")
    ap.add_argument("--max_steps", type=int, default=256)
    args = ap.parse_args()

    with open(args.vocab, "r") as f:
        vocab = json.load(f)
    n_chords  = len(vocab["chords"])
    n_pitches = 129  # 0..128 (0=rest)

    # for demo, make a simple ii–V–i in A minor (Dm7 ~ D:min -> G:maj -> A:min triad proxies)
    # map chord strings to ids if present; else fallback to a repeating minor loop
    chord_to_id = {c:i for i,c in enumerate(vocab["chords"])}
    demo = ["D:min","G:maj","A:min","A:min"]
    chords = [chord_to_id.get(c, chord_to_id["A:min"]) for c in demo]
    # repeat to desired length
    reps = (args.max_steps // len(chords)) + 1
    chord_ids = (chords * reps)[:args.max_steps]
    x = torch.tensor(chord_ids, dtype=torch.long).unsqueeze(0)

    # load model
    sd = torch.load(args.ckpt, map_location="cpu")
    dims = sd.get("_hparams", {"d_model":128, "nhead":4, "nlayers":4})
    model = TinyTransformer(n_chords, n_pitches, **dims).to(args.device)
    model.load_state_dict(sd["model"])
    model.eval()

    with torch.no_grad():
        logits = model(x.to(args.device))      # (1, T, n_pitches)
        probs  = torch.softmax(logits, dim=-1)
        # greedy for MVP; swap to top-k sampling later
        pred   = probs.argmax(-1).squeeze(0).cpu().numpy().tolist()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tokens_to_melody_midi(pred, args.out)
    print(f"✅ Wrote demo solo to {args.out}")

if __name__ == "__main__":
    main()