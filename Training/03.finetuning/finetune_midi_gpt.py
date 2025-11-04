#!/usr/bin/env python3
# finetune_midi_gpt.py
import os
import math
import json
import random
import argparse
from pathlib import Path
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

# MIDI / tokenization
import miditoolkit
from miditok import REMI, TokSequence, TokenizerConfig

# HF
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
)

# ----------------------------
# Tokenizer setup (REMI with chords + tempo)
# ----------------------------
def build_tokenizer(token_dir: Path) -> REMI:
    token_dir.mkdir(parents=True, exist_ok=True)
    cfg = TokenizerConfig(
        use_chords=True,           # add chord tokens (triads from harmony)
        use_tempos=True,           # add tempo tokens
        use_time_signatures=True,  # add TS tokens
        pitch_range=(21, 108),     # piano-ish range
        beat_res={(0, 4): 4, (4, 12): 8},  # finer grid for faster sections
        nb_tempos=32,
        tempo_range=(30, 240),
        program_changes=True,
        num_velocities=16,
    )
    tokenizer = REMI(cfg)
    # Save config so we can reload later
    tokenizer.save_params(token_dir / "tokenizer.json")
    return tokenizer

# ----------------------------
# Dataset: tokenize all MIDIs, pack into blocks
# ----------------------------
class MidiTokenDataset(Dataset):
    def __init__(
        self,
        tokenizer: REMI,
        midi_root: Path,
        block_size: int = 1024,
        cache_dir: Optional[Path] = None,
        max_files: Optional[int] = None,
    ):
        self.block_size = block_size
        self.examples = []

        midi_paths = []
        for ext in ("*.mid", "*.midi", "*.MID"):
            midi_paths.extend(sorted(midi_root.rglob(ext)))
        if max_files is not None:
            midi_paths = midi_paths[:max_files]

        cache_dir = cache_dir or (midi_root / ".tok_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_ids: List[int] = []
        for p in tqdm(midi_paths, desc="Tokenizing MIDIs"):
            try:
                midi = miditoolkit.MidiFile(str(p))
                toks: TokSequence = tokenizer.midi_to_tokens(midi)
                ids = tokenizer([toks])[0].ids  # convert to integer ids
                # remove very short
                if len(ids) < 64:
                    continue
                all_ids.extend(ids + [tokenizer.eos_token_id])  # EOS between songs
            except Exception as e:
                # skip corrupted/unsupported
                continue

        # pack into fixed windows
        n_blocks = len(all_ids) // block_size
        for i in range(n_blocks):
            chunk = all_ids[i * block_size : (i + 1) * block_size]
            self.examples.append(torch.tensor(chunk, dtype=torch.long))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        x = self.examples[idx]
        return {"input_ids": x, "labels": x.clone()}

# ----------------------------
# Build an HF tokenizer wrapper around miditok vocab
# ----------------------------
def build_hf_tokenizer_from_miditok(tokenizer: REMI, save_dir: Path) -> PreTrainedTokenizerFast:
    # HF wants a vocab.json + merges.txt (for BPE) or a simple list for word-level.
    # We'll build a "word-level" tokenizer by enumerating miditok's vocab.
    vocab_list = tokenizer.vocab
    token_to_id = {tok: i for i, tok in enumerate(vocab_list)}
    id_to_token = {i: tok for tok, i in token_to_id.items()}

    vocab_path = save_dir / "vocab_miditok.json"
    with open(vocab_path, "w") as f:
        json.dump({"token_to_id": token_to_id, "id_to_token": id_to_token}, f)

    # use HF fast tokenizer in "pretokenized" mode (we feed ids directly)
    # We'll still register special tokens to make EOS work nicely.
    hf_tok = PreTrainedTokenizerFast(
        tokenizer_file=None,
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
        pad_token="<PAD>",
    )
    # map specials to miditok specials if present, else append
    # Ensure eos id aligns:
    if tokenizer.eos_token is None:
        # define one
        tokenizer.add_to_vocab("<EOS>")
    hf_tok.add_special_tokens({"eos_token": tokenizer.eos_token or "<EOS>"})
    return hf_tok

# ----------------------------
# Simple chord-primed generation (toy)
# ----------------------------
@torch.no_grad()
def generate_from_chords(
    model, tokenizer_hf: PreTrainedTokenizerFast, tokenizer_mt: REMI,
    chord_roots: List[str], n_events: int = 256, temperature: float = 1.0, device="cpu"
):
    """
    chord_roots: e.g. ["C:maj", "F:maj", "G:maj", "C:maj"]
    This function fabricates chord tokens from strings that miditok understands
    and lets the model continue with note/other events.
    """
    # Convert textual chord roots into miditok chord tokens if available
    # miditok uses tokens like "Chord_C:maj" under REMI (depends on version)
    chord_tokens = []
    for c in chord_roots:
        tok = f"Chord_{c}"
        if tok in tokenizer_mt.vocab:
            chord_tokens.append(tokenizer_mt.vocab[tok])
    if not chord_tokens:
        raise ValueError("No valid chord tokens recognized by tokenizer.")

    # Prime with a simple bar start + chords sequence
    prompt_ids = chord_tokens[:]
    input_ids = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    model.eval()
    out = input_ids
    for _ in range(n_events):
        logits = model(out).logits[:, -1, :] / max(1e-6, temperature)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        out = torch.cat([out, next_id], dim=1)

    return out[0].tolist()

# ----------------------------
# Main
# ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--midi_root", type=str, required=True, help="Folder with .mid files (e.g., Slakh MIDI dir)")
    parser.add_argument("--out_dir", type=str, default="runs/midi_gpt")
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--n_layer", type=int, default=8)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_embd", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_files", type=int, default=None, help="For smoke tests, cap number of MIDIs")
    parser.add_argument("--eval_split", type=float, default=0.02)
    parser.add_argument("--generate_demo", action="store_true", help="After training, sample on a chord grid")
    parser.add_argument("--device", type=str, default=None, help="cpu|cuda|mps (auto if None)")
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    midi_root = Path(args.midi_root)

    # device
    if args.device:
        device = args.device
    else:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    print(f"▶ Using device: {device}")

    # tokenizer
    tok_dir = out_dir / "tokenizer"
    tokenizer_mt = build_tokenizer(tok_dir)
    tokenizer_hf = build_hf_tokenizer_from_miditok(tokenizer_mt, tok_dir)

    # dataset
    ds = MidiTokenDataset(
        tokenizer=tokenizer_mt,
        midi_root=midi_root,
        block_size=args.block_size,
        cache_dir=out_dir / ".tok_cache",
        max_files=args.max_files,
    )
    if len(ds) < 10:
        print("Not enough sequences after packing. Add more MIDIs or reduce block_size.")
        return

    # split
    n = len(ds)
    n_eval = max(1, int(n * args.eval_split))
    indices = list(range(n))
    random.shuffle(indices)
    eval_idx = indices[:n_eval]
    train_idx = indices[n_eval:]

    train_ds = torch.utils.data.Subset(ds, train_idx)
    eval_ds  = torch.utils.data.Subset(ds, eval_idx)

    # model
    vocab_size = len(tokenizer_mt.vocab)
    print(f"Vocab size: {vocab_size}, Train chunks: {len(train_ds)}, Eval chunks: {len(eval_ds)}")

    cfg = GPT2Config(
        vocab_size=vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        bos_token_id=None,
        eos_token_id=tokenizer_mt.eos_token_id,  # used in packing
    )
    model = GPT2LMHeadModel(cfg)
    model.to(device)

    # trainer
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer_hf, mlm=False)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    save_steps = max(100, steps_per_epoch)  # save every ~epoch

    targs = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="steps",
        eval_steps=save_steps,
        save_steps=save_steps,
        logging_steps=50,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=0.01,
        fp16=torch.cuda.is_available(),
        bf16=False,  # set True if on newer GPUs w/ bf16
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(out_dir / "model"))
    # Save miditok tokenizer params for later
    tokenizer_mt.save_params(out_dir / "tokenizer" / "tokenizer.json")

    print("✅ Training complete.")
    if args.generate_demo:
        print("🎹 Generating a short solo conditioned on a toy chord loop…")
        chord_loop = ["C:maj", "F:maj", "G:maj", "C:maj"]
        ids = generate_from_chords(
            model.eval(), tokenizer_hf, tokenizer_mt, chord_loop,
            n_events=256, temperature=1.0, device=device
        )
        # Write back to MIDI using miditok
        seq = TokSequence(ids=ids)
        # Convert ids -> tokens then tokens -> midi
        # (miditok expects a list[TokSequence])
        try:
            midi_out = tokenizer_mt.tokens_to_midi([seq])
            midi_path = out_dir / "demo_solo.mid"
            midi_out.dump(midi_path)
            print(f"💾 Wrote {midi_path}")
        except Exception as e:
            print(f"Could not render demo MIDI: {e}")

if __name__ == "__main__":
    main()