from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math, json
import numpy as np

_PITCH_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
_NAME_TO_PC = {n:i for i,n in enumerate(_PITCH_NAMES)}

def _pc_of(name: str) -> int:
    # accepts "C", "F#", "Bb" (-> A#)
    name = name.strip().replace("♯","#").replace("♭","b")
    if "b" in name and "#" not in name:
        # map flats to sharps for simplicity
        flats = {"Db":"C#","Eb":"D#","Gb":"F#","Ab":"G#","Bb":"A#"}
        name = flats.get(name, name)
    return _NAME_TO_PC[name]

def _parse_chord_label(lbl: str) -> Tuple[int, str]:
    # expects labels like "C:maj", "A#:min", "N"
    if lbl == "N":
        return -1, "N"
    root, qual = lbl.split(":")
    return _pc_of(root), qual

# scale pitch-classes (relative to root)
SCALE_PCS: Dict[str, List[int]] = {
    "ionian":        [0,2,4,5,7,9,11],
    "dorian":        [0,2,3,5,7,9,10],
    "mixolydian":    [0,2,4,5,7,9,10],
    "aeolian":       [0,2,3,5,7,8,10],
    "major_pent":    [0,2,4,7,9],
    "minor_pent":    [0,3,5,7,10],
    "blues":         [0,3,5,6,7,10],
    "altered":       [0,1,3,4,6,8,10],     # super-Locrian (mm7)
    "lyd_dom":       [0,2,4,6,7,9,10],     # melodic minor mode 4
    "whole_half":    [0,2,3,5,6,8,9,11],   # dim(7) context
    "locrian":       [0,1,3,5,6,8,10],
}

# chord tone pitch-classes (relative to root)
CHORD_PCS: Dict[str, List[int]] = {
    "maj":  [0,4,7],
    "min":  [0,3,7],
    "7":    [0,4,7,10],   # dom7
    "maj7": [0,4,7,11],
    "min7": [0,3,7,10],
    "dim":  [0,3,6],
    "m7b5": [0,3,6,10],   # half-dim
}

def _choose_scale_for_chord(qual: str) -> str:
    # simple, safe defaults; expand as needed
    if qual in ("maj","maj7"):    return "ionian"
    if qual in ("min","min7"):    return "dorian"   # dorian is improviser-friendly
    if qual in ("7",):            return "mixolydian"
    if qual in ("dim",):          return "whole_half"
    if qual in ("m7b5",):         return "locrian"
    return "ionian"

@dataclass
class Scaffold:
    steps_per_beat: int = 4
    pitch_min: int = 48    # C3
    pitch_max: int = 84    # C6
    y_rest_token: int = 0  # your dataset uses 0=rest, else pitch+1
    w_ct: float = 3.0      # weight: chord tones
    w_gt: float = 1.5      # extra weight for guide tones (3rd & 7th)
    w_sc: float = 1.6      # in-scale non-chord tones
    w_app: float = 1.2     # chromatic approaches/enclosures around chord tones
    w_off: float = 0.6     # off-scale fallback
    beat_boost: float = 1.25  # boost on strong beats

    def _pitch_range(self) -> List[int]:
        return list(range(self.pitch_min, self.pitch_max + 1))

    def _mask_from_pcs(self, root_pc: int, pcs: List[int]) -> np.ndarray:
        pcs_set = { (root_pc + p) % 12 for p in pcs }
        mask = np.zeros(self.pitch_max - self.pitch_min + 1, dtype=bool)
        for i, p in enumerate(self._pitch_range()):
            if (p % 12) in pcs_set:
                mask[i] = True
        return mask

    def _guide_tone_mask(self, root_pc: int, qual: str) -> np.ndarray:
        pcs = CHORD_PCS.get(qual, CHORD_PCS["maj"])
        gts = []
        if 4 in pcs: gts.append(4)   # 3rd
        if 3 in pcs: gts.append(3)
        if 10 in pcs or 11 in pcs:   # 7th (dom/maj7/min7)
            gts.append(10 if 10 in pcs else 11)
        return self._mask_from_pcs(root_pc, gts) if gts else np.zeros(self.pitch_max - self.pitch_min + 1, bool)

    def _approach_mask(self, root_pc: int, qual: str) -> np.ndarray:
        pcs = CHORD_PCS.get(qual, CHORD_PCS["maj"])
        # ±1 semitone “enclosure” around chord tones
        around = []
        for p in pcs:
            around += [ (p-1) % 12, (p+1) % 12 ]
        return self._mask_from_pcs(root_pc, around)

    def rhythmic_weight(self, t: int) -> float:
        # strong on downbeats; mild on backbeats
        pos = t % self.steps_per_beat
        if pos == 0:     return self.beat_boost
        if pos == self.steps_per_beat // 2:  return 1.1
        return 1.0

    def prior_logits(
        self,
        chord_label: str,      # e.g., "C:maj", "G:7", or "N"
        t: int,                # timestep index
        scale_override: Optional[str] = None,
        add_blues_on_dom: bool = True,
        add_bebop_on_strong: bool = True,
    ) -> np.ndarray:
        """
        Returns log-bias for each **pitch token** (index matches model's Y: 0=rest, 1..=pitch+1 in MIDI range).
        """
        size = (self.pitch_max - self.pitch_min + 1) + 1  # + rest
        w = np.full(size, self.w_off, dtype=np.float32)
        w[self.y_rest_token] = 0.9  # allow rests but not dominate

        root_pc, qual = _parse_chord_label(chord_label)
        if root_pc < 0:   # "N" (no chord) → uniform-ish
            return np.log(w / w.sum())

        # chord tones & guide tones
        ct_mask = self._mask_from_pcs(root_pc, CHORD_PCS.get(qual, CHORD_PCS["maj"]))
        gt_mask = self._guide_tone_mask(root_pc, qual)
        # scale tones
        scale_name = scale_override or _choose_scale_for_chord(qual)
        sc_mask = self._mask_from_pcs(root_pc, SCALE_PCS[scale_name])

        # optional color sets
        add_mask = np.zeros_like(ct_mask)
        if qual == "7" and add_blues_on_dom:
            add_mask |= self._mask_from_pcs(root_pc, SCALE_PCS["blues"])
        # bebop idea: add a passing tone so chord tones land on strong beats
        if add_bebop_on_strong and (t % self.steps_per_beat == 0):
            if qual in ("maj","maj7"):
                add_mask |= self._mask_from_pcs(root_pc, [0,2,4,5,7,9,11,6])    # major bebop (add ♯5/#11-ish pass)
            elif qual in ("7",):
                add_mask |= self._mask_from_pcs(root_pc, [0,2,4,5,7,9,10,11])  # dominant bebop (+maj7 as pass)

        app_mask = self._approach_mask(root_pc, qual)

        # fill weights
        # pitch tokens start at 1 (pitch+1 in your dataset)
        off = 1
        w[off:off+len(ct_mask)][ct_mask] = self.w_ct
        w[off:off+len(gt_mask)][gt_mask] += self.w_gt
        w[off:off+len(sc_mask)][sc_mask] = np.maximum(w[off:off+len(sc_mask)][sc_mask], self.w_sc)
        w[off:off+len(app_mask)][app_mask] = np.maximum(w[off:off+len(app_mask)][app_mask], self.w_app)
        w[off:off+len(add_mask)][add_mask] = np.maximum(w[off:off+len(add_mask)][add_mask], self.w_sc)

        # rhythmic emphasis
        w *= self.rhythmic_weight(t)

        # normalize → log
        w = np.clip(w, 1e-6, None)
        w /= w.sum()
        return np.log(w)

    def loss_weight_for_token(self, chord_label: str, t: int, y_token: int) -> float:
        """
        Return a scalar multiplier for CE loss at (t), larger when y is a chord/guide tone on strong beats.
        """
        if y_token == self.y_rest_token:
            return 1.0
        pitch = self.pitch_min + (y_token - 1)
        root_pc, qual = _parse_chord_label(chord_label)
        if root_pc < 0: return 1.0

        pc = pitch % 12
        ct = set((root_pc + p) % 12 for p in CHORD_PCS.get(qual, CHORD_PCS["maj"]))
        gt = set((root_pc + p) % 12 for p in [3,4,10,11])  # 3rds & 7ths (any present)

        base = 1.0
        if pc in ct: base *= 1.6
        if pc in gt: base *= 1.3
        if (t % self.steps_per_beat) == 0: base *= 1.2  # strong beat
        return base