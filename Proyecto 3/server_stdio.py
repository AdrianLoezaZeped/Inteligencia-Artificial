#!/usr/bin/env python3
"""Autocompletado Keras; JSON por línea en stdin/stdout."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent

_model = None
_stoi: dict[str, int] = {}
_itos: dict[int, str] = {}
_BLOCK_SIZE = 32


def _load() -> None:
    global _model, _stoi, _itos, _BLOCK_SIZE
    meta = json.loads((ROOT / "meta.json").read_text(encoding="utf-8"))
    _BLOCK_SIZE = int(meta["block_size"])
    chars = meta["chars"]
    _stoi = {c: i for i, c in enumerate(chars)}
    _itos = {i: c for c, i in _stoi.items()}
    _model = tf.keras.models.load_model(ROOT / "model.keras")


def _encode(s: str) -> list[int]:
    return [_stoi[c] for c in s]


def _decode(ids: list[int]) -> str:
    return "".join(_itos[i] for i in ids)


def _complete(prefix: str, max_new: int = 80, temperature: float = 0.01) -> str:
    ids = _encode(prefix)
    rng = np.random.default_rng() # Quitamos el 42 fijo
    
    # Buscamos el ID del espacio en blanco. Si no existe, usamos el 0 por defecto.
    pad_char = _stoi.get(' ', 0)
    
    for _ in range(max_new):
        x = np.array(ids[-_BLOCK_SIZE:], dtype=np.int64)
        if x.shape[0] < _BLOCK_SIZE:
            # EL ARREGLO: Rellenamos con espacios en blanco en lugar de la primera letra
            pad = np.full(_BLOCK_SIZE - x.shape[0], pad_char, dtype=np.int64)
            x = np.concatenate([pad, x])
            
        logits = _model(x.reshape(1, _BLOCK_SIZE), training=False).numpy()[0, -1, :]
        logits = logits / max(temperature, 1e-6)
        logits = logits - logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        ids.append(int(rng.choice(len(probs), p=probs)))
        
    return _decode(ids)

def _suggest(prefix: str, n: int = 5) -> list[str]:
    seen, out = set(), []
    for i in range(n * 3):
        text = _complete(prefix, max_new=50, temperature=0.7 + 0.05 * i)
        line = (prefix + text[len(prefix) :].split("\n")[0])[:80]
        if line not in seen and len(line) > len(prefix):
            seen.add(line)
            out.append(line)
        if len(out) >= n:
            break
    return out


def main() -> None:
    _load()
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            req = json.loads(line)
            method = req.get("method")
            if method == "complete":
                res = _complete(
                    req.get("prefix", ""),
                    max_new=req.get("max_new", 60),
                    temperature=req.get("temperature", 0.75),
                )
                print(json.dumps({"ok": True, "text": res}), flush=True)
            elif method == "suggest":
                res = _suggest(req.get("prefix", ""), n=req.get("n", 5))
                print(json.dumps({"ok": True, "items": res}), flush=True)
            else:
                print(json.dumps({"ok": False, "error": "unknown method"}), flush=True)
        except Exception as e:
            print(json.dumps({"ok": False, "error": str(e)}), flush=True)


if __name__ == "__main__":
    main()