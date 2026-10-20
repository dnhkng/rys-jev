"""Bake an RYS layer duplication and a symmetric alpha into a standalone checkpoint.

For span ``[i, j)`` the executed stack is ``0..j-1, i..j-1, j..N-1``.  A block's
residual update is linear in its output projections, so both copies of the
duplicated blocks are scaled by alpha exactly:

* full-attention layers: ``self_attn.o_proj.weight``
* linear-attention layers: ``linear_attn.out_proj.weight``
* every layer: ``mlp.down_proj.weight``

After this, an ordinary forward over the new stack (with the model's usual
bidirectional prompt attention) is the duplicated, alpha-scaled model -- no
runtime layer map needed.

    python -m rys_jev.materialize --src base/ --dst rys-jev/ --span 41-49 --alpha 1.25
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

LAYER_RE = re.compile(r"^model\.language_model\.layers\.(\d+)\.(.+)$")
SCALED = ("self_attn.o_proj.weight", "linear_attn.out_proj.weight", "mlp.down_proj.weight")


def parse_span(span: str) -> tuple[int, int]:
    i, j = (int(x) for x in span.split("-"))
    if not 0 <= i < j:
        raise ValueError(f"bad span {span!r}: use i-j with 0 <= i < j")
    return i, j


def execution_order(n: int, i: int, j: int) -> list[int]:
    """Physical block indices of the executed stack: 0..j-1, i..j-1, j..n-1."""
    return list(range(0, j)) + list(range(i, n))


def new_positions(p: int, i: int, j: int) -> list[int]:
    """New stack positions of physical layer p."""
    if p < i:
        return [p]
    if p < j:
        return [p, p + (j - i)]
    return [p + (j - i)]


def scale_for(rest: str, alpha: float) -> float:
    return alpha if rest.endswith(SCALED) else 1.0


def materialize(src: str, dst: str, span: str = "41-49", alpha: float = 1.25,
                shard_gb: float = 4.5, verbose: bool = True) -> dict:
    src, dst = Path(src), Path(dst)
    i, j = parse_span(span)
    cfg = json.load(open(src / "config.json"))
    text = cfg["text_config"]
    n = int(text["num_hidden_layers"])
    types = list(text["layer_types"])
    if len(types) != n or not (0 <= i < j <= n):
        raise ValueError(f"span [{i},{j}) does not fit {n} layers")
    new_types = types[:j] + types[i:j] + types[j:]
    new_n = n + (j - i)

    idx = json.load(open(src / "model.safetensors.index.json"))
    shards = sorted(set(idx["weight_map"].values()))
    tensors = {}
    for s in shards:
        tensors.update(load_file(src / s, device="cpu"))
        if verbose:
            print("loaded", s, len(tensors), flush=True)

    out = {}
    for k, t in tensors.items():
        m = LAYER_RE.match(k)
        if not m:
            out[k] = t.clone()
            continue
        p, rest = int(m.group(1)), m.group(2)
        scale = scale_for(rest, alpha)
        for q in new_positions(p, i, j):
            key = f"model.language_model.layers.{q}.{rest}"
            out[key] = (t.float() * scale).to(t.dtype) if scale != 1.0 else t.clone()

    dst.mkdir(parents=True, exist_ok=True)
    order = [k for k in out if ".layers." not in k] + [k for k in out if ".layers." in k]
    limit = int(shard_gb * 1e9)
    shard, size, weight_map, shard_no = {}, 0, {}, 1

    def flush():
        nonlocal shard, size, shard_no
        if not shard:
            return
        name = f"model-{shard_no:05d}-of-{shard_no:05d}.safetensors"
        save_file(shard, str(dst / name), metadata={"format": "pt"})
        for k in shard:
            weight_map[k] = name
        shard, size = {}, 0
        shard_no += 1

    for k in order:
        t = out[k]
        nb = t.numel() * t.element_size()
        if size + nb > limit:
            flush()
        shard[k] = t
        size += nb
        if k in ("model.language_model.embed_tokens.weight", "lm_head.weight",
                 "model.language_model.norm.weight"):
            flush()
    flush()

    total = shard_no - 1
    files = sorted(dst.glob("model-*.safetensors"))
    rename = {f.name: f"model-{n_:05d}-of-{total:05d}.safetensors" for n_, f in enumerate(files, 1)}
    for old, new in rename.items():
        if old != new:
            (dst / old).rename(dst / new)
    weight_map = {k: rename.get(v, v) for k, v in weight_map.items()}

    total_size = sum(t.numel() * t.element_size() for t in out.values())
    json.dump({"metadata": {"format": "pt", "total_size": total_size}, "weight_map": weight_map},
              open(dst / "model.safetensors.index.json", "w"), indent=1)

    text["num_hidden_layers"] = new_n
    text["layer_types"] = new_types
    cfg["rys"] = {"source": src.name, "span": [i, j], "alpha": alpha,
                  "duplicated_layers": list(range(i, j)),
                  "note": "layers i..j-1 executed twice; both copies scaled by alpha via output projections"}
    json.dump(cfg, open(dst / "config.json", "w"), indent=1)
    for f in ["generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
              "merges.txt", "chat_template.jinja", "preprocessor_config.json",
              "video_preprocessor_config.json", "decision_config.json", "requirements.txt",
              "LICENSE", "NOTICE"]:
        if (src / f).exists():
            shutil.copy2(src / f, dst / f)
    report = {"layers": new_n, "shards": total, "tensors": len(weight_map),
              "bytes": total_size, "span": [i, j], "alpha": alpha}
    if verbose:
        print(json.dumps(report, indent=1))
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--span", default="41-49")
    ap.add_argument("--alpha", type=float, default=1.25)
    ap.add_argument("--shard-gb", type=float, default=4.5)
    a = ap.parse_args()
    materialize(a.src, a.dst, a.span, a.alpha, a.shard_gb)


if __name__ == "__main__":
    main()
