# RYS-JEV

**RYS-JEV** is a layer-duplicated [StartLux-Decision v1.1 27B Instant](https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant):
blocks **41–48 run twice** (72 blocks total) and the residual update of **both copies** is scaled by **α = 1.25**.

This repository is an independent inference engine and toolchain for that checkpoint. It implements the model's
bidirectional prompt attention, the decision read-out, a `/v1/systemone`-compatible server, and the script that bakes
the duplication and α into a normal 72-layer checkpoint.

- Checkpoint: https://huggingface.co/dnhkng/RYS-JEV (CC BY-NC 4.0, non-commercial)
- Base model: https://huggingface.co/startlux-models/StartLux-Decision-v1.1-27B-Instant

## Results

Jev-Bench suite, 17 sources, 17,773 test questions split 8,932 development / 8,841 held-out. One-pass letter
read-out, no fine-tuning:

| model | dev | held-out | Δ vs base | W/L | p |
|---|---:|---:|---:|---:|---:|
| base 27B Instant | 80.52 | 80.40 | — | — | — |
| **RYS-JEV (41–49, α=1.25)** | **80.86** | **80.99** | **+0.59 pp** | 157/105 | 0.0016 |
| materialized checkpoint, dev re-run | 80.83 | — | — | — | — |

The materialized checkpoint matches the virtual duplication engine to 0.03 pp and 40/40 sampled argmaxes. On a
separate synthetic suite the gain was +0.06 pp (p=0.88) — the effect is benchmark-specific, not a general capability
increase.

## Install

```bash
git clone https://github.com/dnhkng/rys-jev.git
cd rys-jev
pip install -e .
```

Requirements: Python 3.10+, PyTorch 2.6+, `transformers` 5.17+ (for the Qwen3.5 hybrid class), `safetensors`,
`aiohttp`. A BF16 27B model needs about 62 GB of GPU memory; Flash SDPA is used when available.

## Run

```bash
hf download dnhkng/RYS-JEV --local-dir models/RYS-JEV
rys-jev serve --model models/RYS-JEV --port 8090
```

```bash
curl -s localhost:8090/v1/systemone -H 'Content-Type: application/json' -d '{
  "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
  "questions": {
    "team":     {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery",
                              "technical": "App and login"}},
    "urgent":   {"type": "noul", "instructions": "Should this ticket be answered today?"},
    "severity": {"type": "score", "instructions": "How severe is the impact?",
                 "criteria": ["cosmetic", "annoying", "blocks the customer"]}
  }
}'
```

Python:

```python
from rys_jev import load

model = load("models/RYS-JEV")
out = model.score(
    {"ticket": "I was charged twice for order #4411."},
    {"team": {"type": "choice", "instructions": "Which team should handle this ticket?",
              "criteria": {"billing": "Payments and refunds", "technical": "App and login"}},
     "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"}},
)
print(out["answers"])
```

`choice` returns `choice`, `confidence` and the full `probabilities`; `noul` returns the probability of *yes*; `score`
returns the expected level, `confidence`, per-level `probabilities` and the `legend`.

## Build the checkpoint from the base model

The 72-layer checkpoint is reproducible from the public base model:

```bash
rys-jev materialize --src models/StartLux-Decision-v1.1-27B-Instant \
                    --dst  models/RYS-JEV --span 41-49 --alpha 1.25
```

Why this is exact: a transformer block computes `h' = h + f(h)` where `f` is its residual branch. Scaling both of the
block's output projections by α gives `h + α·(mixer_out + mlp_out) = h + α·(block(h) − h)`, which is precisely the
symmetric-α RYS update `h ← prev + α·(h − prev)` applied to both copies. The projections are:

- `self_attn.o_proj` on full-attention layers,
- `linear_attn.out_proj` on linear-attention layers,
- `mlp.down_proj` on every layer.

The tool duplicates layers 41–48 (positions 49–56), applies α to both copies of those blocks, extends `layer_types`,
and writes a normal safetensors model that any loader can run. Runtime duplication on a 64-layer base is also
supported (`score(..., layer_map=..., alpha=1.25)`), which is how the checkpoint was validated.

## How inference works

StartLux-Decision Instant models were trained with **bidirectional attention over the prompt** in their full-attention
layers; linear-attention layers keep their recurrent path. Engines that read the prompt causally produce a different,
worse model. `engine.enable()` reroutes the full-attention layers through a bidirectional SDPA implementation and
self-checks, on the loaded weights, that (1) the rerouted causal forward still reproduces the stock forward, (2) padding
never reaches a real token, and (3) a late token reaches position 0.

The answer is read from the softmax over the option letters (A–Z) at the last prompt token, with the per-type
temperatures from `decision_config.json`. Nothing is generated: one forward pass per question.

## Limits

- Instant checkpoints only; AutoThink (thought generation) is not implemented.
- Text and JSON evidence; image slots are not handled.
- One question per forward pass; >26 options (wide choice, multi-pass knock-out) are not implemented.
- The 27B model needs a 62 GB GPU; CPU runs are possible but very slow.

## License and attribution

The code in this repository is Apache-2.0. The model weights are a derivative of StartLux-Decision and remain
**CC BY-NC 4.0** — non-commercial use only; commercial use requires a license from StartLux. This is an unofficial
research project, not affiliated with or endorsed by StartLux Labs. See `NOTICE`.

Base model and technical report: Zhuang et al., *StartLux-Decision: AutoThink Decision Models with Reinforcement
Learning for Calibrated Deliberative Decisions*, arXiv:2610.17309.
