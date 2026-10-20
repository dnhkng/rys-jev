"""Inference for StartLux-Decision Instant checkpoints, with optional RYS layer duplication.

Full-attention layers read the whole prompt without a causal term (the model was
trained that way); linear-attention layers keep their stock recurrent path.  The
answer is the softmax over option letters at the last prompt token.

RYS: executing a block twice at its natural position (or at runtime via
``layer_map``) and scaling both copies' residual update by alpha is exactly what
:mod:`rys_jev.materialize` bakes into a 72-layer checkpoint; this module can also
do it at runtime on a 64-layer base model.
"""

from __future__ import annotations

import copy
from contextlib import nullcontext
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from . import specs

IMPL = "rys_bidirectional"


def _attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None,
               is_causal=None, **kwargs):
    """Full attention over each row's real tokens, with no causal term."""
    lengths = kwargs.pop("startlux_row_lengths", None)
    if lengths is None:
        from transformers.integrations.sdpa_attention import sdpa_attention_forward
        return sdpa_attention_forward(module, query, key, value, attention_mask,
                                      dropout=dropout, scaling=scaling, is_causal=is_causal, **kwargs)
    if attention_mask is not None:
        raise ValueError("bidirectional rows take no attention mask")
    rows, heads, length, _ = query.shape
    if len(lengths) != rows:
        raise ValueError("one length per row required")
    gqa = key.shape[1] != heads
    if all(n == length for n in lengths):
        out = F.scaled_dot_product_attention(query, key, value, dropout_p=dropout,
                                             is_causal=False, scale=scaling, enable_gqa=gqa)
        return out.transpose(1, 2).contiguous(), None
    out = query.new_zeros(rows, length, heads, query.shape[-1])
    for i, n in enumerate(lengths):
        o = F.scaled_dot_product_attention(query[i:i + 1, :, :n].contiguous(),
                                           key[i:i + 1, :, :n].contiguous(),
                                           value[i:i + 1, :, :n].contiguous(),
                                           dropout_p=dropout, is_causal=False,
                                           scale=scaling, enable_gqa=gqa)
        out[i, :n] = o[0].transpose(0, 1)
    return out, None


def _kernel_guard(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    from torch.nn.attention import SDPBackend, sdpa_kernel
    return sdpa_kernel([SDPBackend.FLASH_ATTENTION])


def _masks(body, embeds, attention_mask, position_ids):
    from transformers.masking_utils import create_causal_mask, create_recurrent_attention_mask
    cfg = body.config
    linear = create_recurrent_attention_mask(config=cfg, inputs_embeds=embeds, attention_mask=attention_mask,
                                             past_key_values=None, position_ids=position_ids)
    causal = create_causal_mask(config=cfg, inputs_embeds=embeds, attention_mask=attention_mask,
                                past_key_values=None, position_ids=position_ids)
    return linear, causal


@torch.no_grad()
def forward(body, input_ids: torch.Tensor, valid: torch.Tensor | None = None,
            layer_map: Sequence[int] | None = None, bidirectional: bool = True,
            alpha: float = 1.0) -> torch.Tensor:
    """Last hidden state ``[B, L, H]`` of the text decoder.

    ``valid`` is a right-padded ``[B, L]`` mask of real tokens (None = no padding).
    ``layer_map`` is the execution order of physical block indices; repeated
    blocks get ``h = prev + alpha * (h - prev)``.
    """
    cfg = body.config
    B, L = input_ids.shape
    device = input_ids.device
    embeds = body.embed_tokens(input_ids)
    pos = torch.arange(L, device=device).view(1, 1, -1).expand(4, B, -1)
    text_pos = pos[0]
    position_embeddings = body.rotary_emb(embeds, pos[1:])

    if valid is None:
        attention_mask = None
        lengths = [L] * B
    else:
        vb = valid.to(device=device, dtype=torch.bool)
        n = vb.sum(dim=-1)
        if not bool((vb == (torch.arange(L, device=device)[None] < n[:, None])).all()):
            raise ValueError("rows must be right-padded")
        lengths = n.tolist()
        attention_mask = vb.long()

    linear_mask, causal_mask = _masks(body, embeds, attention_mask, text_pos)
    if bidirectional:
        full_mask, extra = None, {"startlux_row_lengths": lengths}
    else:
        full_mask, extra = causal_mask, {}

    order = list(range(cfg.num_hidden_layers)) if layer_map is None else list(layer_map)
    repeated = {idx for idx in order if order.count(idx) > 1}
    hidden = embeds
    with _kernel_guard(device) if bidirectional else nullcontext():
        for idx in order:
            prev = hidden
            layer = body.layers[idx]
            if cfg.layer_types[idx] == "linear_attention":
                hidden = layer(hidden, position_embeddings=position_embeddings,
                               attention_mask=linear_mask, position_ids=text_pos,
                               past_key_values=None, use_cache=False)
            else:
                hidden = layer(hidden, position_embeddings=position_embeddings,
                               attention_mask=full_mask, position_ids=text_pos,
                               past_key_values=None, use_cache=False, **extra)
            if idx in repeated and alpha != 1.0:
                hidden = prev + alpha * (hidden - prev)
    return body.norm(hidden)


def _full_layers(body) -> list:
    cfg = body.config
    return [layer for i, layer in enumerate(body.layers[:cfg.num_hidden_layers])
            if cfg.layer_types[i] == "full_attention"]


def _diff(a, b, what: str) -> float:
    d = float((a.float() - b.float()).abs().max())
    scale = float(a.float().abs().max())
    if not d <= 1e-3 * (1.0 + scale):
        raise RuntimeError(f"bidirectional check failed: {what} (max diff {d}, scale {scale})")
    return d


@torch.no_grad()
def enable(body, tokens: int = 24) -> dict:
    """Route full-attention layers through :func:`_attention` and verify the adaptation.

    Checks: the rerouted stock causal forward reproduces the untouched forward;
    padding never leaks into real tokens; a late token reaches position 0.
    """
    from transformers import AttentionInterface

    cfg = body.config
    if cfg._attn_implementation != "sdpa":
        raise ValueError(f"needs sdpa attention, not {cfg._attn_implementation!r}")
    types = list(cfg.layer_types[:cfg.num_hidden_layers])
    if set(types) - {"full_attention", "linear_attention"}:
        raise ValueError(f"unexpected layer types {sorted(set(types))}")
    for i, layer in enumerate(body.layers[:cfg.num_hidden_layers]):
        if layer.block_type != types[i]:
            raise ValueError(f"layer {i} does not match declared type {types[i]}")

    AttentionInterface.register(IMPL, _attention)
    dev = body.embed_tokens.weight.device
    g = torch.Generator(device="cpu").manual_seed(1234)
    ids = torch.randint(0, min(1000, cfg.vocab_size), (2, tokens), generator=g).to(dev)
    am = torch.ones_like(ids)
    am[1, tokens * 2 // 3:] = 0
    refs = [body(input_ids=ids, attention_mask=m, use_cache=False,
                 return_dict=True).last_hidden_state for m in (None, am)]

    for layer in _full_layers(body):
        if layer.self_attn.config._attn_implementation != IMPL:
            c = copy.copy(layer.self_attn.config)
            c._attn_implementation = IMPL
            layer.self_attn.config = c

    worst = 0.0
    for ref, m in zip(refs, (None, am)):
        worst = max(worst, _diff(ref, body(input_ids=ids, attention_mask=m, use_cache=False,
                                           return_dict=True).last_hidden_state, "rerouted forward"))
        worst = max(worst, _diff(ref, forward(body, ids, valid=m, bidirectional=False), "causal loop"))

    n = int(am[1].sum())
    both = forward(body, ids, valid=am)
    alt = ids.clone()
    alt[1, n:] = (alt[1, n:] + 7) % min(1000, cfg.vocab_size)
    pad = max(_diff(both[0], forward(body, alt, valid=am)[0], "padding of another row"),
              _diff(both[1, :n], forward(body, alt, valid=am)[1, :n], "own padding"))
    late = ids.clone()
    late[0, -1] = (late[0, -1] + 1) % min(1000, cfg.vocab_size)
    reach = float((forward(body, late, valid=am)[0, 0].float() - both[0, 0].float()).abs().max())
    if not reach > 0:
        raise RuntimeError("bidirectional check failed: late token does not reach position 0")

    return {"impl": IMPL, "layers": len(types),
            "full_attention_layers": [i for i, t in enumerate(types) if t == "full_attention"],
            "causal_max_diff": worst, "padding_invariance_max_diff": pad, "late_token_reach": reach}


def load(path: str, device: str = "cuda:0", dtype: str = "bfloat16", check: bool = True) -> "DecisionModel":
    return DecisionModel(path, device=device, dtype=dtype, check=check)


class DecisionModel:
    """A loaded Instant checkpoint plus its tokenizer and read-out configuration."""

    def __init__(self, path: str, device: str = "cuda:0", dtype: str = "bfloat16", check: bool = True):
        import transformers

        self.path = path
        self.device = device
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(path)
        config = transformers.AutoConfig.from_pretrained(path)
        cls = getattr(transformers, config.architectures[0])
        self.model = cls.from_pretrained(path, dtype=getattr(torch, dtype),
                                         device_map={"": device}, attn_implementation="sdpa").eval()
        inner = self.model.model
        self.body = getattr(inner, "language_model", inner)
        self.head = self.model.get_output_embeddings().weight
        self.letters = specs.letter_ids(self.tokenizer, f"{path}/decision_config.json")
        self.temps = specs.temperatures(f"{path}/decision_config.json")
        self.report = enable(self.body) if check else {}

    @property
    def max_options(self) -> int:
        return len(self.letters)

    @torch.no_grad()
    def question(self, state: Any, spec: dict, layer_map: Sequence[int] | None = None,
                 alpha: float = 1.0) -> dict:
        """Score one /v1/systemone question."""
        row = specs.from_systemone(state, spec)
        text = specs.plain_prompt(self.tokenizer, row)
        ids = torch.tensor([self.tokenizer(text, add_special_tokens=False)["input_ids"]],
                           device=self.device)
        h = forward(self.body, ids, layer_map=layer_map, alpha=alpha)
        n = len(row["options"])
        lid = torch.tensor(self.letters[:n], device=self.device)
        logits = (h[0, -1] @ self.head[lid].T).float()
        temp = float(self.temps.get(row["type"], 1.0))
        p = torch.softmax(logits / temp, dim=-1).tolist()
        t = row["type"]
        if t == "choice":
            keys = [o["id"] for o in row["options"]]
            probs = dict(zip(keys, p))
            best = max(probs, key=probs.get)
            return {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
        if t == "noul":
            return {"type": "noul", "noul": float(p[0])}
        legend = {o["id"]: (o["criterion"] or o["id"]) for o in row["options"]}
        expected = sum(i * pi for i, pi in enumerate(p))
        return {"type": "score", "score": expected, "confidence": max(p),
                "probabilities": dict(zip(legend, p)), "legend": legend}

    @torch.no_grad()
    def score(self, state: Any, questions: dict, layer_map: Sequence[int] | None = None,
              alpha: float = 1.0) -> dict:
        """Answer a /v1/systemone request: ``{"answers": {name: answer}}``."""
        if not isinstance(questions, dict) or not questions:
            raise ValueError("questions must be a non-empty object")
        answers = {name: self.question(state, spec, layer_map=layer_map, alpha=alpha)
                   for name, spec in questions.items()}
        return {"answers": answers, "model": self.model.config._name_or_path}
