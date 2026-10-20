"""Prompt format and question specs for StartLux-Decision Instant models.

The models were trained on one chat prompt per question: a fixed system line, then
Evidence / Question / Options blocks, with the assistant turn opened and the empty
think block closed (thinking off).  The answer is read from the next-token
probabilities of the option letters directly after that prompt, so nothing is
generated.  These strings are part of the trained format -- changing them changes
the answers.
"""

from __future__ import annotations

import json
import re
import string
from typing import Any

SYSTEM = "Apply the criterion to the evidence. Choose exactly one listed option. Answer with its letter only."
LETTERS = string.ascii_uppercase
TYPES = ("choice", "noul", "score")
MAX_OPTIONS = 26
NOUL_INSTRUCTIONS = "Which answer fits the evidence?"
THINK_OFF_SUFFIX = "<think>\n\n</think>\n\n"

_BARE = re.compile(r"^(?:\(?[A-Za-z][\).]?|\(?\d{1,3}[\).]?|option[_ ]?\d{1,3}|opt[_ ]?\d{1,3})$", re.I)


def _norm(s: Any) -> str:
    return re.sub(r"[\s_\-]+", " ", str(s)).strip().lower()


def state_text(state: Any) -> str:
    """Evidence as text: a string as-is, anything else as compact JSON."""
    if isinstance(state, str):
        return state
    if state is None or state == {} or state == []:
        return ""
    return json.dumps(state, ensure_ascii=False)


def value_text(v: Any) -> str | None:
    if v is None:
        return None
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = s.strip()
    return s or None


def score_keys(criteria: dict) -> list:
    try:
        return sorted(criteria, key=lambda k: float(k))
    except (TypeError, ValueError):
        return list(criteria)


def from_systemone(state: Any, spec: dict) -> dict:
    """One /v1/systemone question -> a rendered row."""
    t = spec.get("type", "choice")
    t = "noul" if t == "bool" else t
    if t not in TYPES:
        raise ValueError(f"unknown question type {t!r}: use choice, noul or score")
    crit = spec.get("criteria", spec.get("options"))
    ins = value_text(spec.get("instructions", spec.get("question"))) or ""
    if t == "choice":
        if isinstance(crit, (list, tuple)):
            crit = {str(c): None for c in crit}
        if not isinstance(crit, dict):
            raise ValueError("a choice question needs criteria: {option: description} or a list of options")
        opts = [{"id": str(k), "criterion": value_text(v)} for k, v in crit.items()]
    elif t == "score":
        if isinstance(crit, dict):
            crit = [crit[k] for k in score_keys(crit)]
        if not isinstance(crit, (list, tuple)):
            raise ValueError("a score question needs criteria: a list of levels, lowest first")
        opts = [{"id": str(i), "criterion": value_text(c)} for i, c in enumerate(crit)]
    else:
        c = crit if isinstance(crit, dict) else {}
        opts = [{"id": "true", "criterion": value_text(c.get("true"))},
                {"id": "false", "criterion": value_text(c.get("false"))}]
        if not ins:
            ins = NOUL_INSTRUCTIONS
    if len(opts) > MAX_OPTIONS:
        raise ValueError(f"{len(opts)} options exceed the {MAX_OPTIONS}-letter readout")
    return {"type": t, "state": state_text(state), "instructions": ins, "options": opts}


def option_lines(row: dict) -> list[str]:
    opts = row["options"]
    hide = all(o.get("criterion") for o in opts) and all(_BARE.match(o["id"]) for o in opts)
    lines = []
    for k, o in enumerate(opts):
        name = o["id"]
        crit = o.get("criterion")
        if hide and crit and _norm(crit) != _norm(name):
            body = crit
        elif crit is None or not crit.strip() or _norm(crit) == _norm(name):
            body = name
        else:
            body = f"{name}: {crit}"
        lines.append(f"{LETTERS[k]}) {body}")
    return lines


def messages(row: dict) -> list[dict]:
    state = row["state"].strip() or "(none)"
    content = ("Evidence:\n" + state + "\n\nQuestion: " + row["instructions"].strip()
               + "\nOptions:\n" + "\n".join(option_lines(row)))
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


def plain_prompt(tokenizer: Any, row: dict) -> str:
    text = tokenizer.apply_chat_template(messages(row), tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
    if not text.endswith(THINK_OFF_SUFFIX):
        raise ValueError("this tokenizer does not render the thinking-off assistant prefix; "
                         f"expected it to end with {THINK_OFF_SUFFIX!r}")
    return text


def letter_ids(tokenizer: Any, path: str | None = None) -> list[int]:
    """The 26 letter token ids, A to Z.  Read from decision_config.json when available."""
    if path:
        try:
            cfg = json.load(open(path))
            ids = cfg.get("letter_token_ids")
            if isinstance(ids, list) and len(ids) == 26:
                return [int(x) for x in ids]
        except (OSError, ValueError):
            pass
    probe = plain_prompt(tokenizer, {"type": "noul", "state": "s", "instructions": "q",
                                     "options": [{"id": "true", "criterion": None},
                                                 {"id": "false", "criterion": None}]})
    base = tokenizer(probe, add_special_tokens=False)["input_ids"]
    ids = []
    for letter in LETTERS:
        t = tokenizer(letter, add_special_tokens=False)["input_ids"]
        if len(t) != 1 or tokenizer(probe + letter, add_special_tokens=False)["input_ids"] != base + t:
            raise ValueError(f"letter {letter} is not a single token after the prompt")
        ids.append(t[0])
    if len(set(ids)) != 26:
        raise ValueError("letter token ids are not distinct")
    return ids


def temperatures(path: str | None = None) -> dict:
    """Per-type softmax temperatures from decision_config.json (1.0 when absent)."""
    out = {"choice": 1.0, "noul": 1.0, "score": 1.0}
    if path:
        try:
            cfg = json.load(open(path))
            for k, v in (cfg.get("temperature_by_type") or {}).items():
                out[k] = float(v)
        except (OSError, ValueError):
            pass
    return out
