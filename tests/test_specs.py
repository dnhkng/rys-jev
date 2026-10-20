import json

import pytest

from rys_jev import specs


def test_choice_rendering():
    row = specs.from_systemone({"ticket": "charged twice"}, {
        "type": "choice", "instructions": "Which team?",
        "criteria": {"billing": "Payments and refunds", "technical": "App and login"}})
    assert row["type"] == "choice"
    assert [o["id"] for o in row["options"]] == ["billing", "technical"]
    lines = specs.option_lines(row)
    assert lines == ["A) billing: Payments and refunds", "B) technical: App and login"]
    msgs = specs.messages(row)
    assert msgs[0]["role"] == "system"
    assert "Evidence:" in msgs[1]["content"] and "Options:" in msgs[1]["content"]


def test_noul_defaults_to_yes_no():
    row = specs.from_systemone("comment", {"type": "noul", "instructions": "Is it rude?"})
    assert [o["id"] for o in row["options"]] == ["true", "false"]
    assert specs.option_lines(row) == ["A) true", "B) false"]


def test_noul_keeps_descriptions():
    row = specs.from_systemone("comment", {"type": "noul", "instructions": "Is it rude?",
                                           "criteria": {"true": "Rude", "false": "Civil"}})
    assert specs.option_lines(row) == ["A) true: Rude", "B) false: Civil"]


def test_score_levels_are_lowest_first():
    row = specs.from_systemone("x", {"type": "score", "instructions": "Rate",
                                     "criteria": {"2": "high", "0": "low", "1": "mid"}})
    assert [o["criterion"] for o in row["options"]] == ["low", "mid", "high"]
    assert [o["id"] for o in row["options"]] == ["0", "1", "2"]


def test_bare_ids_are_hidden_when_described():
    row = specs.from_systemone("x", {"type": "choice", "instructions": "Pick",
                                     "criteria": {"A": "Paris", "B": "Rome"}})
    assert specs.option_lines(row) == ["A) Paris", "B) Rome"]


def test_state_is_serialized_as_json():
    row = specs.from_systemone({"a": 1}, {"type": "noul", "instructions": "?"})
    assert json.loads(row["state"]) == {"a": 1}


def test_unknown_type_rejected():
    with pytest.raises(ValueError):
        specs.from_systemone("x", {"type": "ranking", "instructions": "?"})
