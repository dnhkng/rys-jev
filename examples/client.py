"""Minimal client for the /v1/systemone endpoint (standard library only)."""

import json
import sys
import urllib.request

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8090"

payload = {
    "state": {"ticket": "I was charged twice for order #4411 and the app still shows it as unpaid."},
    "questions": {
        "team": {"type": "choice", "instructions": "Which team should handle this ticket?",
                 "criteria": {"billing": "Payments and refunds", "shipping": "Delivery",
                              "technical": "App and login"}},
        "urgent": {"type": "noul", "instructions": "Should this ticket be answered today?"},
        "severity": {"type": "score", "instructions": "How severe is the impact?",
                     "criteria": ["cosmetic", "annoying", "blocks the customer"]},
    },
}

req = urllib.request.Request(URL + "/v1/systemone",
                             data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    print(json.dumps(json.load(r), indent=2))
