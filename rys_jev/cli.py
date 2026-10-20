"""Command line entry point: serve, score, materialize, info."""

from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    from . import __version__

    ap = argparse.ArgumentParser(prog="rys-jev", description="RYS-JEV decision model tools")
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("serve", help="start the /v1/systemone HTTP server")
    s.add_argument("--model", required=True)
    s.add_argument("--host", default="0.0.0.0")
    s.add_argument("--port", type=int, default=8090)
    s.add_argument("--device", default="cuda:0")

    q = sub.add_parser("score", help="score one request from the command line")
    q.add_argument("--model", required=True)
    q.add_argument("--state", required=True, help="JSON string or @file")
    q.add_argument("--questions", required=True, help="JSON string or @file")
    q.add_argument("--device", default="cuda:0")
    q.add_argument("--span", default="", help="runtime RYS span i-j (base 64-layer model only)")
    q.add_argument("--alpha", type=float, default=1.25)

    m = sub.add_parser("materialize", help="bake an RYS duplication + alpha into a new checkpoint")
    m.add_argument("--src", required=True)
    m.add_argument("--dst", required=True)
    m.add_argument("--span", default="41-49")
    m.add_argument("--alpha", type=float, default=1.25)
    m.add_argument("--shard-gb", type=float, default=4.5)

    i = sub.add_parser("info", help="load a checkpoint and print its attention checks")
    i.add_argument("--model", required=True)
    i.add_argument("--device", default="cuda:0")

    a = ap.parse_args()
    if a.cmd == "serve":
        from .api import serve
        serve(a.model, a.host, a.port, a.device)
    elif a.cmd == "materialize":
        from .materialize import materialize
        materialize(a.src, a.dst, a.span, a.alpha, a.shard_gb)
    elif a.cmd == "info":
        from .engine import DecisionModel
        m = DecisionModel(a.model, device=a.device)
        print(json.dumps({"model": a.model, "layers": m.body.config.num_hidden_layers,
                          "checks": m.report}, indent=1))
    else:
        from .engine import DecisionModel

        def load(v):
            if v.startswith("@"):
                return json.load(open(v[1:]))
            return json.loads(v)

        layer_map = None
        if a.span:
            from .materialize import parse_span
            i0, j0 = parse_span(a.span)
            n = 64
            layer_map = list(range(0, j0)) + list(range(i0, n))
        model = DecisionModel(a.model, device=a.device)
        print(json.dumps(model.score(load(a.state), load(a.questions), layer_map=layer_map,
                                     alpha=a.alpha), indent=1))


if __name__ == "__main__":
    sys.exit(main())
