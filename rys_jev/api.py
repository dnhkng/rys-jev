"""HTTP server exposing the StartLux ``/v1/systemone`` shape."""

from __future__ import annotations

import argparse
import logging

from aiohttp import web

from .engine import DecisionModel

log = logging.getLogger("rys_jev")


async def health(request: web.Request) -> web.Response:
    m = request.app["model"]
    return web.json_response({"status": "ok", "model": str(m.path),
                              "layers": m.body.config.num_hidden_layers})


async def systemone(request: web.Request) -> web.Response:
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "body must be JSON"}, status=400)
    if not isinstance(data, dict):
        return web.json_response({"error": "body must be an object"}, status=400)
    state, questions = data.get("state"), data.get("questions")
    if not isinstance(questions, dict) or not questions:
        return web.json_response({"error": "questions must be a non-empty object"}, status=400)
    try:
        result = request.app["model"].score(state, questions)
    except ValueError as e:
        return web.json_response({"error": str(e)}, status=400)
    return web.json_response(result)


def create_app(model: DecisionModel) -> web.Application:
    app = web.Application(client_max_size=16 * 1024 * 1024)
    app["model"] = model
    app.router.add_get("/health", health)
    app.router.add_post("/v1/systemone", systemone)
    return app


def serve(model_dir: str, host: str = "0.0.0.0", port: int = 8090, device: str = "cuda:0") -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log.info("loading %s on %s", model_dir, device)
    model = DecisionModel(model_dir, device=device)
    log.info("ready: %d layers, checks %s", model.body.config.num_hidden_layers, model.report)
    web.run_app(create_app(model), host=host, port=port)


def main() -> None:
    ap = argparse.ArgumentParser(description="serve a decision model with the /v1/systemone API")
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    serve(a.model, a.host, a.port, a.device)


if __name__ == "__main__":
    main()
