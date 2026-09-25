import json

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from tests.conftest import ORIGIN

FAST_RUN = {"population_size": 10, "max_generations": 2, "track_length": "short", "speed": "max", "name": "Test Run"}


def collect_until_ended(ws, limit=20000):
    messages = []
    for _ in range(limit):
        message = json.loads(ws.receive_text())
        messages.append(message)
        if message["type"] == "ended":
            return messages
    raise AssertionError("run did not end")


def open_socket(client: TestClient):
    ticket = client.post("/api/simulation/ticket").json()["ticket"]
    return client.websocket_connect(f"/ws/simulation?ticket={ticket}", headers={"origin": ORIGIN})


def test_run_streams_frames_and_code(client: TestClient):
    with open_socket(client) as ws:
        hello = json.loads(ws.receive_text())
        assert hello["type"] == "hello"
        assert hello["trial_id"]
        step_ids = [s["id"] for s in hello["code"]]
        assert step_ids == ["train", "tick", "sense", "think", "decide", "move", "score", "evaluate", "evolve"]
        assert "def sense" in hello["code"][2]["snippets"][0]["code"]

        ws.send_text(json.dumps({"type": "start", "config": FAST_RUN}))
        messages = collect_until_ended(ws)

    kinds = {m["type"] for m in messages}
    assert {"started", "status", "generation_start", "road", "network", "frame", "generation", "ended"} <= kinds
    started = next(m for m in messages if m["type"] == "started")
    assert started["usage"]["used_today"] == 1
    frame = next(m for m in messages if m["type"] == "frame")
    assert set(frame["trace"]) == {"genome_id", "sense", "think", "decide", "move", "score"}
    assert len(frame["trace"]["sense"]["inputs"]) == 9
    generations = [m for m in messages if m["type"] == "generation"]
    assert len(generations) >= 1
    assert generations[0]["evolution"] is None or "offspring" in generations[0]["evolution"]
    ended = messages[-1]
    assert ended["status"] == "completed"
    assert ended["reason"] in {"max_generations", "solved"}


def test_pause_step_and_stop(client: TestClient):
    with open_socket(client) as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "start", "config": {**FAST_RUN, "speed": "1", "max_generations": 5}}))
        ws.send_text(json.dumps({"type": "pause"}))
        ws.send_text(json.dumps({"type": "step"}))
        stepped = None
        for _ in range(500):
            message = json.loads(ws.receive_text())
            if message["type"] == "frame" and message["stepped"]:
                stepped = message
                break
        assert stepped is not None
        ws.send_text(json.dumps({"type": "stop"}))
        messages = collect_until_ended(ws)
    assert messages[-1]["status"] == "stopped"
    assert messages[-1]["reason"] == "user"


def test_rejects_bad_ticket_and_origin(client: TestClient):
    with client.websocket_connect("/ws/simulation?ticket=bogus", headers={"origin": ORIGIN}) as ws:
        message = json.loads(ws.receive_text())
        assert message["code"] == "ticket_invalid"
    ticket = client.post("/api/simulation/ticket").json()["ticket"]
    try:
        with client.websocket_connect(f"/ws/simulation?ticket={ticket}", headers={"origin": "https://evil.example"}):
            raise AssertionError("should not connect")
    except WebSocketDisconnect as exc:
        assert exc.code == 1008


def test_invalid_config_is_reported(client: TestClient):
    with open_socket(client) as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "start", "config": {"population_size": 5000}}))
        message = json.loads(ws.receive_text())
        assert message["code"] == "invalid_config"
        assert "population_size" in message["details"]
        ws.send_text(json.dumps({"type": "pause"}))
        assert json.loads(ws.receive_text())["code"] == "not_running"
