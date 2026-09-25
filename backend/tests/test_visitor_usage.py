import json

from fastapi.testclient import TestClient

from tests.conftest import ORIGIN, browser_id

FAST_RUN = {"population_size": 10, "max_generations": 1, "track_length": "short", "speed": "max"}


def run_once(client: TestClient) -> list[dict]:
    response = client.post("/api/simulation/ticket")
    assert response.status_code == 200, response.text
    ticket = response.json()["ticket"]
    with client.websocket_connect(f"/ws/simulation?ticket={ticket}", headers={"origin": ORIGIN}) as ws:
        json.loads(ws.receive_text())
        ws.send_text(json.dumps({"type": "start", "config": FAST_RUN}))
        messages = []
        while True:
            message = json.loads(ws.receive_text())
            messages.append(message)
            if message["type"] in {"ended", "error"}:
                return messages


def test_health(client: TestClient):
    assert client.get("/health").json()["status"] == "ok"


def test_missing_or_invalid_visitor_id(make_client):
    client = make_client()
    response = client.get("/api/me", headers={"x-visitor-id": "not-a-uuid"})
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "missing_visitor"


def test_workspace_is_per_browser_and_network(make_client):
    shared = browser_id()
    home = make_client(shared, ip="203.0.113.10")
    again = make_client(shared, ip="203.0.113.10")
    cafe = make_client(shared, ip="192.0.2.44")
    neighbour = make_client(browser_id(), ip="203.0.113.10")
    ids = {c.get("/api/me").json()["id"] for c in (home, cafe, neighbour)}
    assert len(ids) == 3
    assert home.get("/api/me").json()["id"] == again.get("/api/me").json()["id"]


def test_usage_summary_defaults(client: TestClient):
    usage = client.get("/api/me/usage").json()
    assert usage["runs_per_day"] == 5
    assert usage["runs_per_week"] == 20
    assert usage["left_today"] == 5
    assert usage["day_resets_at"] <= usage["week_resets_at"]


def test_daily_limit_is_counted_per_ip(make_client):
    first = make_client(ip="203.0.113.50")
    second = make_client(ip="203.0.113.50")
    other_network = make_client(ip="203.0.113.99")
    for _ in range(3):
        assert run_once(first)[-1]["type"] == "ended"
    for _ in range(2):
        ended = run_once(second)[-1]
        assert ended["type"] == "ended"
    assert ended["usage"]["left_today"] == 0

    blocked = first.post("/api/simulation/ticket")
    assert blocked.status_code == 429
    body = blocked.json()["error"]
    assert body["code"] == "trial_limit"
    assert "midnight UTC" in body["message"]
    assert body["details"]["usage"]["used_today"] == 5

    assert other_network.post("/api/simulation/ticket").status_code == 200
    assert first.get("/api/runs").json()["total"] == 3
    assert second.get("/api/runs").json()["total"] == 2


def test_weekly_limit(make_client, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("TRIAL_RUNS_PER_DAY", "10")
    monkeypatch.setenv("TRIAL_RUNS_PER_WEEK", "2")
    get_settings.cache_clear()
    client = make_client(ip="203.0.113.77")
    run_once(client)
    run_once(client)
    blocked = client.post("/api/simulation/ticket")
    assert blocked.status_code == 429
    assert "Monday" in blocked.json()["error"]["message"]


def test_runs_are_private_to_each_workspace(make_client):
    alice = make_client()
    bob = make_client()
    run_id = run_once(alice)[-1]["run_id"]
    assert alice.get(f"/api/runs/{run_id}").status_code == 200
    assert bob.get(f"/api/runs/{run_id}").status_code == 404
    assert bob.delete(f"/api/runs/{run_id}").status_code == 404
    assert bob.get("/api/runs").json()["total"] == 0


def test_run_history_rename_export_delete(client: TestClient):
    run_id = run_once(client)[-1]["run_id"]
    detail = client.get(f"/api/runs/{run_id}").json()
    assert detail["status"] == "completed"
    assert detail["generations"] and detail["champion"]["nodes"]
    assert client.patch(f"/api/runs/{run_id}", json={"name": "  Night   Drive "}).json()["name"] == "Night Drive"
    assert client.get("/api/runs?q=night").json()["total"] == 1

    overview = client.get("/api/me/overview").json()
    assert overview["total_runs"] == 1 and overview["best_run_id"] == run_id

    export = client.get("/api/me/export")
    assert export.json()["runs"][0]["id"] == run_id

    assert client.delete("/api/me/runs").json() == {"deleted": 1}
    assert client.get("/api/runs").json()["total"] == 0
    assert client.delete("/api/me").status_code == 204
    assert client.get("/api/me/usage").json()["used_today"] == 1


def test_refund_removes_usage(client: TestClient):
    import asyncio

    from app.db.session import get_sessionmaker
    from app.services import usage_service

    run_once(client)
    assert client.get("/api/me/usage").json()["used_today"] == 1

    async def refund_all():
        async with get_sessionmaker()() as db:
            from sqlalchemy import select

            from app.db.models import UsageEvent

            for event_id in await db.scalars(select(UsageEvent.id)):
                await usage_service.refund(db, event_id)

    asyncio.run(refund_all())
    assert client.get("/api/me/usage").json()["used_today"] == 0


def test_admin_report(make_client):
    user = make_client(ip="203.0.113.5")
    run_once(user)
    admin = make_client()
    assert admin.get("/api/admin/usage").status_code == 401
    assert admin.get("/api/admin/usage", headers={"authorization": "Bearer nope"}).status_code == 401
    report = admin.get("/api/admin/usage", headers={"authorization": "Bearer test-admin-token"}).json()
    assert report["totals"]["runs"] == 1
    assert report["ips"][0]["ip"] == "203.0.113.5"
    assert report["ips"][0]["runs_today"] == 1
    assert report["visitors"][0]["runs"] == 1


def test_admin_disabled_without_token(make_client, monkeypatch):
    from app.core.config import get_settings

    monkeypatch.setenv("ADMIN_TOKEN", "")
    get_settings.cache_clear()
    assert make_client().get("/api/admin/usage").status_code == 404


def test_purge_clears_old_usage_and_ip(client: TestClient):
    import asyncio
    from datetime import timedelta

    from sqlalchemy import select, update

    from app.db.models import UsageEvent, Visitor, utcnow
    from app.db.session import get_sessionmaker
    from app.services import usage_service

    run_once(client)
    old = utcnow() - timedelta(days=120)

    async def age_and_purge():
        async with get_sessionmaker()() as db:
            await db.execute(update(UsageEvent).values(created_at=old))
            await db.execute(update(Visitor).values(last_seen_at=old))
            await db.commit()
            purged = await usage_service.purge_old_usage(db)
            visitor = (await db.scalars(select(Visitor))).one()
            return purged, visitor.last_ip

    purged, last_ip = asyncio.run(age_and_purge())
    assert purged == 1
    assert last_ip is None


def test_concurrent_first_requests_do_not_crash(client: TestClient):
    import asyncio

    from app.core.visitor import identify
    from app.db.session import get_sessionmaker
    from app.services import usage_service

    identity = identify("aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee", "203.0.113.200", "pytest")

    async def race():
        async with get_sessionmaker()() as first, get_sessionmaker()() as second:
            original_get = second.get

            async def stale_get(*args, **kwargs):
                second.get = original_get
                return None

            second.get = stale_get
            await usage_service.touch_visitor(first, identity)
            visitor = await usage_service.touch_visitor(second, identity)
            return visitor.id

    assert asyncio.run(race()) == identity.id


def test_unexpected_errors_keep_cors_headers(client: TestClient):
    def boom():
        raise RuntimeError("boom")

    client.app.add_api_route("/api/boom", boom)
    response = client.get("/api/boom")
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "internal_error"
    assert response.headers.get("access-control-allow-origin") == "http://localhost:3000"
