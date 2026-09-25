import uuid

import pytest
from fastapi.testclient import TestClient

ORIGIN = "http://localhost:3000"


def browser_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture
def make_client(tmp_path, monkeypatch):
    monkeypatch.setenv("APP_ENV", "test")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FRONTEND_URL", ORIGIN)
    monkeypatch.setenv("TRUST_PROXY", "true")
    monkeypatch.setenv("ADMIN_TOKEN", "test-admin-token")
    from app.core.config import get_settings
    from app.core.rate_limit import rate_limiter
    from app.db import session as db_session

    get_settings.cache_clear()
    rate_limiter.reset()
    db_session._engine = None
    db_session._sessionmaker = None
    from app.main import create_app

    app = create_app()
    clients = []

    def factory(visitor: str | None = None, ip: str = "203.0.113.10") -> TestClient:
        headers = {"origin": ORIGIN, "x-forwarded-for": f"198.51.100.1, {ip}"}
        headers["x-visitor-id"] = visitor or browser_id()
        client = TestClient(app, headers=headers)
        client.__enter__()
        clients.append(client)
        return client

    yield factory
    for client in reversed(clients):
        client.__exit__(None, None, None)
    get_settings.cache_clear()


@pytest.fixture
def client(make_client) -> TestClient:
    return make_client()
