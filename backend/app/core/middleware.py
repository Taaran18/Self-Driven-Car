import json
import logging
import time
import uuid

from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.errors import error_body
from app.core.logging import request_id_var

logger = logging.getLogger("app.http")

_UNSAFE_METHODS = {"POST", "PUT", "PATCH", "DELETE"}
_SECURITY_HEADERS = [
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"strict-origin-when-cross-origin"),
    (b"cache-control", b"no-store"),
]


class RequestContextMiddleware:
    def __init__(self, app: ASGIApp, allowed_origins: list[str]) -> None:
        self.app = app
        self.allowed_origins = set(allowed_origins)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        request_id = headers.get("x-request-id", "")[:64] or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        started = time.perf_counter()
        status_holder = {"code": 500}
        response_started = {"value": False}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_started["value"] = True
                status_holder["code"] = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((b"x-request-id", request_id.encode()))
                message["headers"].extend(_SECURITY_HEADERS)
            await send(message)

        try:
            origin = headers.get("origin")
            if scope["method"] in _UNSAFE_METHODS and origin and origin.rstrip("/") not in self.allowed_origins:
                await self._send_json(
                    send_wrapper,
                    403,
                    error_body("forbidden_origin", "This request came from an origin that isn't allowed."),
                )
                return
            await self.app(scope, receive, send_wrapper)
        except Exception:
            logger.exception("unhandled_error", extra={"path": scope.get("path"), "method": scope["method"]})
            if not response_started["value"]:
                await self._send_json(
                    send_wrapper,
                    500,
                    error_body("internal_error", "Something went wrong on our side. Try again in a moment."),
                )
        finally:
            path = scope.get("path", "")
            if path != "/health":
                logger.info(
                    "request",
                    extra={
                        "method": scope["method"],
                        "path": path,
                        "status": status_holder["code"],
                        "duration_ms": round((time.perf_counter() - started) * 1000, 1),
                    },
                )
            request_id_var.reset(token)

    @staticmethod
    async def _send_json(send: Send, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        await send(
            {
                "type": "http.response.start",
                "status": status,
                "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())],
            }
        )
        await send({"type": "http.response.body", "body": body})
