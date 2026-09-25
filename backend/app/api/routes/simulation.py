import asyncio
import contextlib
import json
import logging
import time
import uuid

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from app.core.config import get_settings
from app.core.errors import AppError
from app.core.rate_limit import rate_limiter
from app.core.visitor import CurrentVisitor, VisitorIdentity
from app.db.models import new_id
from app.db.session import DB, get_sessionmaker
from app.services import run_service, usage_service
from app.simulation.manager import ActiveSimulation, CapacityError, SimulationManager, TicketStore
from app.simulation.schemas import SPEED_FACTORS, RunConfig
from app.simulation.session import SimulationSession
from app.simulation.source import code_steps

logger = logging.getLogger("app.simulation")
router = APIRouter(tags=["simulation"])

_QUEUE_SOFT_LIMIT = 12
_RECEIVE_TIMEOUT = 15
_MAX_MESSAGE_BYTES = 8192


@router.post("/api/simulation/ticket")
async def issue_ticket(request: Request, visitor: CurrentVisitor, db: DB):
    rate_limiter.hit(f"ticket:{visitor.ip}", limit=30, window_seconds=60)
    usage = await usage_service.ensure_available(db, visitor.ip)
    manager: SimulationManager = request.app.state.simulations
    tickets: TicketStore = request.app.state.tickets
    return {
        "ticket": tickets.issue(visitor),
        "trial_id": visitor.id,
        "usage": usage,
        "capacity": {"active": manager.active_count, "max": manager.max_concurrent},
    }


@router.websocket("/ws/simulation")
async def simulation_socket(websocket: WebSocket):
    settings = get_settings()
    origin = (websocket.headers.get("origin") or "").rstrip("/")
    if origin not in settings.origins:
        await websocket.close(code=1008)
        return
    ticket = websocket.app.state.tickets.redeem(websocket.query_params.get("ticket"))
    await websocket.accept()
    if ticket is None:
        await websocket.send_json(
            {
                "type": "error",
                "code": "ticket_invalid",
                "message": "Your connection link expired. Press Play again to reconnect.",
            }
        )
        await websocket.close(code=4401)
        return
    connection = SimulationConnection(websocket, websocket.app.state.simulations, ticket.visitor)
    await connection.serve()


class SimulationConnection:
    def __init__(self, websocket: WebSocket, manager: SimulationManager, visitor: VisitorIdentity):
        self.ws = websocket
        self.manager = manager
        self.visitor = visitor
        self.owner_key = visitor.id
        self.usage_event_id: int | None = None
        self.id = uuid.uuid4().hex
        self.settings = get_settings()
        self.loop = asyncio.get_running_loop()
        self.queue: asyncio.Queue[dict] = asyncio.Queue()
        self.active: ActiveSimulation | None = None
        self.closed = False
        self.last_activity = time.monotonic()

    async def serve(self) -> None:
        sender = asyncio.create_task(self._send_loop())
        await self._send(
            {
                "type": "hello",
                "trial_id": self.visitor.id,
                "capacity": {"active": self.manager.active_count, "max": self.manager.max_concurrent},
                "limits": {
                    "max_minutes": self.settings.max_simulation_minutes,
                    "idle_minutes": self.settings.idle_disconnect_minutes,
                },
                "code": code_steps(),
            }
        )
        try:
            await self._receive_loop()
        except WebSocketDisconnect:
            pass
        except Exception:
            logger.exception("simulation_socket_error")
        finally:
            self.closed = True
            if self.active is not None:
                self.active.session.stop("disconnected")
                await asyncio.to_thread(self.active.session.join, 5)
            try:
                await asyncio.wait_for(sender, timeout=10)
            except (TimeoutError, asyncio.CancelledError):
                sender.cancel()

    async def _receive_loop(self) -> None:
        while True:
            try:
                raw = await asyncio.wait_for(self.ws.receive_text(), timeout=_RECEIVE_TIMEOUT)
            except TimeoutError:
                if self._idle_expired():
                    await self._send(
                        {
                            "type": "idle",
                            "message": "We disconnected to save resources because nothing was running. "
                            "Press Play to reconnect.",
                        }
                    )
                    await self.ws.close(code=4000)
                    return
                continue
            if len(raw) > _MAX_MESSAGE_BYTES:
                await self._error("message_too_large", "That message was too large to process.")
                continue
            try:
                message = json.loads(raw)
                if not isinstance(message, dict):
                    raise ValueError
            except ValueError:
                await self._error("bad_message", "We couldn't read that command.")
                continue
            if message.get("type") != "ping":
                self.last_activity = time.monotonic()
            await self._handle(message)

    def _idle_expired(self) -> bool:
        running = self.active is not None and self.active.session.alive
        return not running and time.monotonic() - self.last_activity > self.settings.idle_disconnect_minutes * 60

    async def _handle(self, message: dict) -> None:
        kind = message.get("type")
        session = self.active.session if self.active and self.active.session.alive else None
        if kind == "ping":
            await self._send({"type": "pong"})
        elif kind == "start":
            await self._start(message.get("config") or {})
        elif session is None:
            await self._error("not_running", "Nothing is running right now. Press Play to start a run.")
        elif kind == "pause":
            session.pause()
        elif kind == "resume":
            session.resume()
        elif kind == "step":
            session.step()
        elif kind == "skip":
            session.skip_generation()
        elif kind == "stop":
            session.stop("user")
        elif kind == "speed":
            speed = str(message.get("value"))
            if speed in SPEED_FACTORS:
                session.set_speed(speed)
            else:
                await self._error("bad_speed", "Pick one of the available speeds.")
        else:
            await self._error("unknown_command", "That command isn't supported.")

    async def _start(self, raw_config: dict) -> None:
        if self.active is not None and self.active.session.alive:
            await self._error("already_running", "A run is already in progress. Stop it before starting another.")
            return
        try:
            config = RunConfig.model_validate(raw_config)
        except ValidationError as exc:
            fields = {".".join(str(p) for p in e["loc"]): e["msg"] for e in exc.errors()}
            await self._error("invalid_config", "Some settings are out of range. Check them and try again.", fields)
            return

        session = SimulationSession(
            config,
            self._emit_from_thread,
            max_seconds=self.settings.max_simulation_minutes * 60,
            idle_seconds=self.settings.idle_disconnect_minutes * 60,
        )
        try:
            active = await self.manager.claim(self.owner_key, self.id, self.visitor.id, session)
        except CapacityError:
            await self._error(
                "capacity",
                "All simulation slots are busy right now. Try again in a minute or two.",
                {"max": self.manager.max_concurrent},
            )
            return

        run_id = new_id()
        try:
            async with get_sessionmaker()() as db:
                self.usage_event_id, usage = await usage_service.consume(db, self.visitor, run_id)
                run = await run_service.create_run(
                    db, run_id, self.visitor.id, config.name, {**config.model_dump(), "track_seed": session.seed}
                )
        except AppError as exc:
            self.manager.release(self.owner_key, self.id)
            await self._error(exc.code, exc.message, exc.details)
            return
        except Exception:
            logger.exception("simulation_start_failed")
            self.manager.release(self.owner_key, self.id)
            await self._refund()
            await self._error("start_failed", "We couldn't start the run. Try again in a moment.")
            return

        active.run_id = run.id
        self.active = active
        await self._send(
            {
                "type": "started",
                "run_id": run.id,
                "name": run.name,
                "seed": session.seed,
                "config": config.model_dump(),
                "usage": usage,
            }
        )
        logger.info(
            "simulation_started",
            extra={"run_id": run.id, "visitor_id": self.visitor.id, "ip": self.visitor.ip},
        )
        session.start()

    async def _refund(self) -> None:
        if self.usage_event_id is None:
            return
        event_id, self.usage_event_id = self.usage_event_id, None
        await self._persist(usage_service.refund, event_id)

    def _emit_from_thread(self, message: dict, droppable: bool) -> None:
        with contextlib.suppress(RuntimeError):
            self.loop.call_soon_threadsafe(self._enqueue, message, droppable)

    def _enqueue(self, message: dict, droppable: bool) -> None:
        if droppable and (self.closed or self.queue.qsize() > _QUEUE_SOFT_LIMIT):
            return
        self.queue.put_nowait(message)

    async def _send_loop(self) -> None:
        while True:
            if self.closed and self.queue.empty() and (self.active is None or not self.active.session.alive):
                return
            try:
                message = await asyncio.wait_for(self.queue.get(), timeout=1)
            except TimeoutError:
                continue
            active = self.active
            run_id = active.run_id if active else None
            kind = message.get("type")
            if kind == "generation" and run_id:
                await self._persist(run_service.record_generation, run_id, message, message.get("champion"))
            elif kind == "ended":
                if run_id:
                    await self._persist(
                        run_service.finish_run, run_id, message["status"], message["reason"], message["elapsed_seconds"]
                    )
                refunded = message["status"] == "failed" and message["generations_completed"] == 0
                if refunded:
                    await self._refund()
                self.usage_event_id = None
                usage = None
                with contextlib.suppress(Exception):
                    async with get_sessionmaker()() as db:
                        usage = await usage_service.summary(db, self.visitor.ip)
                message = {**message, "run_id": run_id, "refunded": refunded, "usage": usage}
                if active is not None:
                    self.manager.release(self.owner_key, self.id)
                logger.info("simulation_ended", extra={"run_id": run_id, "reason": message["reason"]})
            await self._send(message)

    async def _persist(self, fn, *args) -> None:
        try:
            async with get_sessionmaker()() as db:
                await fn(db, *args)
        except Exception:
            logger.exception("simulation_persist_failed")

    async def _send(self, message: dict) -> None:
        if self.closed:
            return
        try:
            await self.ws.send_text(json.dumps(message, separators=(",", ":")))
        except Exception:
            self.closed = True

    async def _error(self, code: str, message: str, details: dict | None = None) -> None:
        payload = {"type": "error", "code": code, "message": message}
        if details:
            payload["details"] = details
        await self._send(payload)
