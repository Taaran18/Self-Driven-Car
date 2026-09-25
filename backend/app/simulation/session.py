import logging
import random
import threading
import time
from collections.abc import Callable

import numpy as np

from app.simulation.constants import SENSOR_RANGE, TICKS_PER_SECOND
from app.simulation.engine import GenerationRun, Trace
from app.simulation.evolution import describe_network
from app.simulation.schemas import SPEED_FACTORS, RunConfig
from app.simulation.trainer import GenerationEnd, GenerationStart, Trainer

logger = logging.getLogger("app.simulation")

Emit = Callable[[dict, bool], None]

FRAME_INTERVAL = 1 / 45
TURBO_FRAME_INTERVAL = 1 / 12


def _r(value: float, digits: int = 3) -> float:
    return round(float(value), digits)


class SimulationSession:
    def __init__(self, config: RunConfig, emit: Emit, max_seconds: float, idle_seconds: float):
        self.config = config
        self.seed = config.track_seed if config.track_seed is not None else random.randrange(2**31)
        self.emit = emit
        self.max_seconds = max_seconds
        self.idle_seconds = idle_seconds
        self.speed = config.speed
        self.state = "starting"
        self.started_at = time.monotonic()
        self.generations_completed = 0
        self.best_fitness: float | None = None
        self._cond = threading.Condition()
        self._paused = False
        self._paused_at = 0.0
        self._steps = 0
        self._stop_reason: str | None = None
        self._skip = False
        self._generation_started = time.monotonic()
        self._thread = threading.Thread(target=self._run, name="simulation", daemon=True)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def alive(self) -> bool:
        return self._thread.is_alive()

    def start(self) -> None:
        self._thread.start()

    def pause(self) -> None:
        with self._cond:
            if not self._paused:
                self._paused = True
                self._paused_at = time.monotonic()
                self._cond.notify_all()
        self._emit_status()

    def resume(self) -> None:
        with self._cond:
            self._paused = False
            self._steps = 0
            self._cond.notify_all()
        self._emit_status()

    def step(self) -> None:
        with self._cond:
            if not self._paused:
                self._paused = True
                self._paused_at = time.monotonic()
            self._steps += 1
            self._cond.notify_all()
        self._emit_status()

    def skip_generation(self) -> None:
        with self._cond:
            self._skip = True
            self._cond.notify_all()

    def set_speed(self, speed: str) -> None:
        self.speed = speed
        self._emit_status()

    def stop(self, reason: str) -> None:
        with self._cond:
            if self._stop_reason is None:
                self._stop_reason = reason
            self._cond.notify_all()

    def join(self, timeout: float) -> None:
        if self._thread.is_alive():
            self._thread.join(timeout)

    def _emit_status(self) -> None:
        state = "paused" if self._paused else "running"
        self.emit({"type": "status", "state": state, "speed": self.speed}, False)

    def _run(self) -> None:
        status, reason = "completed", "max_generations"
        try:
            trainer = Trainer(self.config, self.seed)
            self.state = "running"
            self._emit_status()
            focus_genome: int | None = None
            last_frame = 0.0
            next_tick = time.monotonic()
            road_version = -1
            cars: GenerationRun | None = None

            for event in trainer.train():
                if self._stop_reason:
                    break
                if isinstance(event, GenerationStart):
                    cars = event.cars
                    focus_genome = None
                    self._generation_started = time.monotonic()
                    self._skip = False
                    self.emit(
                        {
                            "type": "generation_start",
                            "generation": event.generation,
                            "population": len(cars.fleet),
                            "max_ticks": cars.max_ticks,
                            "finish_y": cars.road.finish_y,
                        },
                        False,
                    )
                    continue
                if isinstance(event, GenerationEnd):
                    self._emit_generation(event)
                    continue

                assert cars is not None
                stepped = self._wait_while_paused()
                if self._stop_reason:
                    break
                if self._skip:
                    cars.max_ticks = cars.tick_count
                if self.elapsed > self.max_seconds:
                    self.stop("time_limit")
                    break

                now = time.monotonic()
                factor = SPEED_FACTORS.get(self.speed)
                interval = TURBO_FRAME_INTERVAL if factor is None else FRAME_INTERVAL
                if stepped or now - last_frame >= interval or event.eliminated or event.finished:
                    if cars.road.version != road_version:
                        road_version = cars.road.version
                        self.emit({"type": "road", **cars.road.snapshot()}, False)
                    if event.genome_id != focus_genome and event.genome_id in cars.genomes:
                        focus_genome = event.genome_id
                        network = describe_network(cars.genomes[focus_genome], trainer.config)
                        self.emit({"type": "network", **network}, False)
                    self.emit(self._frame(cars, event, stepped), not stepped)
                    last_frame = now

                if factor is not None and not stepped:
                    next_tick += 1 / (TICKS_PER_SECOND * factor)
                    delay = next_tick - time.monotonic()
                    if delay > 0:
                        with self._cond:
                            self._cond.wait(delay)
                    elif delay < -0.25:
                        next_tick = time.monotonic()
                elif factor is None:
                    time.sleep(0)
                    next_tick = time.monotonic()

            if self._stop_reason:
                status = "stopped" if self._stop_reason == "user" else "interrupted"
                reason = self._stop_reason
            elif trainer.solved:
                reason = "solved"
        except Exception:
            logger.exception("simulation_failed", extra={"seed": self.seed})
            status, reason = "failed", "engine_error"
            self.emit(
                {
                    "type": "error",
                    "code": "engine_error",
                    "message": "The simulation hit an unexpected error and stopped. Start a new run to try again.",
                },
                False,
            )
        finally:
            self.state = "ended"
            self.emit(
                {
                    "type": "ended",
                    "status": status,
                    "reason": reason,
                    "generations_completed": self.generations_completed,
                    "best_fitness": self.best_fitness,
                    "elapsed_seconds": round(self.elapsed, 2),
                },
                False,
            )

    def _wait_while_paused(self) -> bool:
        with self._cond:
            while self._paused and not self._stop_reason:
                if self._steps > 0:
                    self._steps -= 1
                    return True
                if time.monotonic() - self._paused_at > self.idle_seconds:
                    self._stop_reason = "idle"
                    return False
                self._cond.wait(1.0)
        return False

    def _frame(self, cars: GenerationRun, trace: Trace, stepped: bool) -> dict:
        fleet = cars.fleet
        car_rows = np.column_stack(
            [fleet.genome_ids, np.round(fleet.x, 1), np.round(fleet.y, 1), np.round(fleet.rotation, 1), fleet.braking]
        ).tolist()
        return {
            "type": "frame",
            "stepped": stepped,
            "tick": trace.tick,
            "alive": len(fleet),
            "finished": len(cars.finishers),
            "leader_y": _r(cars.leader_y, 1),
            "cars": car_rows,
            "eliminated": trace.eliminated,
            "crossed": trace.finished,
            "trace": {
                "genome_id": trace.genome_id,
                "sense": {
                    "distances": [_r(d, 1) for d in trace.distances],
                    "inputs": [_r(v) for v in trace.inputs],
                    "range": SENSOR_RANGE,
                },
                "think": {
                    "outputs": [_r(v) for v in trace.outputs],
                    "nodes": {str(k): _r(v) for k, v in trace.node_values.items()},
                },
                "decide": trace.decisions,
                "move": {
                    "speed_before": _r(trace.speed_before),
                    "speed": _r(trace.after["speed"]),
                    "rotation_before": _r(trace.rotation_before, 1),
                    "rotation": _r(trace.after["rotation"], 1),
                    "x": _r(trace.after["x"], 1),
                    "y": _r(trace.after["y"], 1),
                },
                "score": {
                    "progress": _r((trace.y_before - trace.after["y"]) / 100),
                    "fitness_before": _r(trace.fitness_before),
                    "fitness": _r(trace.after["fitness"]),
                    "checks": trace.checks,
                    "eliminated": any(trace.checks.values()),
                },
            },
        }

    def _emit_generation(self, event: GenerationEnd) -> None:
        summary = event.summary
        best = summary["best_genome"]
        self.generations_completed = event.generation + 1
        if self.best_fitness is None or summary["best_fitness"] > self.best_fitness:
            self.best_fitness = summary["best_fitness"]
        self.emit(
            {
                "type": "generation",
                "generation": event.generation,
                "best_fitness": _r(summary["best_fitness"]),
                "mean_fitness": _r(summary["mean_fitness"]),
                "std_fitness": _r(summary["std_fitness"]),
                "species_count": summary["species_count"],
                "species": [
                    {**s, "best_fitness": _r(s["best_fitness"]), "mean_fitness": _r(s["mean_fitness"])}
                    for s in summary["species"][:12]
                ],
                "best_genome_id": best.key,
                "best_genome_nodes": len(best.nodes),
                "best_genome_connections": sum(1 for c in best.connections.values() if c.enabled),
                "ticks": event.cars.tick_count,
                "finishers": len(event.cars.finishers),
                "duration_ms": int((time.monotonic() - self._generation_started) * 1000),
                "elapsed_seconds": round(self.elapsed, 2),
                "evolution": event.evolution,
                "champion": {**event.champion, "fitness": _r(summary["best_fitness"])},
            },
            False,
        )
