import math
from dataclasses import dataclass, field

import neat
import numpy as np

from app.simulation.constants import (
    ACCELERATION,
    ACTIVATION_THRESHOLD,
    BRAKING,
    CAR_LENGTH,
    CAR_WIDTH,
    ELIMINATION_PENALTY,
    FALL_BEHIND_LIMIT,
    FRICTION,
    GRACE_TICKS,
    MAX_SPEED,
    MIN_AVERAGE_SPEED,
    MIN_SPEED,
    PROGRESS_SCALE,
    SENSOR_ANGLES,
    SENSOR_RANGE,
    START_SPEED,
    TURN_RATE,
)
from app.simulation.geometry import boxes_hit_walls, car_corners, cast_rays
from app.simulation.road import Road

ELIMINATION_REASONS = ("crashed", "fell_behind", "reversing", "stalled")


@dataclass
class Fleet:
    genome_ids: np.ndarray
    x: np.ndarray
    y: np.ndarray
    rotation: np.ndarray
    speed: np.ndarray
    fitness: np.ndarray
    braking: np.ndarray

    @classmethod
    def launch(cls, genome_ids: list[int]) -> "Fleet":
        n = len(genome_ids)
        return cls(
            genome_ids=np.asarray(genome_ids),
            x=np.zeros(n),
            y=np.zeros(n),
            rotation=np.zeros(n),
            speed=np.full(n, START_SPEED),
            fitness=np.zeros(n),
            braking=np.zeros(n, dtype=bool),
        )

    def __len__(self) -> int:
        return len(self.genome_ids)

    def keep(self, mask: np.ndarray) -> None:
        for name in ("genome_ids", "x", "y", "rotation", "speed", "fitness", "braking"):
            setattr(self, name, getattr(self, name)[mask])


@dataclass
class Decisions:
    accelerate: np.ndarray
    brake: np.ndarray
    turn_left: np.ndarray
    turn_right: np.ndarray


def sense(fleet: Fleet, road: Road) -> tuple[np.ndarray, np.ndarray]:
    walls = road.walls_between(fleet.y.min() - SENSOR_RANGE, fleet.y.max() + SENSOR_RANGE)
    origins = np.column_stack([fleet.x, fleet.y])
    ray_angles = fleet.rotation[:, None] + SENSOR_ANGLES[None, :]
    distances = cast_rays(origins, ray_angles, SENSOR_RANGE, *walls)
    closeness = 1 - distances / SENSOR_RANGE
    inputs = np.column_stack([closeness, fleet.speed / MAX_SPEED])
    return inputs, distances


def think(networks: list, inputs: np.ndarray) -> np.ndarray:
    return np.array([network.activate(row.tolist()) for network, row in zip(networks, inputs, strict=False)])


def decide(outputs: np.ndarray) -> Decisions:
    accelerate, brake, left, right = outputs.T
    active = outputs > ACTIVATION_THRESHOLD
    return Decisions(
        accelerate=active[:, 0] & (accelerate > brake),
        brake=active[:, 1] & (brake > accelerate),
        turn_left=active[:, 2] & (left > right),
        turn_right=active[:, 3] & (right > left),
    )


def move(fleet: Fleet, decisions: Decisions) -> None:
    acceleration = np.full(len(fleet), FRICTION)
    acceleration[decisions.accelerate] = ACCELERATION
    acceleration[decisions.brake] = -BRAKING
    fleet.rotation += TURN_RATE * (decisions.turn_right.astype(float) - decisions.turn_left.astype(float))
    fleet.speed = np.clip(fleet.speed + acceleration, 0, MAX_SPEED)
    heading = np.radians(fleet.rotation)
    fleet.x += fleet.speed * np.sin(heading)
    fleet.y -= fleet.speed * np.cos(heading)
    fleet.braking = decisions.brake


def score(fleet: Fleet, previous_y: np.ndarray, road: Road, leader_y: float, tick: int) -> np.ndarray:
    walls = road.walls_between(fleet.y.min() - CAR_LENGTH, fleet.y.max() + CAR_LENGTH)
    corners = car_corners(fleet.x, fleet.y, fleet.rotation, CAR_LENGTH, CAR_WIDTH)
    checks = np.stack(
        [
            boxes_hit_walls(corners, *walls),
            fleet.y > leader_y + FALL_BEHIND_LIMIT,
            fleet.y > previous_y,
            fleet.speed < MIN_SPEED,
        ]
    )
    if tick <= GRACE_TICKS:
        checks[:] = False
    eliminated = checks.any(axis=0)
    progress = (previous_y - fleet.y) / PROGRESS_SCALE
    fleet.fitness += np.where(eliminated, -ELIMINATION_PENALTY, progress)
    return checks


@dataclass
class Trace:
    tick: int
    focus: int
    genome_id: int
    speed_before: float
    rotation_before: float
    fitness_before: float
    y_before: float
    inputs: list[float] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    outputs: list[float] = field(default_factory=list)
    decisions: dict[str, bool] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)
    after: dict[str, float] = field(default_factory=dict)
    node_values: dict[int, float] = field(default_factory=dict)
    eliminated: list[dict] = field(default_factory=list)
    finished: list[dict] = field(default_factory=list)

    @classmethod
    def begin(cls, fleet: Fleet, focus: int, tick: int) -> "Trace":
        return cls(
            tick=tick,
            focus=focus,
            genome_id=int(fleet.genome_ids[focus]),
            speed_before=float(fleet.speed[focus]),
            rotation_before=float(fleet.rotation[focus]),
            fitness_before=float(fleet.fitness[focus]),
            y_before=float(fleet.y[focus]),
        )

    def record(self, fleet, inputs, distances, outputs, decisions, checks, network) -> None:
        i = self.focus
        self.inputs = inputs[i].tolist()
        self.distances = distances[i].tolist()
        self.outputs = outputs[i].tolist()
        self.decisions = {name: bool(getattr(decisions, name)[i]) for name in Decisions.__dataclass_fields__}
        self.checks = {reason: bool(checks[k, i]) for k, reason in enumerate(ELIMINATION_REASONS)}
        self.after = {
            "x": float(fleet.x[i]),
            "y": float(fleet.y[i]),
            "speed": float(fleet.speed[i]),
            "rotation": float(fleet.rotation[i]),
            "fitness": float(fleet.fitness[i]),
        }
        self.node_values = dict(network.values)


class GenerationRun:
    def __init__(self, genomes: list, neat_config: neat.Config, road: Road, max_ticks: int):
        self.genomes = dict(genomes)
        self.networks = [neat.nn.FeedForwardNetwork.create(g, neat_config) for _, g in genomes]
        self.fleet = Fleet.launch([genome_id for genome_id, _ in genomes])
        self.road = road
        self.max_ticks = max_ticks
        self.tick_count = 0
        self.leader_y = 0.0
        self.results: dict[int, float] = {}
        self.finishers: list[int] = []
        for genome in self.genomes.values():
            genome.fitness = 0.0

    @property
    def done(self) -> bool:
        return len(self.fleet) == 0 or self.tick_count >= self.max_ticks

    @property
    def focus_index(self) -> int:
        return int(np.argmax(self.fleet.fitness))

    def tick(self) -> Trace:
        self.tick_count += 1
        fleet = self.fleet
        trace = Trace.begin(fleet, self.focus_index, self.tick_count)
        previous_y = fleet.y.copy()

        inputs, distances = sense(fleet, self.road)
        outputs = think(self.networks, inputs)
        decisions = decide(outputs)
        move(fleet, decisions)
        checks = score(fleet, previous_y, self.road, self.leader_y, self.tick_count)

        trace.record(fleet, inputs, distances, outputs, decisions, checks, self.networks[trace.focus])
        self.retire(checks, trace)
        self.follow_leader()
        return trace

    def follow_leader(self) -> None:
        if len(self.fleet):
            self.leader_y = float(self.fleet.y.min())
            self.road.extend_to(self.leader_y)
            self.road.trim_behind(float(self.fleet.y.max()))

    def finish(self) -> None:
        for genome_id, fitness in zip(self.fleet.genome_ids, self.fleet.fitness, strict=False):
            self.results[int(genome_id)] = float(fitness)
        self.fleet.keep(np.zeros(len(self.fleet), dtype=bool))
        self.networks = []
        for genome_id, genome in self.genomes.items():
            genome.fitness = self.results.get(genome_id, 0.0)

    def retire(self, checks: np.ndarray, trace: Trace) -> None:
        eliminated = checks.any(axis=0)
        crossed = (self.fleet.y <= self.road.finish_y) & ~eliminated
        leaving = eliminated | crossed
        if not leaving.any():
            return
        fleet = self.fleet
        for i in np.flatnonzero(leaving):
            genome_id = int(fleet.genome_ids[i])
            self.results[genome_id] = float(fleet.fitness[i])
            car = {
                "id": genome_id,
                "x": round(float(fleet.x[i]), 1),
                "y": round(float(fleet.y[i]), 1),
                "rotation": round(float(fleet.rotation[i]), 1),
            }
            if crossed[i]:
                self.finishers.append(genome_id)
                trace.finished.append(car)
            else:
                car["reason"] = ELIMINATION_REASONS[int(np.argmax(checks[:, i]))]
                trace.eliminated.append(car)
        stay = ~leaving
        fleet.keep(stay)
        self.networks = [n for n, s in zip(self.networks, stay, strict=False) if s]


def track_ticks(length: float) -> int:
    return int(math.ceil(length / MIN_AVERAGE_SPEED))
