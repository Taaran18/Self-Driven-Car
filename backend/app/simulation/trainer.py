import random
from collections.abc import Iterator
from dataclasses import dataclass

import neat

from app.simulation.constants import TRACK_CURVINESS, TRACK_LENGTHS, TRACK_WIDTHS
from app.simulation.engine import GenerationRun, Trace, track_ticks
from app.simulation.evolution import build_neat_config, describe_network, evolve, summarize_generation
from app.simulation.road import Road
from app.simulation.schemas import RunConfig


@dataclass
class GenerationStart:
    generation: int
    cars: GenerationRun


@dataclass
class GenerationEnd:
    generation: int
    cars: GenerationRun
    summary: dict
    champion: dict
    evolution: dict | None


class Trainer:
    def __init__(self, run: RunConfig, seed: int):
        self.run = run
        self.seed = seed
        self.track_length = TRACK_LENGTHS[run.track_length]
        self.config = build_neat_config(run, self.track_length)
        self.population = neat.Population(self.config)
        self.track_rng = random.Random(seed)
        self.solved = False

    def build_track(self) -> Road:
        rng = random.Random(self.seed) if self.run.track_mode == "same" else self.track_rng
        return Road(
            rng, TRACK_WIDTHS[self.run.track_width], TRACK_CURVINESS[self.run.track_curviness], self.track_length
        )

    def train(self) -> Iterator[GenerationStart | Trace | GenerationEnd]:
        for generation in range(self.run.max_generations):
            genomes = list(self.population.population.items())
            cars = GenerationRun(genomes, self.config, self.build_track(), track_ticks(self.track_length))
            yield GenerationStart(generation, cars)

            while not cars.done:
                yield cars.tick()
            cars.finish()

            summary = summarize_generation(self.population)
            champion = describe_network(summary["best_genome"], self.config)
            self.solved = summary["best_fitness"] >= self.config.fitness_threshold
            is_last = self.solved or generation == self.run.max_generations - 1
            evolution = None if is_last else evolve(self.population)
            yield GenerationEnd(generation, cars, summary, champion, evolution)

            if self.solved:
                return
