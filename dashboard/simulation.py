import sys
import os
import pygame as py
import neat
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.car import Car
from src.road import Road
from src.world import World
from src.NNdraw import NN
from config.config_variables import *
from dashboard.reporter import NEATReporter

os.environ["SDL_VIDEODRIVER"] = "dummy"
STOP_SIMULATION = False


def set_stop_simulation(value):
    global STOP_SIMULATION
    STOP_SIMULATION = value


def get_stop_simulation():
    global STOP_SIMULATION
    return STOP_SIMULATION


class SimulationRunner:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_path,
        )
        self.population = neat.Population(self.config)
        self.reporter = NEATReporter()
        self.population.add_reporter(self.reporter)

        self.population.add_reporter(neat.StdOutReporter(True))

        py.font.init()
        self.bg = py.Surface((WIN_WIDTH, WIN_HEIGHT))
        self.bg.fill(GRAY)

    def eval_genomes(self, genomes, config):
        global GEN

        if get_stop_simulation():

            raise KeyboardInterrupt("Stopped by user")

        nets = []
        ge = []
        cars = []
        t = 0

        world = World(STARTING_POS, WIN_WIDTH, WIN_HEIGHT)
        world.win.blit(self.bg, (0, 0))

        NNs = []

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)
            nets.append(net)
            cars.append(Car(0, 0, 0))
            g.fitness = 0
            ge.append(g)
            NNs.append(NN(config, g, (90, 210)))

        road = Road(world)
        clock = py.time.Clock()

        run = True
        while run:
            if get_stop_simulation():
                run = False
                py.quit()
                raise KeyboardInterrupt("Stopped by user")

            t += 1

            clock.tick(FPS)
            world.updateScore(0)

            for event in py.event.get():
                if event.type == py.QUIT:
                    run = False
                    py.quit()
                    return

            (xb, yb) = (0, 0)
            i = 0
            while i < len(cars):
                car = cars[i]

                input = car.getInputs(world, road)
                input.append(car.vel / MAX_VEL)
                car.commands = nets[i].activate(tuple(input))

                y_old = car.y
                (x, y) = car.move(road, t)

                if t > 10 and (
                    car.detectCollision(road)
                    or y > world.getBestCarPos()[1] + BAD_GENOME_TRESHOLD
                    or y > y_old
                    or car.vel < 0.1
                ):
                    ge[i].fitness -= 1
                    cars.pop(i)
                    nets.pop(i)
                    ge.pop(i)
                    NNs.pop(i)
                else:
                    ge[i].fitness += -(y - y_old) / 100 + car.vel * SCORE_VEL_MULTIPLIER
                    if ge[i].fitness > world.getScore():
                        world.updateScore(ge[i].fitness)
                        world.bestNN = NNs[i]
                        world.bestInputs = input
                        world.bestCommands = car.commands
                    i += 1

                if y < yb:
                    (xb, yb) = (x, y)

            if len(cars) == 0:
                run = False
                break

            world.updateBestCarPos((xb, yb))
            road.update(world)

            road.draw(world)
            for car in cars:
                car.draw(world)

            text = STAT_FONT.render(
                "Best Car Score: " + str(int(world.getScore())), 1, BLACK
            )
            world.win.blit(text, (world.win_width - text.get_width() - 10, 10))

            if world.bestNN:
                world.bestNN.draw(world)

            py.display.update()

            frame = py.surfarray.array3d(world.win)
            yield frame

            world.win.blit(self.bg, (0, 0))

        pass

    def run(self):

        n = 10000
        for i in range(n):
            self.reporter.start_generation(self.population.generation)

            logger = self.reporter

            gen_iterator = self.eval_genomes(
                list(self.population.population.items()), self.config
            )

            for frame in gen_iterator:
                yield "frame", frame

            best = None
            for g in self.population.population.values():
                if best is None or g.fitness > best.fitness:
                    best = g

            self.reporter.post_evaluate(
                self.config, self.population.population, self.population.species, best
            )

            yield "stats", self.reporter.get_stats()

            if self.config.fitness_criterion == "max":
                fitness = best.fitness

            if fitness >= self.config.fitness_threshold:
                break

            self.population.species.speciate(
                self.config, self.population.population, self.population.generation
            )
            self.population.population = self.population.reproduction.reproduce(
                self.config,
                self.population.species,
                self.config.pop_size,
                self.population.generation,
            )
            self.population.generation += 1
