import sys
import os
import pygame as py
import neat
import numpy as np

# Add the project root to sys.path to allow imports from src and config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.car import Car
from src.road import Road
from src.world import World
from src.NNdraw import NN
from config.config_variables import *
from dashboard.reporter import NEATReporter

# Set headless mode for Pygame to avoid popping up a window
os.environ["SDL_VIDEODRIVER"] = "dummy"

# Global state control
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

        # We also add stdout reporter to see logs in console if needed
        self.population.add_reporter(neat.StdOutReporter(True))

        # Background surface for drawing
        py.font.init()
        self.bg = py.Surface((WIN_WIDTH, WIN_HEIGHT))
        self.bg.fill(GRAY)

    def eval_genomes(self, genomes, config):
        """
        Modified main function to run one generation.
        This is a generator that yields the current frame (image)
        so Streamlit can display it.
        """
        global GEN
        # Since GEN is in config_variables, but we might want to track it locally or update it.
        # However, config_variables.GEN is just an initial value. logic uses a global or passed GEN.
        # We'll rely on the reporter for generation count or track it here.
        # Actually standard main.py uses a global GEN.

        if get_stop_simulation():
            # If stopped, we shouldn't even start this generation, but NEAT calls this.
            # We can just return immediately to stop evaluation.
            # But raising an exception is the best way to stop p.run()
            raise KeyboardInterrupt("Stopped by user")

        nets = []
        ge = []
        cars = []
        t = 0

        # Initialize World
        # We need to ensure we don't open a window if possible, but Pygame needs a video mode.
        # We can use a dummy video driver if we want it headless, but surfarray needs a surface.
        # We'll assume standard window is fine or user acceptable.
        # Actually, let's try to pass a flag to World or handled it before.
        # Since World calls set_mode, it will open a window.
        # We can minimize it or just let it be.

        # For this implementation, we will create a new World instance each generation
        # which is inefficient but matches original logic.
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
            # Limit FPS if we want to watch it, or run as fast as possible if just resolving.
            # But for Streamlit we want to see it.
            clock.tick(FPS)
            world.updateScore(0)

            # Pygame event handling
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

            # Drawing logic from draw_win
            road.draw(world)
            for car in cars:
                car.draw(world)

            text = STAT_FONT.render(
                "Best Car Score: " + str(int(world.getScore())), 1, BLACK
            )
            world.win.blit(text, (world.win_width - text.get_width() - 10, 10))
            # We don't have GEN variable passed in easily unless we track it.
            # For now, omit GEN in the display provided by Pygame or get it from reporter if possible.
            # text = STAT_FONT.render("Gen: " + str(GEN), 1, BLACK)
            # world.win.blit(text, (world.win_width - text.get_width() - 10, 50))

            if world.bestNN:
                world.bestNN.draw(world)

            py.display.update()

            # Capture frame for Streamlit
            # We yield the surface array (transposed to generic image format HxWx3)
            # Pygame is (W, H, 3), we need (H, W, 3) for most image libraries, or just let PIL/Streamlit handle it.
            # array3d returns (width, height, 3).
            frame = py.surfarray.array3d(world.win)
            yield frame

            world.win.blit(self.bg, (0, 0))

        # End of generation cleanup
        # py.quit() # Dropped to avoid killing the context for next gen if we reused it, but we recreate World each time.
        pass

    def run(self):
        # We cannot use p.run() standardly because we want to yield frames from INSIDE the eval_genomes.
        # So we have to write our own generational loop or use a trick.
        # Trick: eval_genomes is a generator. But p.run expects a function that returns None.

        # Alternative: We don't use p.run(). We manually iterate.
        # p.run() does:
        # for i in range(generations):
        #   self.reproduction.reproduce(...)
        #   self.evaluate_drive(...) -> calls eval_genomes
        #   self.message(...)
        #   self.stats...

        # We will iterate manually.

        n = 10000  # Max generations
        for i in range(n):
            self.reporter.start_generation(self.population.generation)

            # Evaluate
            # This is where we want to capture the generator output
            # But doing it manually requires replicating neat.p.run logic.
            # Easier approach: Use p.run but communicate via a shared queue?
            # OR: modifying eval_genomes to write to a shared variable that streamlit polls?
            # Streamlit is driving this loop within the script.

            # Let's try manual iteration logic roughly corresponding to p.run

            logger = self.reporter

            # 1. Create genomes
            # If it's first gen, they are already in self.population.
            # But standard run() loop is:
            #   1. Check completion
            #   2. Evaluate (fitness function)
            #   3. Reporters
            #   4. Reproduce

            # Evaluate genomes
            # current_genomes = list(self.population.population.items())
            # But we want to run the simulation loop which yields frames!

            # So:
            # We call our generator-based eval_genomes.
            # AND we yield the frames up to the app.

            # Get the generator
            gen_iterator = self.eval_genomes(
                list(self.population.population.items()), self.config
            )

            # Yield frames from the simulation
            for frame in gen_iterator:
                yield "frame", frame

            # After generation finishes (generator exhausted):
            # Gather fitness (eval_genomes updates genome.fitness in place)

            # Best genome
            best = None
            for g in self.population.population.values():
                if best is None or g.fitness > best.fitness:
                    best = g

            # Reporters
            self.reporter.post_evaluate(
                self.config, self.population.population, self.population.species, best
            )

            # Yield stats to update table
            yield "stats", self.reporter.get_stats()

            # Check for termination
            if self.config.fitness_criterion == "max":
                fitness = best.fitness
            # ... simplified checks ...

            if fitness >= self.config.fitness_threshold:
                break

            # Reproduce
            # neat-python's population.run logic for reproduction:
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
