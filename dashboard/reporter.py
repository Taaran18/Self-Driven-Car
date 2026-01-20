from neat.reporting import BaseReporter
import time


class NEATReporter(BaseReporter):
    def __init__(self):
        self.stats = []
        self.current_gen = 0
        self.generation_start_time = None

    def start_generation(self, generation):
        self.current_gen = generation
        self.generation_start_time = time.time()

    def post_evaluate(self, config, population, species, best_genome):
        # Calculate statistics
        fitnesses = [c.fitness for c in population.values()]
        fit_mean = sum(fitnesses) / len(fitnesses)
        fit_std = (
            sum([(x - fit_mean) ** 2 for x in fitnesses]) / len(fitnesses)
        ) ** 0.5

        # Store stats
        self.stats.append(
            {
                "Generation": self.current_gen,
                "Max Fitness": best_genome.fitness,
                "Average Fitness": fit_mean,
                "Std Dev": fit_std,
                "Best Genome ID": best_genome.key,
            }
        )

    def get_stats(self):
        return self.stats
