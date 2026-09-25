import statistics
from pathlib import Path

import neat

from app.simulation.constants import INPUT_NAMES, OUTPUT_NAMES, PROGRESS_SCALE
from app.simulation.schemas import RunConfig

CONFIG_PATH = Path(__file__).with_name("neat.cfg")


def build_neat_config(run: RunConfig, track_length: float) -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    config.pop_size = run.population_size
    config.fitness_threshold = track_length / PROGRESS_SCALE
    config.genome_config.weight_mutate_rate = run.weight_mutation_rate
    config.genome_config.conn_add_prob = run.add_connection_rate
    config.genome_config.node_add_prob = run.add_node_rate
    return config


def summarize_generation(population: neat.Population) -> dict:
    genomes = list(population.population.values())
    fitnesses = [g.fitness for g in genomes]
    best = max(genomes, key=lambda g: g.fitness)
    species = []
    for species_id, s in population.species.species.items():
        member_fitness = [m.fitness for m in s.members.values()]
        species.append(
            {
                "id": species_id,
                "size": len(s.members),
                "best_fitness": max(member_fitness),
                "mean_fitness": statistics.fmean(member_fitness),
                "age": population.generation - s.created,
                "stagnant_for": population.generation - s.last_improved,
            }
        )
    return {
        "best_fitness": best.fitness,
        "mean_fitness": statistics.fmean(fitnesses),
        "std_fitness": statistics.pstdev(fitnesses),
        "species_count": len(species),
        "species": sorted(species, key=lambda s: -s["best_fitness"]),
        "best_genome": best,
    }


def evolve(population: neat.Population) -> dict:
    config = population.config
    before = {g.key for g in population.population.values()}
    population.population = population.reproduction.reproduce(
        config, population.species, config.pop_size, population.generation
    )
    extinct = not population.species.species
    if extinct:
        population.population = population.reproduction.create_new(
            config.genome_type, config.genome_config, config.pop_size
        )
    population.species.speciate(config, population.population, population.generation)
    population.generation += 1
    survivors = sum(1 for key in population.population if key in before)
    return {
        "extinct": extinct,
        "elites_kept": survivors,
        "offspring": len(population.population) - survivors,
        "species_after": len(population.species.species),
        "elitism": config.reproduction_config.elitism,
        "survival_threshold": config.reproduction_config.survival_threshold,
    }


def describe_network(genome, config: neat.Config) -> dict:
    inputs = list(config.genome_config.input_keys)
    outputs = list(config.genome_config.output_keys)
    enabled = [(key, c.weight) for key, c in genome.connections.items() if c.enabled]
    depth = {node: 0 for node in inputs}
    changed = True
    passes = 0
    while changed and passes < 50:
        changed = False
        passes += 1
        for (source, target), _ in enabled:
            if source in depth and depth.get(target, -1) < depth[source] + 1:
                depth[target] = depth[source] + 1
                changed = True
    hidden = [n for n in genome.nodes if n not in outputs]
    hidden_layers = {n: max(1, depth.get(n, 1)) for n in hidden}
    output_layer = max(hidden_layers.values(), default=0) + 1
    nodes = [{"id": n, "kind": "input", "label": INPUT_NAMES[i], "layer": 0} for i, n in enumerate(inputs)]
    nodes += [{"id": n, "kind": "hidden", "label": f"Hidden {n}", "layer": hidden_layers[n]} for n in hidden]
    nodes += [
        {"id": n, "kind": "output", "label": OUTPUT_NAMES[i], "layer": output_layer} for i, n in enumerate(outputs)
    ]
    known = {node["id"] for node in nodes}
    return {
        "genome_id": genome.key,
        "nodes": nodes,
        "connections": [
            {"from": s, "to": t, "weight": round(w, 3)} for (s, t), w in enabled if s in known and t in known
        ],
        "layers": output_layer + 1,
    }
