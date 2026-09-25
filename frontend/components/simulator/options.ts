import type { RunConfig, Speed } from "@/lib/types"

export const SPEED_OPTIONS: { value: Speed; label: string; description?: string }[] = [
  { value: "0.25", label: "0.25× Slow Motion", description: "Easiest to follow each decision" },
  { value: "0.5", label: "0.5× Half Speed" },
  { value: "1", label: "1× Real Time", description: "30 ticks per second" },
  { value: "2", label: "2× Fast" },
  { value: "4", label: "4× Faster" },
  { value: "8", label: "8× Very Fast" },
  { value: "max", label: "Max Speed", description: "As fast as the server can run" },
]

export const POPULATION_OPTIONS = [20, 30, 50, 80, 100, 150].map((n) => ({
  value: String(n),
  label: `${n} Cars`,
  description:
    n <= 30
      ? "Faster generations, less variety"
      : n >= 100
        ? "More variety, slower generations"
        : n === 50
          ? "Balanced (recommended)"
          : undefined,
}))

export const GENERATION_OPTIONS = [10, 25, 50, 100, 200].map((n) => ({
  value: String(n),
  label: `${n} Generations`,
}))

export const LENGTH_OPTIONS: { value: RunConfig["track_length"]; label: string }[] = [
  { value: "short", label: "Short" },
  { value: "medium", label: "Medium" },
  { value: "long", label: "Long" },
]

export const WIDTH_OPTIONS: { value: RunConfig["track_width"]; label: string }[] = [
  { value: "wide", label: "Wide" },
  { value: "standard", label: "Standard" },
  { value: "narrow", label: "Narrow" },
]

export const CURVE_OPTIONS: { value: RunConfig["track_curviness"]; label: string }[] = [
  { value: "gentle", label: "Gentle" },
  { value: "standard", label: "Standard" },
  { value: "twisty", label: "Twisty" },
]

export const TRACK_MODE_OPTIONS: {
  value: RunConfig["track_mode"]
  label: string
  description: string
}[] = [
  {
    value: "new",
    label: "New Track Each Generation",
    description: "Cars must learn to drive any road",
  },
  {
    value: "same",
    label: "Same Track Every Generation",
    description: "Cars can memorize one road",
  },
]

export function defaultConfig(prefs: {
  default_population: number
  default_generations: number
  default_speed: Speed
}): RunConfig {
  return {
    name: "",
    population_size: prefs.default_population,
    max_generations: prefs.default_generations,
    track_length: "medium",
    track_width: "standard",
    track_curviness: "standard",
    track_mode: "new",
    track_seed: null,
    weight_mutation_rate: 0.8,
    add_connection_rate: 0.3,
    add_node_rate: 0.2,
    speed: prefs.default_speed,
  }
}
