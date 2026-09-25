export type Speed = "0.25" | "0.5" | "1" | "2" | "4" | "8" | "max"

export interface Preferences {
  default_speed: Speed
  default_population: number
  default_generations: number
  show_sensors: boolean
  show_network: boolean
  show_code: boolean
  confirm_before_stop: boolean
  guided_steps: boolean
}

export type RunStatus = "running" | "completed" | "stopped" | "interrupted" | "failed"
export type RunSort = "newest" | "oldest" | "best_fitness" | "generations" | "name"

export interface RunSummary {
  id: string
  name: string
  status: RunStatus
  stop_reason: string | null
  population_size: number
  max_generations: number
  generations_completed: number
  best_fitness: number | null
  best_generation: number | null
  created_at: string
  finished_at: string | null
  duration_seconds: number
}

export interface GenerationRecord {
  index: number
  best_fitness: number
  mean_fitness: number
  std_fitness: number
  species_count: number
  best_genome_id: number
  best_genome_nodes: number
  best_genome_connections: number
  ticks: number
  duration_ms: number
  created_at: string
}

export interface NetworkNode {
  id: number
  kind: "input" | "hidden" | "output"
  label: string
  layer: number
}

export interface NetworkConnection {
  from: number
  to: number
  weight: number
}

export interface NetworkShape {
  genome_id: number
  nodes: NetworkNode[]
  connections: NetworkConnection[]
  layers: number
  fitness?: number
}

export interface RunConfig {
  name?: string | null
  population_size: number
  max_generations: number
  track_length: "short" | "medium" | "long"
  track_width: "wide" | "standard" | "narrow"
  track_curviness: "gentle" | "standard" | "twisty"
  track_mode: "new" | "same"
  track_seed: number | null
  weight_mutation_rate: number
  add_connection_rate: number
  add_node_rate: number
  speed: Speed
}

export interface RunDetail extends RunSummary {
  config: Partial<RunConfig>
  champion: NetworkShape | null
  notes: string | null
  generations: GenerationRecord[]
}

export interface RunPage {
  items: RunSummary[]
  total: number
  page: number
  page_size: number
  pages: number
}

export interface OverviewStats {
  total_runs: number
  completed_runs: number
  active_runs: number
  total_generations: number
  best_fitness: number | null
  best_run_id: string | null
  best_run_name: string | null
  training_seconds: number
  recent_runs: RunSummary[]
  trend: {
    id: string
    name: string
    best_fitness: number | null
    generations_completed: number
    created_at: string
  }[]
}
