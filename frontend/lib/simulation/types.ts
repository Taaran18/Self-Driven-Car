import type { NetworkShape, RunConfig, Speed } from "@/lib/types"
import type { Usage } from "@/lib/usage"

export type StepId =
  "train" | "tick" | "sense" | "think" | "decide" | "move" | "score" | "evaluate" | "evolve"

export interface CodeSnippet {
  name: string
  file: string
  start_line: number
  code: string
}

export interface CodeStep {
  id: StepId
  snippets: CodeSnippet[]
}

export interface Trace {
  genome_id: number
  sense: { distances: number[]; inputs: number[]; range: number }
  think: { outputs: number[]; nodes: Record<string, number> }
  decide: { accelerate: boolean; brake: boolean; turn_left: boolean; turn_right: boolean }
  move: {
    speed_before: number
    speed: number
    rotation_before: number
    rotation: number
    x: number
    y: number
  }
  score: {
    progress: number
    fitness_before: number
    fitness: number
    checks: { crashed: boolean; fell_behind: boolean; reversing: boolean; stalled: boolean }
    eliminated: boolean
  }
}

export type CarRow = [id: number, x: number, y: number, rotation: number, braking: number]

export interface RetiredCar {
  id: number
  x: number
  y: number
  rotation: number
  reason?: string
}

export interface FrameMessage {
  type: "frame"
  stepped: boolean
  tick: number
  alive: number
  finished: number
  leader_y: number
  cars: CarRow[]
  eliminated: RetiredCar[]
  crossed: RetiredCar[]
  trace: Trace
}

export interface RoadMessage {
  type: "road"
  version: number
  left: number[]
  right: number[]
  finish_y: number
  width: number
}

export interface SpeciesInfo {
  id: number
  size: number
  best_fitness: number
  mean_fitness: number
  age: number
  stagnant_for: number
}

export interface EvolutionInfo {
  extinct: boolean
  elites_kept: number
  offspring: number
  species_after: number
  elitism: number
  survival_threshold: number
}

export interface GenerationMessage {
  type: "generation"
  generation: number
  best_fitness: number
  mean_fitness: number
  std_fitness: number
  species_count: number
  species: SpeciesInfo[]
  best_genome_id: number
  best_genome_nodes: number
  best_genome_connections: number
  ticks: number
  finishers: number
  duration_ms: number
  elapsed_seconds: number
  evolution: EvolutionInfo | null
  champion: NetworkShape
}

export interface HelloMessage {
  type: "hello"
  trial_id: string
  capacity: { active: number; max: number }
  limits: { max_minutes: number; idle_minutes: number }
  code: CodeStep[]
}

export interface StartedMessage {
  type: "started"
  run_id: string
  name: string
  seed: number
  config: RunConfig
  usage: Usage
}

export interface GenerationStartMessage {
  type: "generation_start"
  generation: number
  population: number
  max_ticks: number
  finish_y: number
}

export interface StatusMessage {
  type: "status"
  state: "running" | "paused"
  speed: Speed
}

export interface EndedMessage {
  type: "ended"
  status: "completed" | "stopped" | "interrupted" | "failed"
  reason: string
  generations_completed: number
  best_fitness: number | null
  elapsed_seconds: number
  run_id: string | null
  refunded: boolean
  usage: Usage | null
}

export interface ErrorMessage {
  type: "error"
  code: string
  message: string
  details?: Record<string, unknown>
}

export type ServerMessage =
  | HelloMessage
  | StartedMessage
  | GenerationStartMessage
  | StatusMessage
  | FrameMessage
  | RoadMessage
  | (NetworkShape & { type: "network" })
  | GenerationMessage
  | EndedMessage
  | ErrorMessage
  | { type: "idle"; message: string }
  | { type: "pong" }
