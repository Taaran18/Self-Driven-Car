import type { StepId } from "./types"

export interface StepInfo {
  id: StepId
  label: string
  title: string
  scope: "generation" | "tick"
  summary: string
  watch: string
}

export const STEPS: StepInfo[] = [
  {
    id: "train",
    label: "Training Loop",
    title: "The Training Loop",
    scope: "generation",
    summary:
      "Every generation gets a fresh track. All cars drive at once until they crash or run out of time. Then NEAT scores them and breeds the next generation from the best drivers.",
    watch: "The highlighted line shows which part of the loop is running right now.",
  },
  {
    id: "tick",
    label: "One Tick",
    title: "One Tick of the Simulation",
    scope: "tick",
    summary:
      "Thirty times a second, every car runs the same five steps: sense, think, decide, move, and score. This function is the heartbeat of the whole simulation.",
    watch: "Press Step to pause and walk through one tick, one line at a time.",
  },
  {
    id: "sense",
    label: "Sense",
    title: "Sense the Road",
    scope: "tick",
    summary:
      "Each car casts 8 rays around itself, like a simple lidar. A reading is 0 when no wall is within 200 units and climbs toward 1 as a wall gets closer. The car's speed is the ninth input.",
    watch: "Watch the front readings rise just before the car turns.",
  },
  {
    id: "think",
    label: "Think",
    title: "Think With a Neural Network",
    scope: "tick",
    summary:
      "The 9 readings flow through the car's own neural network. Each connection multiplies a value by its weight, and each neuron squashes the total with tanh. Out come 4 numbers between −1 and 1.",
    watch: "Brighter neurons in the network diagram are firing more strongly.",
  },
  {
    id: "decide",
    label: "Decide",
    title: "Turn Numbers Into Actions",
    scope: "tick",
    summary:
      "An output only counts if it passes 0.5 and beats its opposite. Accelerate competes with Brake, and Turn Left competes with Turn Right. That is how four numbers become driving actions.",
    watch: "Compare each output with the 0.5 threshold and its rival.",
  },
  {
    id: "move",
    label: "Move",
    title: "Apply Simple Physics",
    scope: "tick",
    summary:
      "Turning changes the heading by 2°. The pedal changes speed, while friction slowly takes it away. Then the car moves along its new heading.",
    watch: "Speed tops out at 10 units per tick.",
  },
  {
    id: "score",
    label: "Score",
    title: "Score and Eliminate",
    scope: "tick",
    summary:
      "Moving forward earns fitness. A car is removed if it hits a wall, falls 200 units behind the leader, drives backward, or stalls. Removal costs 1 point.",
    watch: "Fitness is roughly the distance driven, in hundreds of units.",
  },
  {
    id: "evaluate",
    label: "Evaluate",
    title: "Evaluate the Generation",
    scope: "generation",
    summary:
      "When every car has stopped, NEAT collects each network's fitness. It groups similar networks into species, so new ideas get time to improve before they compete with the best.",
    watch: "Updates at the end of every generation.",
  },
  {
    id: "evolve",
    label: "Evolve",
    title: "Breed the Next Generation",
    scope: "generation",
    summary:
      "The best networks survive unchanged. That is called elitism. The rest are replaced by children: crossover mixes two parents, and mutation nudges weights or adds new connections and neurons.",
    watch: "Updates at the end of every generation.",
  },
]

export const TICK_PHASES: StepId[] = ["sense", "think", "decide", "move", "score"]

const TICK_LINE_PATTERNS: Partial<Record<StepId, RegExp>> = {
  sense: /=\s*sense\(/,
  think: /=\s*think\(/,
  decide: /=\s*decide\(/,
  move: /^\s*move\(/,
  score: /=\s*score\(/,
}

export function tickLineFor(code: string, phase: StepId): number | null {
  const pattern = TICK_LINE_PATTERNS[phase]
  if (!pattern) return null
  const index = code.split("\n").findIndex((line) => pattern.test(line))
  return index >= 0 ? index : null
}

export function trainLineFor(code: string, phase: "drive" | "evaluate" | "evolve"): number | null {
  const pattern =
    phase === "drive"
      ? /yield cars\.tick\(\)/
      : phase === "evaluate"
        ? /summarize_generation\(/
        : /evolve\(self/
  const index = code.split("\n").findIndex((line) => pattern.test(line))
  return index >= 0 ? index : null
}
