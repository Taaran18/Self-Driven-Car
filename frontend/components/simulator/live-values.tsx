import { Check, X } from "lucide-react"
import { cn } from "@/lib/cn"
import { formatNumber } from "@/lib/format"
import type { GenerationMessage, Trace } from "@/lib/simulation/types"

const SENSOR_NAMES = [
  "Front",
  "Front Right",
  "Right",
  "Rear Right",
  "Rear",
  "Rear Left",
  "Left",
  "Front Left",
]
const OUTPUT_NAMES = ["Accelerate", "Brake", "Turn Left", "Turn Right"]

function Row({
  label,
  children,
  className,
}: {
  label: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn("flex items-center justify-between gap-3 py-1.5 text-sm", className)}>
      <span className="text-fg-muted">{label}</span>
      <span className="tabular text-right font-semibold text-fg">{children}</span>
    </div>
  )
}

function Bar({
  value,
  min = 0,
  max = 1,
  tone = "primary",
}: {
  value: number
  min?: number
  max?: number
  tone?: "primary" | "warning" | "danger" | "success"
}) {
  const ratio = Math.min(1, Math.max(0, (value - min) / (max - min)))
  const colors = {
    primary: "bg-primary",
    warning: "bg-warning",
    danger: "bg-danger",
    success: "bg-success",
  }
  return (
    <span className="relative block h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
      <span
        className={cn(
          "absolute inset-y-0 left-0 rounded-full transition-[width] duration-150",
          colors[tone],
        )}
        style={{ width: `${ratio * 100}%` }}
      />
    </span>
  )
}

function Verdict({ yes }: { yes: boolean }) {
  return yes ? (
    <span className="inline-flex items-center gap-1 rounded-md bg-success-soft px-1.5 py-0.5 text-xs font-bold text-success">
      <Check className="size-3" /> Yes
    </span>
  ) : (
    <span className="inline-flex items-center gap-1 rounded-md bg-surface-3 px-1.5 py-0.5 text-xs font-bold text-fg-subtle">
      <X className="size-3" /> No
    </span>
  )
}

export function SenseValues({ trace }: { trace: Trace }) {
  const { distances, inputs, range } = trace.sense
  return (
    <div className="space-y-2">
      <div className="grid grid-cols-[minmax(0,1fr)_64px_minmax(0,1.2fr)_44px] items-center gap-x-3 gap-y-2 text-sm">
        <span className="text-xs font-semibold tracking-wider text-fg-subtle uppercase">Ray</span>
        <span className="text-right text-xs font-semibold tracking-wider text-fg-subtle uppercase">
          Distance
        </span>
        <span className="text-xs font-semibold tracking-wider text-fg-subtle uppercase">
          Closeness
        </span>
        <span className="text-right text-xs font-semibold tracking-wider text-fg-subtle uppercase">
          Input
        </span>
        {SENSOR_NAMES.map((name, i) => {
          const reading = inputs[i] ?? 0
          return (
            <div key={name} className="contents">
              <span className="truncate text-fg-muted">{name}</span>
              <span className="tabular text-right text-fg">
                {distances[i] >= range ? "Clear" : formatNumber(distances[i])}
              </span>
              <Bar
                value={reading}
                tone={reading > 0.6 ? "danger" : reading > 0.3 ? "warning" : "success"}
              />
              <span className="tabular text-right font-semibold text-fg">{reading.toFixed(2)}</span>
            </div>
          )
        })}
      </div>
      <Row label="Speed Input (speed ÷ 10)" className="border-t border-border pt-2">
        {(inputs[8] ?? 0).toFixed(2)}
      </Row>
    </div>
  )
}

export function ThinkValues({ trace }: { trace: Trace }) {
  const hidden = Object.keys(trace.think.nodes).filter((k) => Number(k) > 3).length
  return (
    <div className="space-y-3">
      <p className="text-sm text-fg-muted">
        9 inputs in, {hidden} hidden neuron{hidden === 1 ? "" : "s"}, 4 outputs out. Each output is
        tanh of a weighted sum, so it always lands between −1 and 1.
      </p>
      <div className="space-y-2.5">
        {OUTPUT_NAMES.map((name, i) => {
          const value = trace.think.outputs[i] ?? 0
          return (
            <div key={name}>
              <div className="flex items-center justify-between text-sm">
                <span className="text-fg-muted">{name}</span>
                <span className="tabular font-semibold text-fg">{value.toFixed(3)}</span>
              </div>
              <div className="relative mt-1">
                <Bar value={value} min={-1} max={1} tone={value > 0.5 ? "success" : "primary"} />
                <span
                  aria-hidden
                  className="absolute -top-0.5 h-2.5 w-px bg-fg"
                  style={{ left: "75%" }}
                  title="Threshold 0.5"
                />
              </div>
            </div>
          )
        })}
      </div>
      <p className="text-xs text-fg-subtle">
        The thin mark on each bar is the 0.5 threshold used in the next step.
      </p>
    </div>
  )
}

export function DecideValues({ trace }: { trace: Trace }) {
  const [acc, brake, left, right] = trace.think.outputs
  const rows = [
    {
      label: "Accelerate",
      value: acc,
      rival: brake,
      rivalName: "Brake",
      yes: trace.decide.accelerate,
    },
    { label: "Brake", value: brake, rival: acc, rivalName: "Accelerate", yes: trace.decide.brake },
    {
      label: "Turn Left",
      value: left,
      rival: right,
      rivalName: "Turn Right",
      yes: trace.decide.turn_left,
    },
    {
      label: "Turn Right",
      value: right,
      rival: left,
      rivalName: "Turn Left",
      yes: trace.decide.turn_right,
    },
  ]
  return (
    <ul className="divide-y divide-border">
      {rows.map((row) => (
        <li key={row.label} className="flex items-center justify-between gap-3 py-2 text-sm">
          <div className="min-w-0">
            <p className="font-semibold text-fg">{row.label}</p>
            <p className="tabular text-xs text-fg-subtle">
              {row.value.toFixed(2)} {row.value > 0.5 ? ">" : "≤"} 0.5 and {row.value.toFixed(2)}{" "}
              {row.value > row.rival ? ">" : "≤"} {row.rivalName} {row.rival.toFixed(2)}
            </p>
          </div>
          <Verdict yes={row.yes} />
        </li>
      ))}
    </ul>
  )
}

export function MoveValues({ trace }: { trace: Trace }) {
  const m = trace.move
  const turn = m.rotation - m.rotation_before
  return (
    <div className="divide-y divide-border">
      <Row label="Speed">
        {m.speed_before.toFixed(2)} → {m.speed.toFixed(2)}
      </Row>
      <Row label="Heading">
        {m.rotation_before.toFixed(0)}° → {m.rotation.toFixed(0)}°{" "}
        <span className="font-normal text-fg-subtle">
          ({turn === 0 ? "straight" : turn > 0 ? "right" : "left"})
        </span>
      </Row>
      <Row label="Position">
        x {formatNumber(m.x)}, y {formatNumber(m.y)}
      </Row>
      <p className="pt-2 text-xs text-fg-subtle">
        The track runs upward, so moving forward means y gets smaller.
      </p>
    </div>
  )
}

export function ScoreValues({ trace }: { trace: Trace }) {
  const s = trace.score
  const checks = [
    { label: "Hit a wall", value: s.checks.crashed },
    { label: "Fell 200 units behind the leader", value: s.checks.fell_behind },
    { label: "Drove backward", value: s.checks.reversing },
    { label: "Stalled below 0.1 speed", value: s.checks.stalled },
  ]
  return (
    <div className="space-y-3">
      <div className="divide-y divide-border">
        <Row label="Progress This Tick">
          {s.eliminated ? "−1.000 penalty" : `+${s.progress.toFixed(3)}`}
        </Row>
        <Row label="Fitness">
          {s.fitness_before.toFixed(2)} → {s.fitness.toFixed(2)}
        </Row>
      </div>
      <ul className="space-y-1.5">
        {checks.map((check) => (
          <li key={check.label} className="flex items-center justify-between gap-3 text-sm">
            <span className="text-fg-muted">{check.label}</span>
            {check.value ? (
              <span className="rounded-md bg-danger-soft px-1.5 py-0.5 text-xs font-bold text-danger">
                Eliminated
              </span>
            ) : (
              <span className="rounded-md bg-surface-3 px-1.5 py-0.5 text-xs font-bold text-fg-subtle">
                OK
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  )
}

export function EvaluateValues({ generation }: { generation: GenerationMessage | null }) {
  if (!generation)
    return (
      <p className="text-sm text-fg-subtle">Values appear when the first generation finishes.</p>
    )
  return (
    <div className="space-y-3">
      <div className="divide-y divide-border">
        <Row label={`Generation ${generation.generation + 1} Best`}>
          {generation.best_fitness.toFixed(2)}
        </Row>
        <Row label="Average Fitness">{generation.mean_fitness.toFixed(2)}</Row>
        <Row label="Spread (Std Dev)">{generation.std_fitness.toFixed(2)}</Row>
        <Row label="Reached the Finish">{generation.finishers}</Row>
      </div>
      <div>
        <p className="mb-2 text-xs font-semibold tracking-wider text-fg-subtle uppercase">
          Species
        </p>
        <ul className="space-y-1.5">
          {generation.species.slice(0, 5).map((s) => (
            <li key={s.id} className="flex items-center justify-between gap-3 text-sm">
              <span className="text-fg-muted">
                Species {s.id} <span className="text-fg-subtle">· {s.size} cars</span>
              </span>
              <span className="tabular font-semibold text-fg">
                best {s.best_fitness.toFixed(1)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  )
}

export function EvolveValues({ generation }: { generation: GenerationMessage | null }) {
  const evolution = generation?.evolution
  if (!generation)
    return (
      <p className="text-sm text-fg-subtle">Values appear when the first generation finishes.</p>
    )
  if (!evolution)
    return (
      <p className="text-sm text-fg-subtle">
        This was the final generation, so no new generation was bred.
      </p>
    )
  return (
    <div className="divide-y divide-border">
      <Row label="Champions Kept Unchanged">{evolution.elites_kept}</Row>
      <Row label="New Children Bred">{evolution.offspring}</Row>
      <Row label="Parents Chosen From Top">
        {Math.round(evolution.survival_threshold * 100)}% of each species
      </Row>
      <Row label="Species Next Generation">{evolution.species_after}</Row>
      <Row label="Mass Extinction">{evolution.extinct ? "Yes, restarted" : "No"}</Row>
    </div>
  )
}
