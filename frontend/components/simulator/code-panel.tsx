"use client"

import { ChevronDown, ChevronLeft, ChevronRight, Radio } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { CodeBlock } from "@/components/ui/code-block"
import { Switch } from "@/components/ui/controls"
import { cn } from "@/lib/cn"
import { updatePreferences } from "@/lib/preferences"
import type { FrameSummary, Phase } from "@/lib/simulation/controller"
import { STEPS, TICK_PHASES, tickLineFor, trainLineFor } from "@/lib/simulation/steps"
import type { CodeStep, GenerationMessage, StepId } from "@/lib/simulation/types"
import {
  DecideValues,
  EvaluateValues,
  EvolveValues,
  MoveValues,
  ScoreValues,
  SenseValues,
  ThinkValues,
} from "./live-values"

const WALK_MS = 1800

interface CodePanelProps {
  code: CodeStep[] | null
  frame: FrameSummary | null
  phase: Phase
  stepSeq: number
  lastGeneration: GenerationMessage | null
  guided: boolean
}

export function CodePanel({ code, frame, phase, stepSeq, lastGeneration, guided }: CodePanelProps) {
  const [selected, setSelected] = useState<StepId>("sense")
  const [walk, setWalk] = useState<number | null>(null)
  const [seenStep, setSeenStep] = useState(stepSeq)
  const [showContext, setShowContext] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)

  if (stepSeq !== seenStep) {
    setSeenStep(stepSeq)
    if (guided && stepSeq > 0) {
      setWalk(0)
      setSelected(TICK_PHASES[0])
    }
  }

  useEffect(() => {
    if (walk === null) return
    timer.current = setTimeout(() => {
      if (walk < TICK_PHASES.length - 1) {
        setWalk(walk + 1)
        setSelected(TICK_PHASES[walk + 1])
      } else {
        setWalk(null)
      }
    }, WALK_MS)
    return () => {
      if (timer.current) clearTimeout(timer.current)
    }
  }, [walk])

  const snippets = useMemo(() => new Map(code?.map((s) => [s.id, s.snippets]) ?? []), [code])
  const info = STEPS.find((s) => s.id === selected) ?? STEPS[2]
  const isTickStep = TICK_PHASES.includes(selected)
  const contextId: StepId = isTickStep ? "tick" : "train"
  const context = snippets.get(contextId)?.[0]
  const detail = snippets.get(selected) ?? []
  const frozen = walk !== null || phase === "paused"

  const contextHighlight = useMemo(() => {
    if (!context) return undefined
    const line = isTickStep
      ? frozen
        ? tickLineFor(context.code, selected)
        : null
      : trainLineFor(context.code, selected === "evolve" ? "evolve" : "evaluate")
    return line === null ? undefined : [line]
  }, [context, isTickStep, selected, frozen])

  function choose(id: StepId) {
    setWalk(null)
    setSelected(id)
  }

  function moveWalk(delta: number) {
    const current = walk ?? TICK_PHASES.indexOf(selected)
    const next = Math.min(TICK_PHASES.length - 1, Math.max(0, current + delta))
    setWalk(null)
    setSelected(TICK_PHASES[next])
  }

  const values = (() => {
    if (!isTickStep) {
      return selected === "evaluate" ? (
        <EvaluateValues generation={lastGeneration} />
      ) : (
        <EvolveValues generation={lastGeneration} />
      )
    }
    if (!frame) {
      return (
        <p className="text-sm text-fg-subtle">
          Press Start Training to see the real numbers flowing through this step for the leading
          car.
        </p>
      )
    }
    const trace = frame.trace
    switch (selected) {
      case "sense":
        return <SenseValues trace={trace} />
      case "think":
        return <ThinkValues trace={trace} />
      case "decide":
        return <DecideValues trace={trace} />
      case "move":
        return <MoveValues trace={trace} />
      default:
        return <ScoreValues trace={trace} />
    }
  })()

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3">
        {walk !== null ? (
          <Badge tone="primary" dot>
            Walking Through Tick {frame?.tick ?? ""}
          </Badge>
        ) : phase === "running" ? (
          <Badge tone="success">
            <Radio className="size-3" /> Live · 30 Ticks per Second
          </Badge>
        ) : phase === "paused" ? (
          <Badge tone="warning" dot>
            Paused at Tick {frame?.tick ?? 0}
          </Badge>
        ) : (
          <Badge>Real Python From the Server</Badge>
        )}
        <div className="min-w-48">
          <Switch
            label="Guided Walkthrough"
            checked={guided}
            onChange={(v) => updatePreferences({ guided_steps: v })}
          />
        </div>
      </div>

      <nav
        aria-label="Simulation steps"
        className="rounded-2xl border border-border bg-surface p-3"
      >
        <div className="flex items-center justify-between gap-2">
          <p className="text-xs font-semibold tracking-wider text-fg-subtle uppercase">
            Every Tick
          </p>
          <div className="flex gap-1">
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => moveWalk(-1)}
              disabled={!isTickStep || selected === TICK_PHASES[0]}
              aria-label="Previous step"
            >
              <ChevronLeft className="size-4" />
            </Button>
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => moveWalk(1)}
              disabled={!isTickStep || selected === TICK_PHASES[TICK_PHASES.length - 1]}
              aria-label="Next step"
            >
              <ChevronRight className="size-4" />
            </Button>
          </div>
        </div>
        <StepStrip ids={TICK_PHASES} selected={selected} onSelect={choose} />
        <p className="mt-3 text-xs font-semibold tracking-wider text-fg-subtle uppercase">
          Between Generations
        </p>
        <StepStrip ids={["evaluate", "evolve"]} selected={selected} onSelect={choose} />
      </nav>

      <div className="space-y-4">
        <div>
          <h3 className="text-lg font-bold text-fg">{info.title}</h3>
          <p className="mt-1.5 text-sm leading-relaxed text-fg-muted">{info.summary}</p>
        </div>

        <div className="rounded-2xl border border-border bg-surface p-4">
          <div className="mb-3 flex items-center justify-between gap-3">
            <p className="text-sm font-bold text-fg">Live Values</p>
            {isTickStep && frame ? (
              <span className="text-xs text-fg-subtle">
                Car #{frame.trace.genome_id} · tick {frame.tick}
              </span>
            ) : !isTickStep && lastGeneration ? (
              <span className="text-xs text-fg-subtle">
                Generation {lastGeneration.generation + 1}
              </span>
            ) : null}
          </div>
          {values}
          <p className="mt-3 border-t border-border pt-3 text-xs text-fg-subtle">{info.watch}</p>
        </div>

        <div className="space-y-3">
          {detail.length ? (
            detail.map((snippet) => (
              <CodeBlock
                key={snippet.name}
                code={snippet.code}
                file={snippet.file}
                startLine={snippet.start_line}
                maxHeight={300}
              />
            ))
          ) : (
            <CodeSkeleton />
          )}
        </div>

        {context ? (
          <div className="rounded-2xl border border-border bg-surface">
            <button
              type="button"
              onClick={() => setShowContext((v) => !v)}
              aria-expanded={showContext}
              className="flex w-full items-center justify-between gap-3 rounded-2xl px-4 py-3 text-left text-sm font-semibold text-fg hover:bg-surface-2"
            >
              <span>
                {isTickStep ? "Where This Runs Each Tick" : "Where This Runs in Training"}
                <span className="ml-2 font-mono text-xs font-normal text-fg-subtle">
                  {isTickStep ? "tick()" : "train()"}
                </span>
              </span>
              <ChevronDown
                className={cn(
                  "size-4 text-fg-subtle transition-transform",
                  showContext && "rotate-180",
                )}
              />
            </button>
            {showContext ? (
              <div className="border-t border-border p-2">
                <CodeBlock
                  code={context.code}
                  file={context.file}
                  startLine={context.start_line}
                  highlight={contextHighlight}
                  maxHeight={320}
                  scrollToHighlight
                />
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function StepStrip({
  ids,
  selected,
  onSelect,
}: {
  ids: StepId[]
  selected: StepId
  onSelect: (id: StepId) => void
}) {
  return (
    <ol className="mt-2 flex items-center gap-1">
      {ids.map((id, i) => {
        const step = STEPS.find((s) => s.id === id)
        const active = id === selected
        return (
          <li key={id} className="flex min-w-0 flex-1 items-center gap-1">
            <button
              type="button"
              onClick={() => onSelect(id)}
              aria-pressed={active}
              className={cn(
                "w-full truncate rounded-lg px-1.5 py-1.5 text-xs font-semibold transition-colors sm:text-sm",
                active ? "bg-primary text-primary-fg" : "bg-surface-3 text-fg-muted hover:text-fg",
              )}
            >
              {step?.label}
            </button>
            {i < ids.length - 1 ? (
              <ChevronRight aria-hidden className="size-3 shrink-0 text-fg-subtle" />
            ) : null}
          </li>
        )
      })}
    </ol>
  )
}

function CodeSkeleton() {
  return (
    <div
      className="space-y-2 rounded-2xl border border-dashed border-border p-4"
      aria-label="Code loads when you start a run"
    >
      {[80, 64, 92, 48, 72].map((w, i) => (
        <div
          key={i}
          className="h-3 animate-pulse rounded bg-surface-3"
          style={{ width: `${w}%` }}
        />
      ))}
      <p className="pt-2 text-xs text-fg-subtle">
        The real source code loads from the server when you start a run.
      </p>
    </div>
  )
}
