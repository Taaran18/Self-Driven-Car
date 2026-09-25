"use client"

import {
  Activity,
  BrainCircuit,
  CarFront,
  CodeXml,
  Dna,
  Flag,
  Gauge,
  History,
  Radar,
  SlidersHorizontal,
  Square,
  Trophy,
} from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { LineChart } from "@/components/charts/line-chart"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { StatCard } from "@/components/ui/card"
import { Dialog } from "@/components/ui/dialog"
import { EmptyState } from "@/components/ui/feedback"
import { TabList, TabPanel } from "@/components/ui/tabs"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { useToast } from "@/components/ui/toast"
import { useAwayPause } from "@/hooks/use-away-pause"
import { useSimulation } from "@/hooks/use-simulation"
import { cn } from "@/lib/cn"
import { formatFitness, formatNumber } from "@/lib/format"
import { updatePreferences, usePreferences } from "@/lib/preferences"
import type { SimSnapshot } from "@/lib/simulation/controller"
import type { GenerationMessage } from "@/lib/simulation/types"
import type { RunConfig } from "@/lib/types"
import { CodePanel } from "./code-panel"
import { ControlBar } from "./control-bar"
import { NetworkView } from "./network-view"
import { defaultConfig } from "./options"
import { RunSetup } from "./run-setup"
import { StageOverlay } from "./stage-overlay"
import { TrackCanvas } from "./track-canvas"

type InspectorTab = "setup" | "stats" | "brain" | "code" | "history"

const TABS: { value: InspectorTab; label: string; icon: React.ReactNode }[] = [
  { value: "setup", label: "Setup", icon: <SlidersHorizontal /> },
  { value: "stats", label: "Stats", icon: <Activity /> },
  { value: "brain", label: "Brain", icon: <BrainCircuit /> },
  { value: "code", label: "Code", icon: <CodeXml /> },
  { value: "history", label: "History", icon: <History /> },
]

export function SimulatorView() {
  const { controller, snapshot } = useSimulation()
  const prefs = usePreferences()
  const toast = useToast()
  const [config, setConfig] = useState<RunConfig | null>(null)
  const [tab, setTab] = useState<InspectorTab>("setup")
  const effectiveConfig = config ?? defaultConfig(prefs)
  const { phase, frame, generation, history, run } = snapshot
  const active = phase === "running" || phase === "paused"
  const connecting = phase === "connecting" || phase === "waking" || phase === "starting"
  const lastGeneration = history.at(-1) ?? null
  const population =
    generation?.population ?? run?.config.population_size ?? effectiveConfig.population_size
  const away = useAwayPause(controller, phase)
  const trackRef = useRef<HTMLDivElement>(null)

  const runId = run?.run_id
  const [seenRun, setSeenRun] = useState(runId)
  const [seenStep, setSeenStep] = useState(snapshot.stepSeq)
  if (runId !== seenRun) {
    setSeenRun(runId)
    if (runId) setTab("stats")
  }
  if (snapshot.stepSeq !== seenStep) {
    setSeenStep(snapshot.stepSeq)
    if (prefs.guided_steps && snapshot.stepSeq > 0) setTab("code")
  }

  useEffect(() => {
    if (!runId || window.innerWidth >= 1024) return
    trackRef.current?.scrollIntoView({ behavior: "smooth", block: "start" })
  }, [runId])

  const transientError = active ? snapshot.error : null
  useEffect(() => {
    if (transientError) toast.error("That Command Didn't Work", transientError.message)
  }, [transientError, toast])

  const statusText = active
    ? `Generation ${(generation?.generation ?? 0) + 1}. ${frame?.alive ?? population} of ${population} cars still driving.`
    : phase === "ended"
      ? "Run finished."
      : "Simulation idle."

  function start() {
    void controller.play({ ...effectiveConfig, name: effectiveConfig.name?.trim() || null })
  }

  return (
    <div className="flex flex-col lg:h-dvh lg:flex-row">
      <section
        ref={trackRef}
        className="flex min-w-0 scroll-mt-16 flex-col lg:flex-1"
        aria-label="Simulation track"
      >
        <header className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-border bg-bg px-4 py-3 sm:px-5 lg:h-[76px] lg:shrink-0 lg:flex-nowrap lg:py-0">
          <div className="min-w-0">
            <h1 className="truncate text-xl font-bold tracking-tight text-fg sm:text-2xl">
              Watch Cars Learn to Drive
            </h1>
            <div className="mt-0.5 flex min-w-0 items-center gap-2 text-sm text-fg-muted">
              <span className="truncate">{run ? run.name : "Live neuroevolution lab"}</span>
              {phase === "running" ? (
                <Badge tone="success" dot>
                  Training
                </Badge>
              ) : phase === "paused" ? (
                <Badge tone="warning" dot>
                  Paused
                </Badge>
              ) : null}
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <span className="tabular rounded-lg bg-surface-2 px-2.5 py-1.5 text-xs text-fg-muted">
              Generation{" "}
              <strong className="text-fg">{generation ? generation.generation + 1 : "—"}</strong>
              {run ? ` / ${run.config.max_generations}` : ""}
            </span>
            <span className="tabular rounded-lg bg-surface-2 px-2.5 py-1.5 text-xs text-fg-muted">
              Tick <strong className="text-fg">{frame ? formatNumber(frame.tick) : "—"}</strong>
              {generation ? ` / ${formatNumber(generation.max_ticks)}` : ""}
            </span>
            <button
              type="button"
              onClick={() => updatePreferences({ show_sensors: !prefs.show_sensors })}
              aria-pressed={prefs.show_sensors}
              title={prefs.show_sensors ? "Hide sensor rays" : "Show sensor rays"}
              className={cn(
                "inline-flex h-8 items-center gap-1.5 rounded-lg px-2.5 text-xs font-semibold transition-colors",
                prefs.show_sensors
                  ? "bg-primary-soft text-primary-soft-fg"
                  : "bg-surface-2 text-fg-muted hover:text-fg",
              )}
            >
              <Radar className="size-3.5" /> Sensors
            </button>
          </div>
        </header>

        <div className="relative h-[56dvh] min-h-[320px] bg-[var(--track-ground)] lg:h-auto lg:min-h-0 lg:flex-1">
          <TrackCanvas
            controller={controller}
            showSensors={prefs.show_sensors}
            label={`Simulation track. ${statusText}`}
          />
          {active ? (
            <div className="pointer-events-none absolute top-3 left-3 flex flex-wrap gap-2">
              <span className="rounded-lg bg-surface/90 px-2.5 py-1 text-xs font-semibold text-fg shadow-sm backdrop-blur">
                <CarFront aria-hidden className="mr-1 inline size-3.5 text-primary" />
                {frame?.alive ?? population} of {population} driving
              </span>
              {frame?.finished ? (
                <span className="rounded-lg bg-success-soft px-2.5 py-1 text-xs font-semibold text-success backdrop-blur">
                  <Flag aria-hidden className="mr-1 inline size-3.5" />
                  {frame.finished} finished
                </span>
              ) : null}
            </div>
          ) : null}
          <StageOverlay
            snapshot={snapshot}
            onRetry={start}
            onNewRun={() => {
              controller.reset()
              setTab("setup")
            }}
          />
          <p className="sr-only" aria-live="polite">
            {statusText}
          </p>
        </div>

        <ControlBar
          phase={phase}
          speed={snapshot.speed}
          confirmStop={prefs.confirm_before_stop}
          generationsDone={history.length}
          onStart={start}
          starting={connecting}
          onPause={() => controller.pause()}
          onResume={() => controller.resume()}
          onStep={() => controller.step()}
          onSkip={() => controller.skip()}
          onStop={() => controller.stop()}
          onSpeed={(speed) => controller.setSpeed(speed)}
        />
      </section>

      <aside
        aria-label="Inspector"
        className={cn(
          "flex flex-col border-border bg-bg-alt lg:h-dvh lg:w-[400px] lg:shrink-0 lg:border-l xl:w-[460px] 2xl:w-[540px]",
          !active && phase !== "ended" && !connecting && "order-first lg:order-none",
        )}
      >
        <div className="sticky top-16 z-20 flex items-center border-y border-border bg-surface px-2 py-2 lg:static lg:h-[76px] lg:shrink-0 lg:border-t-0 lg:py-0">
          <TabList
            items={TABS}
            value={tab}
            onChange={setTab}
            label="Inspector panels"
            idPrefix="inspector"
            stretch
            className="w-full [&>button]:px-2 [&>button]:py-2.5 [&>button]:text-xs sm:[&>button]:text-sm"
          />
        </div>
        <TabPanel
          idPrefix="inspector"
          value={tab}
          className="flex-1 p-4 sm:p-5 lg:min-h-0 lg:overflow-y-auto lg:overscroll-contain"
        >
          {tab === "setup" ? (
            active ? (
              <RunInProgress snapshot={snapshot} onStop={() => controller.stop()} />
            ) : (
              <div className="space-y-5">
                <div>
                  <h2 className="text-lg font-bold text-fg">Set Up a Training Run</h2>
                  <p className="mt-1 text-sm text-fg-muted">
                    Pick how many cars learn together and how long they train.
                  </p>
                </div>
                <RunSetup
                  bare
                  config={effectiveConfig}
                  onChange={setConfig}
                  onStart={start}
                  onReset={() => setConfig(defaultConfig(prefs))}
                  busy={connecting}
                  busyLabel={phase === "waking" ? "Waking Server" : "Starting"}
                />
              </div>
            )
          ) : null}
          {tab === "stats" ? (
            <StatsPanel snapshot={snapshot} lastGeneration={lastGeneration} />
          ) : null}
          {tab === "brain" ? (
            <div className="space-y-4">
              <div>
                <h2 className="text-lg font-bold text-fg">Leading Car&apos;s Brain</h2>
                <p className="mt-1 text-sm text-fg-muted">
                  {snapshot.network
                    ? `Genome #${snapshot.network.genome_id}. Brighter neurons are firing harder right now.`
                    : "The network appears once a run starts."}
                </p>
              </div>
              <div className="rounded-2xl border border-border bg-surface p-3">
                <NetworkView
                  network={snapshot.network}
                  values={frame?.trace.think.nodes}
                  height={400}
                />
              </div>
              <p className="text-xs leading-relaxed text-fg-subtle">
                Line thickness shows how strong a connection is. Blue connections pass a signal on,
                red ones push it the other way. Hidden neurons appear when evolution adds them.
              </p>
            </div>
          ) : null}
          {tab === "code" ? (
            <CodePanel
              code={snapshot.code}
              frame={frame}
              phase={phase}
              stepSeq={snapshot.stepSeq}
              lastGeneration={lastGeneration}
              guided={prefs.guided_steps}
            />
          ) : null}
          {tab === "history" ? <HistoryPanel history={history} /> : null}
        </TabPanel>
      </aside>

      <Dialog
        open={away.open}
        onClose={away.close}
        title="You Were Away, So We Paused"
        description="You left this tab for more than 5 seconds, so training paused to save server time. A paused run stops by itself after 5 minutes."
        icon={<Gauge className="size-5" />}
        size="sm"
        footer={
          <>
            <Button
              variant="danger-outline"
              onClick={() => {
                controller.stop()
                away.close()
              }}
            >
              <Square className="size-4" /> Stop Run
            </Button>
            <Button
              onClick={() => {
                controller.resume()
                away.close()
              }}
              data-autofocus
            >
              Continue Training
            </Button>
          </>
        }
      />
    </div>
  )
}

function RunInProgress({ snapshot, onStop }: { snapshot: SimSnapshot; onStop: () => void }) {
  const cfg = snapshot.run?.config
  const rows: [string, string][] = cfg
    ? [
        ["Cars per Generation", String(cfg.population_size)],
        ["Generation Limit", String(cfg.max_generations)],
        ["Track Length", cfg.track_length],
        ["Track Width", cfg.track_width],
        ["Curves", cfg.track_curviness],
        ["Track Variety", cfg.track_mode === "same" ? "Same track" : "New track each time"],
        ["Track Seed", String(snapshot.run?.seed ?? "Random")],
      ]
    : []
  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-bold text-fg">Run in Progress</h2>
        <p className="mt-1 text-sm text-fg-muted">
          Settings are locked while training. Stop this run to change them.
        </p>
      </div>
      {rows.length ? (
        <dl className="divide-y divide-border rounded-2xl border border-border bg-surface px-4">
          {rows.map(([label, value]) => (
            <div key={label} className="flex items-center justify-between gap-3 py-2.5 text-sm">
              <dt className="text-fg-muted">{label}</dt>
              <dd className="font-semibold text-fg capitalize">{value}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      <Button variant="danger-outline" className="w-full" onClick={onStop} disabled={!snapshot.run}>
        <Square className="size-4" /> Stop Run
      </Button>
    </div>
  )
}

function StatsPanel({
  snapshot,
  lastGeneration,
}: {
  snapshot: SimSnapshot
  lastGeneration: GenerationMessage | null
}) {
  const { generation, run, frame, history } = snapshot
  const series = useMemo(
    () => [
      {
        key: "best",
        label: "Best",
        color: "var(--chart-1)",
        values: history.map((g) => g.best_fitness),
      },
      {
        key: "mean",
        label: "Average",
        color: "var(--chart-2)",
        values: history.map((g) => g.mean_fitness),
      },
    ],
    [history],
  )
  if (!run) {
    return (
      <EmptyState
        icon={<Activity />}
        title="No Run Yet"
        description="Start a run from the Setup tab. Live statistics for every generation appear here."
      />
    )
  }
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        <StatCard
          label="Generation"
          value={generation ? generation.generation + 1 : "—"}
          hint={`of ${run.config.max_generations}`}
          icon={<Dna />}
        />
        <StatCard
          label="Leader"
          value={formatFitness(frame?.trace.score.fitness)}
          hint="Fitness right now"
          icon={<Gauge />}
          tone="success"
        />
        <StatCard
          label="Best Ever"
          value={formatFitness(snapshot.bestEver)}
          hint="Across generations"
          icon={<Trophy />}
          tone="warning"
        />
        <StatCard
          label="Species"
          value={lastGeneration?.species_count ?? 1}
          hint="Similar network groups"
          icon={<Activity />}
          tone="violet"
        />
      </div>
      <div className="rounded-2xl border border-border bg-surface p-4">
        <p className="text-sm font-bold text-fg">Fitness Over Time</p>
        <p className="mb-3 text-xs text-fg-subtle">Roughly distance driven, in hundreds of units</p>
        <LineChart
          series={series}
          xLabels={history.map((g) => String(g.generation + 1))}
          xTitle="Generation"
          ariaLabel={`Fitness by generation. ${history.length} generations recorded.`}
          emptyText="Fills in as each generation finishes."
          height={200}
        />
      </div>
      <div className="rounded-2xl border border-border bg-surface p-4">
        <p className="mb-3 text-sm font-bold text-fg">Latest Evolution Step</p>
        {lastGeneration ? (
          <dl className="grid grid-cols-2 gap-2">
            {(
              [
                ["Champions Kept", lastGeneration.evolution?.elites_kept ?? "—"],
                ["Children Bred", lastGeneration.evolution?.offspring ?? "—"],
                ["Hidden Neurons", Math.max(0, lastGeneration.best_genome_nodes - 4)],
                ["Connections", lastGeneration.best_genome_connections],
              ] as const
            ).map(([label, value]) => (
              <div key={label} className="rounded-xl bg-surface-2 p-3">
                <dt className="text-xs text-fg-subtle">{label}</dt>
                <dd className="tabular mt-0.5 text-lg font-bold text-fg">{value}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="text-sm text-fg-subtle">
            When every car in a generation stops, NEAT breeds the next one. Results appear here.
          </p>
        )}
      </div>
    </div>
  )
}

function HistoryPanel({ history }: { history: GenerationMessage[] }) {
  if (!history.length) {
    return (
      <EmptyState
        icon={<History />}
        title="No Generations Yet"
        description="Each finished generation adds a row here with its best and average fitness."
      />
    )
  }
  const best = history.reduce((a, g) => (g.best_fitness > a.best_fitness ? g : a), history[0])
  const rows = [...history].reverse()
  return (
    <div className="space-y-3">
      <div>
        <h2 className="text-lg font-bold text-fg">Generation History</h2>
        <p className="mt-1 text-sm text-fg-muted">
          Newest first. The best generation is highlighted.
        </p>
      </div>
      <TableWrap label="Generation history" compact>
        <THead>
          <tr>
            <th scope="col">Gen</th>
            <th scope="col" className="text-right">
              Best
            </th>
            <th scope="col" className="text-right">
              Average
            </th>
            <th scope="col" className="text-right">
              Hidden
            </th>
            <th scope="col" className="text-right">
              Done
            </th>
          </tr>
        </THead>
        <TBody>
          {rows.map((g) => (
            <tr key={g.generation} className={g === best ? "!bg-success-soft" : undefined}>
              <td className="font-semibold text-fg">
                {g.generation + 1}
                {g === best ? (
                  <Trophy aria-label="Best" className="ml-1.5 inline size-3.5 text-success" />
                ) : null}
              </td>
              <td className="tabular text-right font-semibold text-fg">
                {g.best_fitness.toFixed(1)}
              </td>
              <td className="tabular text-right text-fg-muted">{g.mean_fitness.toFixed(1)}</td>
              <td className="tabular text-right text-fg-muted">
                {Math.max(0, g.best_genome_nodes - 4)}
              </td>
              <td className="tabular text-right text-fg-muted">{g.finishers}</td>
            </tr>
          ))}
        </TBody>
      </TableWrap>
      <p className="text-xs text-fg-subtle">
        “Done” counts the cars that crossed the finish line in that generation.
      </p>
    </div>
  )
}
