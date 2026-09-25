"use client"

import {
  ArrowLeft,
  BrainCircuit,
  Clock,
  Dna,
  FileText,
  LineChart as LineChartIcon,
  Pencil,
  SlidersHorizontal,
  Trash2,
  Trophy,
} from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useState } from "react"
import { LineChart } from "@/components/charts/line-chart"
import { NetworkView } from "@/components/simulator/network-view"
import { describeReason, StatusBadge } from "@/components/ui/badge"
import { Button, ButtonLink } from "@/components/ui/button"
import { Card, CardHeader, StatCard } from "@/components/ui/card"
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback"
import { Textarea } from "@/components/ui/field"
import { useToast } from "@/components/ui/toast"
import { useApi } from "@/hooks/use-api"
import { api, errorMessage } from "@/lib/api"
import { formatDateTime, formatDuration, formatFitness } from "@/lib/format"
import type { RunDetail, RunSummary } from "@/lib/types"
import { GenerationTable } from "./generation-table"
import { DeleteRunDialog, RenameRunDialog } from "./run-dialogs"

const configLabels: [keyof RunDetail["config"], string, (v: unknown) => string][] = [
  ["population_size", "Cars per Generation", (v) => String(v)],
  ["max_generations", "Generation Limit", (v) => String(v)],
  ["track_length", "Track Length", (v) => String(v).replace(/^\w/, (c) => c.toUpperCase())],
  ["track_width", "Track Width", (v) => String(v).replace(/^\w/, (c) => c.toUpperCase())],
  ["track_curviness", "Curves", (v) => String(v).replace(/^\w/, (c) => c.toUpperCase())],
  [
    "track_mode",
    "Track Variety",
    (v) => (v === "same" ? "Same track every generation" : "New track each generation"),
  ],
  ["track_seed", "Track Seed", (v) => (v === null || v === undefined ? "Random" : String(v))],
  ["weight_mutation_rate", "Weight Mutation", (v) => `${Math.round(Number(v) * 100)}%`],
  ["add_connection_rate", "New Connection", (v) => `${Math.round(Number(v) * 100)}%`],
  ["add_node_rate", "New Neuron", (v) => `${Math.round(Number(v) * 100)}%`],
]

export function RunDetailView({ id }: { id: string }) {
  const router = useRouter()
  const toast = useToast()
  const { data: run, error, loading, reload, mutate } = useApi<RunDetail>(`/api/runs/${id}`)
  const [renaming, setRenaming] = useState<RunSummary | null>(null)
  const [deleting, setDeleting] = useState<RunSummary | null>(null)
  const [notes, setNotes] = useState<string | null>(null)
  const [savingNotes, setSavingNotes] = useState(false)
  const currentNotes = notes ?? run?.notes ?? ""
  const notesChanged = notes !== null && notes !== (run?.notes ?? "")

  async function saveNotes() {
    setSavingNotes(true)
    try {
      await api.patch(`/api/runs/${id}`, { notes: currentNotes })
      mutate((r) => (r ? { ...r, notes: currentNotes.trim() || null } : r))
      setNotes(null)
      toast.success("Notes Saved")
    } catch (err) {
      toast.error("We Couldn't Save Your Notes", errorMessage(err))
    } finally {
      setSavingNotes(false)
    }
  }

  if (error && !run) {
    return (
      <div className="px-4 py-16 sm:px-6 lg:px-8">
        <Card className="mx-auto max-w-2xl">
          <ErrorState title="We Couldn't Load This Run" message={error} onRetry={reload} />
          <div className="pb-8 text-center">
            <ButtonLink href="/runs" variant="ghost">
              Back to All Runs
            </ButtonLink>
          </div>
        </Card>
      </div>
    )
  }

  if (loading && !run) {
    return (
      <div className="mx-auto max-w-[1600px] space-y-6 px-4 py-10 sm:px-6 lg:px-8">
        <Skeleton className="mx-auto h-14 w-2/3 max-w-xl" />
        <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-32 rounded-2xl" />
          ))}
        </div>
        <Skeleton className="h-80 rounded-2xl" />
      </div>
    )
  }

  if (!run) return null

  return (
    <div>
      <section className="bg-bg px-4 pt-8 pb-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1600px]">
          <Link
            href="/runs"
            className="inline-flex items-center gap-1.5 text-sm font-semibold text-fg-muted hover:text-fg"
          >
            <ArrowLeft className="size-4" /> All Runs
          </Link>
          <header className="mx-auto mt-4 flex max-w-3xl flex-col items-center text-center">
            <StatusBadge status={run.status} />
            <h1 className="mt-3 text-3xl font-bold tracking-tight break-words text-fg sm:text-4xl lg:text-5xl">
              {run.name}
            </h1>
            <p className="mt-3 text-fg-muted">
              Started {formatDateTime(run.created_at)} ·{" "}
              {describeReason(run.stop_reason ?? (run.status === "running" ? null : run.status))}
            </p>
            <div className="mt-5 flex flex-wrap justify-center gap-2">
              <Button variant="outline" onClick={() => setRenaming(run)}>
                <Pencil className="size-4" /> Rename
              </Button>
              <Button
                variant="danger-outline"
                onClick={() => setDeleting(run)}
                disabled={run.status === "running"}
              >
                <Trash2 className="size-4" /> Delete
              </Button>
            </div>
          </header>
        </div>
      </section>

      <section className="border-y border-border bg-bg-alt px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1600px] space-y-6">
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatCard
              label="Best Fitness"
              value={formatFitness(run.best_fitness)}
              hint={
                run.best_generation !== null
                  ? `Generation ${run.best_generation + 1}`
                  : "No generations yet"
              }
              icon={<Trophy />}
              tone="warning"
            />
            <StatCard
              label="Generations"
              value={`${run.generations_completed} / ${run.max_generations}`}
              hint={`${run.population_size} cars each`}
              icon={<Dna />}
              tone="violet"
            />
            <StatCard
              label="Training Time"
              value={formatDuration(run.duration_seconds)}
              hint={run.finished_at ? `Ended ${formatDateTime(run.finished_at)}` : "Still running"}
              icon={<Clock />}
              tone="success"
            />
            <StatCard
              label="Champion Size"
              value={
                run.champion
                  ? `${run.champion.nodes.filter((n) => n.kind === "hidden").length} hidden`
                  : "—"
              }
              hint={
                run.champion
                  ? `${run.champion.connections.length} connections`
                  : "Appears after a generation"
              }
              icon={<BrainCircuit />}
            />
          </div>
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
            <Card>
              <CardHeader
                title="Fitness Over Time"
                description="Best and average fitness for each generation"
                icon={<LineChartIcon />}
              />
              <div className="p-4 sm:p-5">
                <LineChart
                  series={[
                    {
                      key: "best",
                      label: "Best",
                      color: "var(--chart-1)",
                      values: run.generations.map((g) => g.best_fitness),
                    },
                    {
                      key: "mean",
                      label: "Average",
                      color: "var(--chart-2)",
                      values: run.generations.map((g) => g.mean_fitness),
                    },
                  ]}
                  xLabels={run.generations.map((g) => String(g.index + 1))}
                  xTitle="Generation"
                  ariaLabel={`Fitness by generation for ${run.name}.`}
                  emptyText="This run ended before its first generation finished."
                  height={300}
                />
              </div>
            </Card>
            <Card>
              <CardHeader
                title="Champion Network"
                description={
                  run.champion
                    ? `Genome #${run.champion.genome_id}, the best driver of this run`
                    : "The best network found"
                }
                icon={<BrainCircuit />}
              />
              <div className="p-4">
                {run.champion ? (
                  <NetworkView network={run.champion} height={340} />
                ) : (
                  <EmptyState
                    className="py-8"
                    icon={<BrainCircuit />}
                    title="No Champion Yet"
                    description="A champion is saved when the first generation finishes."
                  />
                )}
              </div>
            </Card>
          </div>
        </div>
      </section>

      <section className="bg-bg px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto grid max-w-[1600px] gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
          <div className="min-w-0 space-y-4">
            <h2 className="text-2xl font-bold text-fg">Generation History</h2>
            {run.generations.length ? (
              <GenerationTable rows={run.generations} />
            ) : (
              <Card>
                <EmptyState
                  icon={<Dna />}
                  title="No Generations Recorded"
                  description="This run stopped before any generation finished, so there's nothing to show yet."
                />
              </Card>
            )}
          </div>
          <div className="space-y-6">
            <Card>
              <CardHeader
                title="Notes"
                description="What did you try in this run?"
                icon={<FileText />}
              />
              <div className="space-y-3 p-4">
                <Textarea
                  aria-label="Run notes"
                  value={currentNotes}
                  onChange={(e) => setNotes(e.target.value)}
                  maxLength={2000}
                  placeholder="For example: narrow track with a high new-neuron chance."
                  rows={4}
                />
                <div className="flex items-center justify-between gap-3">
                  <span className="text-xs text-fg-subtle">{currentNotes.length} / 2000</span>
                  <Button
                    size="sm"
                    onClick={saveNotes}
                    disabled={!notesChanged}
                    loading={savingNotes}
                    loadingText="Saving"
                  >
                    Save Notes
                  </Button>
                </div>
              </div>
            </Card>
            <Card>
              <CardHeader title="Settings Used" icon={<SlidersHorizontal />} />
              <dl className="divide-y divide-border px-4 py-2">
                {configLabels.map(([key, label, format]) =>
                  key in run.config ? (
                    <div
                      key={key}
                      className="flex items-center justify-between gap-3 py-2.5 text-sm"
                    >
                      <dt className="text-fg-muted">{label}</dt>
                      <dd className="text-right font-semibold text-fg">
                        {format(run.config[key])}
                      </dd>
                    </div>
                  ) : null,
                )}
              </dl>
            </Card>
          </div>
        </div>
      </section>

      <RenameRunDialog
        run={renaming}
        onClose={() => setRenaming(null)}
        onRenamed={(updated) => mutate((r) => (r ? { ...r, name: updated.name } : r))}
      />
      <DeleteRunDialog
        run={deleting}
        onClose={() => setDeleting(null)}
        onDeleted={() => router.push("/runs")}
      />
    </div>
  )
}
