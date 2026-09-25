"use client"

import {
  ArrowRight,
  CarFront,
  Clock,
  Dna,
  Gauge,
  History,
  LineChart as LineChartIcon,
  Trophy,
} from "lucide-react"
import Link from "next/link"
import { useEffect } from "react"
import { LineChart } from "@/components/charts/line-chart"
import { StatusBadge } from "@/components/ui/badge"
import { ButtonLink } from "@/components/ui/button"
import { Card, CardHeader, StatCard } from "@/components/ui/card"
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback"
import { PageHeader } from "@/components/ui/page-header"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { useApi } from "@/hooks/use-api"
import {
  formatDateTime,
  formatDuration,
  formatFitness,
  formatNumber,
  formatRelative,
} from "@/lib/format"
import type { OverviewStats } from "@/lib/types"
import { refreshUsage, useUsage } from "@/lib/usage"
import { UsageMeter } from "./usage-meter"

export function DashboardView() {
  const { data, error, loading, reload } = useApi<OverviewStats>("/api/me/overview")
  const usage = useUsage()

  useEffect(() => {
    refreshUsage().catch(() => {})
  }, [])

  const trend = data?.trend.filter((t) => t.best_fitness !== null) ?? []

  return (
    <div>
      <section className="bg-bg px-4 pt-10 pb-10 sm:px-6 lg:px-8">
        <PageHeader
          eyebrow="Your Workspace"
          title="Training Dashboard"
          description="Your runs, your best drivers, and how much of today's free training you have left."
          actions={
            <ButtonLink href="/simulator">
              Start a New Run <ArrowRight className="size-4" />
            </ButtonLink>
          }
        />
      </section>

      <section className="border-y border-border bg-bg-alt px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1600px] space-y-6">
          {error && !data ? (
            <Card>
              <ErrorState message={error} onRetry={reload} />
            </Card>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                {loading && !data ? (
                  Array.from({ length: 4 }, (_, i) => (
                    <Skeleton key={i} className="h-32 rounded-2xl" />
                  ))
                ) : (
                  <>
                    <StatCard
                      label="Training Runs"
                      value={formatNumber(data?.total_runs)}
                      hint={`${data?.completed_runs ?? 0} completed`}
                      icon={<History />}
                    />
                    <StatCard
                      label="Generations Trained"
                      value={formatNumber(data?.total_generations)}
                      hint="Across all runs"
                      icon={<Dna />}
                      tone="violet"
                    />
                    <StatCard
                      label="Best Fitness"
                      value={formatFitness(data?.best_fitness)}
                      hint={data?.best_run_name ?? "No runs yet"}
                      icon={<Trophy />}
                      tone="warning"
                    />
                    <StatCard
                      label="Training Time"
                      value={formatDuration(data?.training_seconds)}
                      hint="Time spent evolving"
                      icon={<Clock />}
                      tone="success"
                    />
                  </>
                )}
              </div>

              <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_380px]">
                <Card>
                  <CardHeader
                    title="Best Fitness by Run"
                    description="Your last 20 runs, oldest to newest"
                    icon={<LineChartIcon />}
                  />
                  <div className="p-4 sm:p-5">
                    {loading && !data ? (
                      <Skeleton className="h-[260px]" />
                    ) : (
                      <LineChart
                        series={[
                          {
                            key: "best",
                            label: "Best fitness",
                            color: "var(--chart-1)",
                            values: trend.map((t) => t.best_fitness),
                          },
                        ]}
                        xLabels={trend.map((_, i) => String(i + 1))}
                        xTitle="Run"
                        ariaLabel={`Best fitness for your last ${trend.length} runs.`}
                        emptyText="Finish a generation in the simulator to see your first point here."
                      />
                    )}
                  </div>
                </Card>
                <Card>
                  <CardHeader
                    title="Free Trial Usage"
                    description="Counted per network (IP address)"
                    icon={<Gauge />}
                  />
                  <div className="space-y-6 p-5">
                    <UsageMeter
                      label="Runs Today"
                      used={usage.used_today}
                      limit={usage.runs_per_day}
                      hint={
                        usage.day_resets_at
                          ? `Resets ${formatDateTime(usage.day_resets_at)} (your time)`
                          : undefined
                      }
                    />
                    <UsageMeter
                      label="Runs This Week"
                      used={usage.used_this_week}
                      limit={usage.runs_per_week}
                      hint={
                        usage.week_resets_at
                          ? `Resets ${formatDateTime(usage.week_resets_at)} (your time)`
                          : undefined
                      }
                    />
                    <Link
                      href="/settings?tab=usage"
                      className="inline-flex text-sm font-semibold text-primary hover:underline"
                    >
                      How limits work
                    </Link>
                  </div>
                </Card>
              </div>
            </>
          )}
        </div>
      </section>

      <section className="bg-bg px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1600px]">
          <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
            <div>
              <h2 className="text-2xl font-bold text-fg">Recent Runs</h2>
              <p className="mt-1 text-sm text-fg-muted">Your five latest training runs.</p>
            </div>
            <ButtonLink href="/runs" variant="outline" size="sm">
              View All Runs
            </ButtonLink>
          </div>
          {loading && !data ? (
            <Skeleton className="h-64 rounded-2xl" />
          ) : data && data.recent_runs.length ? (
            <TableWrap label="Recent runs">
              <THead>
                <tr>
                  <th scope="col">Run</th>
                  <th scope="col">Status</th>
                  <th scope="col" className="text-right">
                    Generations
                  </th>
                  <th scope="col" className="text-right">
                    Best Fitness
                  </th>
                  <th scope="col" className="text-right">
                    Started
                  </th>
                </tr>
              </THead>
              <TBody>
                {data.recent_runs.map((run) => (
                  <tr key={run.id}>
                    <td>
                      <Link
                        href={`/runs/${run.id}`}
                        className="font-semibold text-fg hover:text-primary"
                      >
                        {run.name}
                      </Link>
                    </td>
                    <td>
                      <StatusBadge status={run.status} />
                    </td>
                    <td className="tabular text-right text-fg-muted">
                      {run.generations_completed} / {run.max_generations}
                    </td>
                    <td className="tabular text-right font-semibold text-fg">
                      {formatFitness(run.best_fitness)}
                    </td>
                    <td className="text-right text-fg-muted" title={formatDateTime(run.created_at)}>
                      {formatRelative(run.created_at)}
                    </td>
                  </tr>
                ))}
              </TBody>
            </TableWrap>
          ) : data ? (
            <Card>
              <EmptyState
                icon={<CarFront />}
                title="No Training Runs Yet"
                description="Start your first run and this dashboard will fill with generations, fitness scores, and champion networks."
                action={<ButtonLink href="/simulator">Open the Simulator</ButtonLink>}
              />
            </Card>
          ) : null}
        </div>
      </section>
    </div>
  )
}
