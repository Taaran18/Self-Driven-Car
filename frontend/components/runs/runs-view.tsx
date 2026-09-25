"use client"

import {
  ChevronLeft,
  ChevronRight,
  Ellipsis,
  Eye,
  History,
  ListFilter,
  Pencil,
  Search,
  Trash2,
  X,
} from "lucide-react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { useEffect, useState } from "react"
import { StatusBadge, statusOptions } from "@/components/ui/badge"
import { Button, ButtonLink } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { EmptyState, ErrorState, Skeleton } from "@/components/ui/feedback"
import { Input } from "@/components/ui/field"
import { Menu } from "@/components/ui/menu"
import { PageHeader } from "@/components/ui/page-header"
import { Select } from "@/components/ui/select"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { useApi } from "@/hooks/use-api"
import { formatDateTime, formatDuration, formatFitness, formatRelative } from "@/lib/format"
import type { RunPage, RunSort, RunStatus, RunSummary } from "@/lib/types"
import { DeleteRunDialog, RenameRunDialog } from "./run-dialogs"

type StatusFilter = RunStatus | "all"

const sortOptions: { value: RunSort; label: string }[] = [
  { value: "newest", label: "Newest First" },
  { value: "oldest", label: "Oldest First" },
  { value: "best_fitness", label: "Best Fitness" },
  { value: "generations", label: "Most Generations" },
  { value: "name", label: "Name (A to Z)" },
]

const pageSizes = ["10", "20", "50"].map((v) => ({ value: v, label: `${v} per page` }))

export function RunsView() {
  const router = useRouter()
  const [search, setSearch] = useState("")
  const [query, setQuery] = useState("")
  const [status, setStatus] = useState<StatusFilter>("all")
  const [sort, setSort] = useState<RunSort>("newest")
  const [page, setPage] = useState(1)
  const [pageSize, setPageSize] = useState("10")
  const [renaming, setRenaming] = useState<RunSummary | null>(null)
  const [deleting, setDeleting] = useState<RunSummary | null>(null)

  useEffect(() => {
    const timer = setTimeout(() => {
      setQuery(search.trim())
      setPage(1)
    }, 300)
    return () => clearTimeout(timer)
  }, [search])

  const params = new URLSearchParams({ sort, page: String(page), page_size: pageSize })
  if (status !== "all") params.set("status", status)
  if (query) params.set("q", query)
  const { data, error, loading, reload, mutate } = useApi<RunPage>(`/api/runs?${params}`)
  const filtered = status !== "all" || Boolean(query)

  function clearFilters() {
    setSearch("")
    setQuery("")
    setStatus("all")
    setPage(1)
  }

  function actions(run: RunSummary) {
    return [
      { label: "View Details", icon: <Eye />, onSelect: () => router.push(`/runs/${run.id}`) },
      { label: "Rename", icon: <Pencil />, onSelect: () => setRenaming(run) },
      {
        label: "Delete",
        icon: <Trash2 />,
        tone: "danger" as const,
        onSelect: () => setDeleting(run),
        disabled: run.status === "running",
      },
    ]
  }

  return (
    <div>
      <section className="bg-bg px-4 pt-10 pb-10 sm:px-6 lg:px-8">
        <PageHeader
          eyebrow="History"
          title="Training Runs"
          description="Every run you've started from this browser and network, with its results and champion network."
        />
      </section>
      <section className="border-t border-border bg-bg-alt px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1600px] space-y-5">
          <Card className="grid gap-3 p-4 sm:grid-cols-2 lg:grid-cols-[minmax(0,1fr)_200px_200px_auto]">
            <div className="relative sm:col-span-2 lg:col-span-1">
              <Search
                aria-hidden
                className="pointer-events-none absolute top-1/2 left-3.5 size-4 -translate-y-1/2 text-fg-subtle"
              />
              <Input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="Search runs by name"
                aria-label="Search runs by name"
                className="pl-10"
                maxLength={80}
              />
            </div>
            <Select<StatusFilter>
              value={status}
              onChange={(v) => {
                setStatus(v)
                setPage(1)
              }}
              options={[{ value: "all", label: "All Statuses" }, ...statusOptions]}
              label="Filter by status"
              prefix={<ListFilter className="size-4" />}
            />
            <Select<RunSort>
              value={sort}
              onChange={(v) => {
                setSort(v)
                setPage(1)
              }}
              options={sortOptions}
              label="Sort runs"
            />
            {filtered ? (
              <Button variant="ghost" onClick={clearFilters}>
                <X className="size-4" /> Clear Filters
              </Button>
            ) : null}
          </Card>

          {error && !data ? (
            <Card>
              <ErrorState message={error} onRetry={reload} />
            </Card>
          ) : loading && !data ? (
            <Skeleton className="h-96 rounded-2xl" />
          ) : data && data.items.length ? (
            <>
              <div
                className={loading ? "opacity-60 transition-opacity" : "transition-opacity"}
                aria-busy={loading}
              >
                <div className="hidden md:block">
                  <TableWrap label="Training runs">
                    <THead>
                      <tr>
                        <th scope="col">Run</th>
                        <th scope="col">Status</th>
                        <th scope="col" className="text-right">
                          Cars
                        </th>
                        <th scope="col" className="text-right">
                          Generations
                        </th>
                        <th scope="col" className="text-right">
                          Best Fitness
                        </th>
                        <th scope="col" className="text-right">
                          Duration
                        </th>
                        <th scope="col" className="text-right">
                          Started
                        </th>
                        <th scope="col">
                          <span className="sr-only">Actions</span>
                        </th>
                      </tr>
                    </THead>
                    <TBody>
                      {data.items.map((run) => (
                        <tr key={run.id}>
                          <td className="max-w-[280px]">
                            <Link
                              href={`/runs/${run.id}`}
                              className="block truncate font-semibold text-fg hover:text-primary"
                            >
                              {run.name}
                            </Link>
                          </td>
                          <td>
                            <StatusBadge status={run.status} />
                          </td>
                          <td className="tabular text-right text-fg-muted">
                            {run.population_size}
                          </td>
                          <td className="tabular text-right text-fg-muted">
                            {run.generations_completed} / {run.max_generations}
                          </td>
                          <td className="tabular text-right font-semibold text-fg">
                            {formatFitness(run.best_fitness)}
                          </td>
                          <td className="tabular text-right text-fg-muted">
                            {formatDuration(run.duration_seconds)}
                          </td>
                          <td
                            className="text-right whitespace-nowrap text-fg-muted"
                            title={formatDateTime(run.created_at)}
                          >
                            {formatRelative(run.created_at)}
                          </td>
                          <td className="w-12 text-right">
                            <Menu
                              label={`Actions for ${run.name}`}
                              items={actions(run)}
                              triggerClassName="inline-flex size-9 items-center justify-center rounded-lg text-fg-muted hover:bg-surface-3 hover:text-fg"
                            >
                              <Ellipsis className="size-5" />
                            </Menu>
                          </td>
                        </tr>
                      ))}
                    </TBody>
                  </TableWrap>
                </div>
                <ul className="space-y-3 md:hidden">
                  {data.items.map((run) => (
                    <li key={run.id}>
                      <Card className="p-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <Link
                              href={`/runs/${run.id}`}
                              className="block truncate font-semibold text-fg"
                            >
                              {run.name}
                            </Link>
                            <p className="mt-1 text-xs text-fg-subtle">
                              {formatRelative(run.created_at)}
                            </p>
                          </div>
                          <Menu
                            label={`Actions for ${run.name}`}
                            items={actions(run)}
                            triggerClassName="-mt-1 -mr-1 inline-flex size-9 shrink-0 items-center justify-center rounded-lg text-fg-muted hover:bg-surface-3"
                          >
                            <Ellipsis className="size-5" />
                          </Menu>
                        </div>
                        <div className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-2 text-sm">
                          <StatusBadge status={run.status} />
                          <span className="tabular text-fg-muted">
                            Gen {run.generations_completed}/{run.max_generations}
                          </span>
                          <span className="tabular font-semibold text-fg">
                            Best {formatFitness(run.best_fitness)}
                          </span>
                        </div>
                      </Card>
                    </li>
                  ))}
                </ul>
              </div>
              <nav
                aria-label="Pagination"
                className="flex flex-col items-center justify-between gap-3 sm:flex-row"
              >
                <p className="text-sm text-fg-muted">
                  Showing {(data.page - 1) * data.page_size + 1}–
                  {Math.min(data.total, data.page * data.page_size)} of {data.total} runs
                </p>
                <div className="flex items-center gap-2">
                  <div className="w-36">
                    <Select
                      size="sm"
                      value={pageSize}
                      onChange={(v) => {
                        setPageSize(v)
                        setPage(1)
                      }}
                      options={pageSizes}
                      label="Runs per page"
                    />
                  </div>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    onClick={() => setPage((p) => p - 1)}
                    disabled={data.page <= 1}
                    aria-label="Previous page"
                  >
                    <ChevronLeft className="size-4" />
                  </Button>
                  <span className="tabular px-1 text-sm text-fg-muted">
                    Page {data.page} of {data.pages}
                  </span>
                  <Button
                    variant="outline"
                    size="icon-sm"
                    onClick={() => setPage((p) => p + 1)}
                    disabled={data.page >= data.pages}
                    aria-label="Next page"
                  >
                    <ChevronRight className="size-4" />
                  </Button>
                </div>
              </nav>
            </>
          ) : data && filtered ? (
            <Card>
              <EmptyState
                icon={<Search />}
                title="No Runs Match These Filters"
                description="Try a different name or status, or clear the filters to see every run."
                action={<Button onClick={clearFilters}>Clear Filters</Button>}
              />
            </Card>
          ) : data ? (
            <Card>
              <EmptyState
                icon={<History />}
                title="You Haven't Started Any Runs Yet"
                description="Each run you start in the simulator is saved here, with every generation and the best network it found."
                action={<ButtonLink href="/simulator">Start Your First Run</ButtonLink>}
              />
            </Card>
          ) : null}
        </div>
      </section>
      <RenameRunDialog
        run={renaming}
        onClose={() => setRenaming(null)}
        onRenamed={(updated) =>
          mutate((d) =>
            d ? { ...d, items: d.items.map((r) => (r.id === updated.id ? updated : r)) } : d,
          )
        }
      />
      <DeleteRunDialog
        run={deleting}
        onClose={() => setDeleting(null)}
        onDeleted={() => reload()}
      />
    </div>
  )
}
