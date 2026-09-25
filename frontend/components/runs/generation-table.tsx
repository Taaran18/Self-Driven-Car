"use client"

import { Trophy } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { formatNumber } from "@/lib/format"

export interface GenerationRow {
  index: number
  best_fitness: number
  mean_fitness: number
  species_count: number
  best_genome_nodes: number
  best_genome_connections: number
  ticks: number
  duration_ms: number
  finishers?: number
}

export function GenerationTable({
  rows,
  initial = 10,
}: {
  rows: GenerationRow[]
  initial?: number
}) {
  const [showAll, setShowAll] = useState(false)
  const ordered = [...rows].sort((a, b) => b.index - a.index)
  const visible = showAll ? ordered : ordered.slice(0, initial)
  const best = rows.reduce<GenerationRow | null>(
    (acc, r) => (!acc || r.best_fitness > acc.best_fitness ? r : acc),
    null,
  )
  const hasFinishers = rows.some((r) => r.finishers !== undefined)

  return (
    <div className="space-y-3">
      <TableWrap label="Generation history">
        <THead>
          <tr>
            <th scope="col">Generation</th>
            <th scope="col" className="text-right">
              Best Fitness
            </th>
            <th scope="col" className="text-right">
              Average
            </th>
            <th scope="col" className="text-right">
              Species
            </th>
            <th scope="col" className="text-right">
              Hidden Neurons
            </th>
            <th scope="col" className="text-right">
              Connections
            </th>
            {hasFinishers ? (
              <th scope="col" className="text-right">
                Finished
              </th>
            ) : null}
            <th scope="col" className="text-right">
              Ticks
            </th>
            <th scope="col" className="text-right">
              Time
            </th>
          </tr>
        </THead>
        <TBody>
          {visible.map((row) => {
            const isBest = best?.index === row.index
            return (
              <tr key={row.index} className={isBest ? "!bg-success-soft" : undefined}>
                <td className="font-semibold text-fg">
                  <span className="inline-flex items-center gap-2">
                    {row.index + 1}
                    {isBest ? (
                      <span className="inline-flex items-center gap-1 text-xs font-bold text-success">
                        <Trophy className="size-3.5" aria-hidden /> Best
                      </span>
                    ) : null}
                  </span>
                </td>
                <td className="tabular text-right font-semibold text-fg">
                  {row.best_fitness.toFixed(1)}
                </td>
                <td className="tabular text-right text-fg-muted">{row.mean_fitness.toFixed(1)}</td>
                <td className="tabular text-right text-fg-muted">{row.species_count}</td>
                <td className="tabular text-right text-fg-muted">
                  {Math.max(0, row.best_genome_nodes - 4)}
                </td>
                <td className="tabular text-right text-fg-muted">{row.best_genome_connections}</td>
                {hasFinishers ? (
                  <td className="tabular text-right text-fg-muted">{row.finishers ?? 0}</td>
                ) : null}
                <td className="tabular text-right text-fg-muted">{formatNumber(row.ticks)}</td>
                <td className="tabular text-right text-fg-muted">
                  {(row.duration_ms / 1000).toFixed(1)}s
                </td>
              </tr>
            )
          })}
        </TBody>
      </TableWrap>
      {ordered.length > initial ? (
        <div className="flex justify-center">
          <Button variant="ghost" size="sm" onClick={() => setShowAll((v) => !v)}>
            {showAll ? "Show Latest Only" : `Show All ${ordered.length} Generations`}
          </Button>
        </div>
      ) : null}
    </div>
  )
}
