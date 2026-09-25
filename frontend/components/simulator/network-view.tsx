"use client"

import { useMemo } from "react"
import type { NetworkShape } from "@/lib/types"

interface NetworkViewProps {
  network: NetworkShape | null
  values?: Record<string, number>
  height?: number
  compact?: boolean
}

const WIDTH = 520

export function NetworkView({ network, values, height = 340, compact }: NetworkViewProps) {
  const layout = useMemo(() => {
    if (!network) return null
    const layers = new Map<number, typeof network.nodes>()
    for (const node of network.nodes) {
      const list = layers.get(node.layer) ?? []
      list.push(node)
      layers.set(node.layer, list)
    }
    const maxLayer = Math.max(...network.nodes.map((n) => n.layer), 1)
    const left = compact ? 24 : 110
    const right = compact ? 24 : 96
    const top = 18
    const usable = height - top * 2
    const positions = new Map<number, { x: number; y: number }>()
    for (const [layer, nodes] of layers) {
      const x = left + (layer / maxLayer) * (WIDTH - left - right)
      const gap = usable / Math.max(nodes.length, 1)
      nodes.forEach((node, i) => positions.set(node.id, { x, y: top + gap * (i + 0.5) }))
    }
    return { positions, maxLayer }
  }, [network, height, compact])

  if (!network || !layout) {
    return (
      <div
        className="flex items-center justify-center rounded-xl border border-dashed border-border px-4 text-center text-sm text-fg-subtle"
        style={{ height }}
      >
        The leading car&apos;s neural network appears here once a run starts.
      </div>
    )
  }

  const hiddenCount = network.nodes.filter((n) => n.kind === "hidden").length
  const radius = compact ? 6 : 8

  return (
    <figure>
      <svg
        viewBox={`0 0 ${WIDTH} ${height}`}
        className="h-auto w-full"
        role="img"
        aria-label={`Neural network of genome ${network.genome_id}: ${network.nodes.length} neurons including ${hiddenCount} hidden, and ${network.connections.length} active connections.`}
      >
        {network.connections.map((c) => {
          const a = layout.positions.get(c.from)
          const b = layout.positions.get(c.to)
          if (!a || !b) return null
          const strength = Math.min(1, Math.abs(c.weight) / 3)
          return (
            <line
              key={`${c.from}-${c.to}`}
              x1={a.x}
              y1={a.y}
              x2={b.x}
              y2={b.y}
              stroke={c.weight >= 0 ? "var(--chart-1)" : "var(--danger)"}
              strokeWidth={0.6 + strength * 2.6}
              strokeOpacity={0.25 + strength * 0.55}
            />
          )
        })}
        {network.nodes.map((node) => {
          const p = layout.positions.get(node.id)
          if (!p) return null
          const value = values?.[String(node.id)] ?? 0
          const activation = Math.min(1, Math.abs(value))
          const color =
            node.kind === "input"
              ? "var(--success)"
              : node.kind === "output"
                ? "var(--warning)"
                : "var(--violet)"
          return (
            <g key={node.id}>
              <circle cx={p.x} cy={p.y} r={radius + 3} fill={color} opacity={activation * 0.35} />
              <circle
                cx={p.x}
                cy={p.y}
                r={radius}
                fill="var(--surface)"
                stroke={color}
                strokeWidth={2}
              />
              <circle cx={p.x} cy={p.y} r={Math.max(1.5, radius * activation)} fill={color} />
              {!compact && node.kind !== "hidden" ? (
                <text
                  x={node.kind === "input" ? p.x - radius - 6 : p.x + radius + 6}
                  y={p.y}
                  dy="0.32em"
                  textAnchor={node.kind === "input" ? "end" : "start"}
                  className="fill-fg-muted text-[11px]"
                >
                  {node.label}
                </text>
              ) : null}
            </g>
          )
        })}
      </svg>
      <figcaption className="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-fg-muted">
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-full border-2 border-success" />
          Inputs
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-full border-2 border-violet" />
          Hidden ({hiddenCount})
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="size-2.5 rounded-full border-2 border-warning" />
          Outputs
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-0.5 w-4 bg-chart-1" />
          Positive weight
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-0.5 w-4 bg-danger" />
          Negative weight
        </span>
      </figcaption>
    </figure>
  )
}
