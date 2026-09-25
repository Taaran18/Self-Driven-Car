"use client"

import { useId, useMemo, useState } from "react"
import { cn } from "@/lib/cn"
import type { NetworkNode, NetworkShape } from "@/lib/types"

interface NetworkViewProps {
  network: NetworkShape | null
  values?: Record<string, number>
  decisions?: Record<string, boolean>
  height?: number
  compact?: boolean
  flowing?: boolean
}

const WIDTH = 560
const DECISION_KEYS: Record<string, string> = {
  Accelerate: "accelerate",
  Brake: "brake",
  "Turn Left": "turn_left",
  "Turn Right": "turn_right",
}

const KIND_COLOR = {
  input: "var(--success)",
  hidden: "var(--violet)",
  output: "var(--warning)",
} as const

function curve(x1: number, y1: number, x2: number, y2: number) {
  const dx = (x2 - x1) * 0.5
  return `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`
}

export function NetworkView({
  network,
  values,
  decisions,
  height = 360,
  compact,
  flowing = false,
}: NetworkViewProps) {
  const glowId = useId().replace(/:/g, "")
  const [hovered, setHovered] = useState<number | null>(null)

  const layout = useMemo(() => {
    if (!network) return null
    const layers = new Map<number, NetworkNode[]>()
    for (const node of network.nodes) {
      const list = layers.get(node.layer) ?? []
      list.push(node)
      layers.set(node.layer, list)
    }
    const maxLayer = Math.max(...network.nodes.map((n) => n.layer), 1)
    const left = compact ? 28 : 128
    const right = compact ? 28 : 136
    const top = compact ? 18 : 44
    const bottom = 18
    const usable = height - top - bottom
    const positions = new Map<number, { x: number; y: number }>()
    const columns: { layer: number; x: number; count: number }[] = []
    for (const [layer, nodes] of [...layers.entries()].sort((a, b) => a[0] - b[0])) {
      const x = left + (layer / maxLayer) * (WIDTH - left - right)
      const gap = usable / Math.max(nodes.length, 1)
      nodes.forEach((node, i) => positions.set(node.id, { x, y: top + gap * (i + 0.5) }))
      columns.push({ layer, x, count: nodes.length })
    }
    return { positions, columns, maxLayer }
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
  const radius = compact ? 7 : 10
  const hasValues = Boolean(values)
  const connected = new Set<string>()
  if (hovered !== null) {
    for (const c of network.connections) {
      if (c.from === hovered || c.to === hovered) connected.add(`${c.from}-${c.to}`)
    }
  }

  const edges = network.connections
    .map((c) => {
      const a = layout.positions.get(c.from)
      const b = layout.positions.get(c.to)
      if (!a || !b) return null
      const key = `${c.from}-${c.to}`
      const strength = Math.min(1, Math.abs(c.weight) / 3)
      const source = values?.[String(c.from)] ?? 0
      const signal = hasValues ? Math.min(1, Math.abs(source * c.weight) / 2) : strength
      const dimmed = hovered !== null && !connected.has(key)
      const opacity = dimmed ? 0.05 : hovered !== null ? 0.95 : 0.12 + signal * 0.78
      return { c, a, b, key, strength, signal, opacity }
    })
    .filter((e): e is NonNullable<typeof e> => e !== null)
    .sort((x, y) => x.signal - y.signal)

  const layerTitle = (layer: number) =>
    layer === 0 ? "Input Layer" : layer === layout.maxLayer ? "Output Layer" : "Hidden"

  return (
    <figure>
      <svg
        viewBox={`0 0 ${WIDTH} ${height}`}
        className="h-auto w-full select-none"
        role="img"
        aria-label={`Neural network of genome ${network.genome_id}: ${network.nodes.length} neurons including ${hiddenCount} hidden, and ${network.connections.length} active connections.`}
        onPointerLeave={() => setHovered(null)}
      >
        <defs>
          <filter id={`glow-${glowId}`} x="-100%" y="-100%" width="300%" height="300%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {!compact
          ? layout.columns.map((col) => (
              <g key={col.layer}>
                <line
                  x1={col.x}
                  x2={col.x}
                  y1={30}
                  y2={height - 8}
                  stroke="var(--border)"
                  strokeDasharray="2 6"
                />
                <text
                  x={col.x}
                  y={16}
                  textAnchor="middle"
                  className="fill-fg-subtle text-[10px] font-semibold tracking-wider uppercase"
                >
                  {layerTitle(col.layer)} ({col.count})
                </text>
              </g>
            ))
          : null}

        {edges.map(({ c, a, b, key, strength, signal, opacity }) => {
          const d = curve(a.x + radius, a.y, b.x - radius, b.y)
          const color = c.weight >= 0 ? "var(--chart-1)" : "var(--danger)"
          const width = 0.6 + strength * 2.8
          return (
            <g key={key}>
              <path
                d={d}
                fill="none"
                stroke={color}
                strokeWidth={width}
                strokeOpacity={opacity}
                strokeLinecap="round"
              />
              {hasValues && signal > 0.18 && hovered === null ? (
                <path
                  d={d}
                  fill="none"
                  stroke={color}
                  strokeWidth={Math.max(1.2, width * 0.7)}
                  strokeOpacity={Math.min(1, 0.35 + signal)}
                  strokeLinecap="round"
                  className={cn("nn-flow", !flowing && "nn-flow-paused")}
                />
              ) : null}
            </g>
          )
        })}

        {network.nodes.map((node) => {
          const p = layout.positions.get(node.id)
          if (!p) return null
          const value = values?.[String(node.id)]
          const activation = Math.min(
            1,
            node.kind === "output" ? Math.max(0, value ?? 0) : Math.abs(value ?? 0),
          )
          const color = KIND_COLOR[node.kind]
          const decisionKey = DECISION_KEYS[node.label]
          const on = node.kind === "output" && decisionKey ? decisions?.[decisionKey] : false
          const active = on || (node.kind !== "output" && activation > 0.6)
          const isHovered = hovered === node.id
          return (
            <g key={node.id} onPointerEnter={() => setHovered(node.id)} className="cursor-pointer">
              <title>{`${node.label}${value !== undefined ? `: ${value.toFixed(3)}` : ""}`}</title>
              <circle cx={p.x} cy={p.y} r={radius + 10} fill="transparent" />
              {hasValues ? (
                <circle
                  cx={p.x}
                  cy={p.y}
                  r={radius + 4}
                  fill={color}
                  opacity={activation * 0.35}
                  filter={active ? `url(#glow-${glowId})` : undefined}
                />
              ) : null}
              <circle
                cx={p.x}
                cy={p.y}
                r={radius}
                fill="var(--surface)"
                stroke={color}
                strokeWidth={isHovered ? 3 : 2}
              />
              <circle
                cx={p.x}
                cy={p.y}
                r={hasValues ? Math.max(0, (radius - 3) * activation) : radius - 4}
                fill={color}
                opacity={hasValues ? 0.9 : 0.55}
              />
              {!compact && node.kind === "input" ? (
                <text
                  x={p.x - radius - 8}
                  y={p.y}
                  dy="0.32em"
                  textAnchor="end"
                  className="fill-fg-muted text-[11px]"
                >
                  {node.label}
                  {value !== undefined ? (
                    <tspan className="tabular fill-fg-subtle" dx="5">
                      {value.toFixed(2)}
                    </tspan>
                  ) : null}
                </text>
              ) : null}
              {!compact && node.kind === "output" ? (
                <text
                  x={p.x + radius + 8}
                  y={p.y}
                  dy="0.32em"
                  className={cn("text-[11px]", on ? "fill-fg font-semibold" : "fill-fg-muted")}
                >
                  {node.label}
                  {value !== undefined ? (
                    <tspan className="tabular fill-fg-subtle font-normal" dx="5">
                      {value.toFixed(2)}
                    </tspan>
                  ) : null}
                </text>
              ) : null}
              {!compact && on ? (
                <g transform={`translate(${p.x + radius + 8}, ${p.y + 9})`}>
                  <rect width="22" height="12" rx="6" fill="var(--success)" />
                  <text
                    x="11"
                    y="6"
                    dy="0.34em"
                    textAnchor="middle"
                    className="fill-[var(--bg)] text-[8px] font-bold"
                  >
                    ON
                  </text>
                </g>
              ) : null}
              {!compact && node.kind === "hidden" && isHovered && value !== undefined ? (
                <text
                  x={p.x}
                  y={p.y - radius - 6}
                  textAnchor="middle"
                  className="tabular fill-fg text-[10px] font-semibold"
                >
                  {value.toFixed(2)}
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
          Excites
        </span>
        <span className="flex items-center gap-1.5">
          <span aria-hidden className="h-0.5 w-4 bg-danger" />
          Inhibits
        </span>
        {hasValues ? (
          <span className="text-fg-subtle">Hover a neuron to trace its links</span>
        ) : null}
      </figcaption>
    </figure>
  )
}
