"use client"

import { useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/cn"

export interface Series {
  key: string
  label: string
  color: string
  values: (number | null)[]
}

interface LineChartProps {
  series: Series[]
  xLabels: string[]
  xTitle?: string
  height?: number
  format?: (value: number) => string
  ariaLabel: string
  className?: string
  emptyText?: string
}

function niceTicks(min: number, max: number, count = 4) {
  if (min === max) {
    min -= 1
    max += 1
  }
  const span = max - min
  const raw = span / count
  const magnitude = 10 ** Math.floor(Math.log10(raw))
  const step =
    [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => span / s <= count) ?? magnitude * 10
  const start = Math.floor(min / step) * step
  const end = Math.ceil(max / step) * step
  const ticks: number[] = []
  for (let v = start; v <= end + step / 2; v += step) ticks.push(Number(v.toFixed(10)))
  return ticks
}

export function LineChart({
  series,
  xLabels,
  xTitle,
  height = 260,
  format = (v) => v.toFixed(1),
  ariaLabel,
  className,
  emptyText = "No data yet.",
}: LineChartProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const [width, setWidth] = useState(0)
  const [hover, setHover] = useState<number | null>(null)

  useEffect(() => {
    const node = containerRef.current
    if (!node) return
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width))
    observer.observe(node)
    return () => observer.disconnect()
  }, [])

  const count = xLabels.length
  const values = series.flatMap((s) => s.values.filter((v): v is number => v !== null))
  const ticks = useMemo(
    () => (values.length ? niceTicks(Math.min(0, ...values), Math.max(...values)) : [0, 1]),
    [values],
  )
  const labelSpace = width < 480 ? 0 : 56
  const margin = { top: 12, right: 12 + labelSpace, bottom: xTitle ? 40 : 26, left: 44 }
  const innerW = Math.max(10, width - margin.left - margin.right)
  const innerH = height - margin.top - margin.bottom
  const yMin = ticks[0]
  const yMax = ticks[ticks.length - 1]
  const x = (i: number) => margin.left + (count <= 1 ? innerW / 2 : (i / (count - 1)) * innerW)
  const y = (v: number) => margin.top + innerH - ((v - yMin) / (yMax - yMin || 1)) * innerH
  const xTickEvery = Math.max(1, Math.ceil(count / Math.max(2, Math.floor(innerW / 70))))

  function onPointerMove(event: React.PointerEvent<SVGSVGElement>) {
    if (!count) return
    const rect = event.currentTarget.getBoundingClientRect()
    const px = event.clientX - rect.left - margin.left
    const index = count <= 1 ? 0 : Math.round((px / innerW) * (count - 1))
    setHover(Math.min(count - 1, Math.max(0, index)))
  }

  const tooltipLeft = hover !== null ? x(hover) : 0
  const flip = tooltipLeft > width - 180

  return (
    <div className={cn("space-y-3", className)}>
      {series.length > 1 ? (
        <ul
          className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-fg-muted"
          aria-label="Legend"
        >
          {series.map((s) => (
            <li key={s.key} className="flex items-center gap-2">
              <span
                aria-hidden
                className="h-0.5 w-4 rounded-full"
                style={{ background: s.color }}
              />
              {s.label}
            </li>
          ))}
        </ul>
      ) : null}
      <div
        ref={containerRef}
        className="relative w-full"
        style={{ height: count ? height : Math.min(height, 160) }}
      >
        {!count ? (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-border text-sm text-fg-subtle">
            {emptyText}
          </div>
        ) : width > 0 ? (
          <svg
            width={width}
            height={height}
            role="img"
            aria-label={ariaLabel}
            onPointerMove={onPointerMove}
            onPointerLeave={() => setHover(null)}
            className="touch-pan-y overflow-visible"
          >
            {ticks.map((tick) => (
              <g key={tick}>
                <line
                  x1={margin.left}
                  x2={margin.left + innerW}
                  y1={y(tick)}
                  y2={y(tick)}
                  stroke="var(--chart-grid)"
                  strokeWidth={1}
                />
                <text
                  x={margin.left - 8}
                  y={y(tick)}
                  dy="0.32em"
                  textAnchor="end"
                  className="tabular fill-fg-subtle text-[11px]"
                >
                  {format(tick)}
                </text>
              </g>
            ))}
            {xLabels.map((label, i) =>
              i % xTickEvery === 0 || i === count - 1 ? (
                <text
                  key={i}
                  x={x(i)}
                  y={margin.top + innerH + 18}
                  textAnchor="middle"
                  className="tabular fill-fg-subtle text-[11px]"
                >
                  {label}
                </text>
              ) : null,
            )}
            {xTitle ? (
              <text
                x={margin.left + innerW / 2}
                y={height - 4}
                textAnchor="middle"
                className="fill-fg-subtle text-[11px]"
              >
                {xTitle}
              </text>
            ) : null}
            {hover !== null ? (
              <line
                x1={x(hover)}
                x2={x(hover)}
                y1={margin.top}
                y2={margin.top + innerH}
                stroke="var(--border-strong)"
                strokeDasharray="3 3"
              />
            ) : null}
            {series.map((s) => {
              const points = s.values
                .map((v, i) => (v === null ? null : `${x(i)},${y(v)}`))
                .filter(Boolean)
                .join(" ")
              const lastIndex = s.values.reduce<number>((acc, v, i) => (v === null ? acc : i), -1)
              const lastValue = lastIndex >= 0 ? s.values[lastIndex] : null
              return (
                <g key={s.key}>
                  <polyline
                    points={points}
                    fill="none"
                    stroke={s.color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  {count === 1 && lastValue !== null ? (
                    <circle cx={x(0)} cy={y(lastValue)} r={4} fill={s.color} />
                  ) : null}
                  {lastValue !== null && labelSpace ? (
                    <text
                      x={x(lastIndex) + 8}
                      y={y(lastValue)}
                      dy="0.32em"
                      className="fill-fg-muted text-[11px] font-medium"
                    >
                      {s.label}
                    </text>
                  ) : null}
                  {hover !== null && s.values[hover] !== null ? (
                    <circle
                      cx={x(hover)}
                      cy={y(s.values[hover] as number)}
                      r={4.5}
                      fill={s.color}
                      stroke="var(--surface)"
                      strokeWidth={2}
                    />
                  ) : null}
                </g>
              )
            })}
          </svg>
        ) : null}
        {hover !== null && count ? (
          <div
            role="status"
            className="pointer-events-none absolute top-2 z-10 min-w-40 rounded-xl border border-border bg-surface px-3 py-2 text-xs shadow-[var(--shadow-pop)]"
            style={flip ? { right: width - tooltipLeft + 12 } : { left: tooltipLeft + 12 }}
          >
            <p className="font-semibold text-fg">
              {xTitle ?? "Point"} {xLabels[hover]}
            </p>
            <ul className="mt-1.5 space-y-1">
              {series.map((s) => (
                <li key={s.key} className="flex items-center justify-between gap-4">
                  <span className="flex items-center gap-1.5 text-fg-muted">
                    <span
                      aria-hidden
                      className="size-2 rounded-full"
                      style={{ background: s.color }}
                    />
                    {s.label}
                  </span>
                  <span className="tabular font-semibold text-fg">
                    {s.values[hover] === null ? "—" : format(s.values[hover] as number)}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </div>
    </div>
  )
}
