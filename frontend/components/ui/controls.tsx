"use client"

import { useId } from "react"
import { cn } from "@/lib/cn"

interface SwitchProps {
  checked: boolean
  onChange: (checked: boolean) => void
  label: string
  description?: string
  disabled?: boolean
}

export function Switch({ checked, onChange, label, description, disabled }: SwitchProps) {
  const id = useId()
  return (
    <div className="flex items-start justify-between gap-4">
      <div className="min-w-0">
        <label htmlFor={id} className="text-sm font-medium text-fg">
          {label}
        </label>
        {description ? <p className="mt-0.5 text-sm text-fg-muted">{description}</p> : null}
      </div>
      <button
        id={id}
        type="button"
        role="switch"
        aria-checked={checked}
        disabled={disabled}
        onClick={() => onChange(!checked)}
        className={cn(
          "relative mt-0.5 inline-flex h-6 w-11 shrink-0 items-center rounded-full border transition-colors disabled:opacity-50",
          checked ? "border-primary bg-primary" : "border-border-strong bg-surface-3",
        )}
      >
        <span
          aria-hidden
          className={cn(
            "inline-block size-4.5 rounded-full shadow transition-transform",
            checked ? "translate-x-[22px] bg-primary-fg" : "translate-x-[3px] bg-fg-subtle",
          )}
        />
      </button>
    </div>
  )
}

interface SegmentedProps<T extends string> {
  value: T
  onChange: (value: T) => void
  options: { value: T; label: string; icon?: React.ReactNode }[]
  label: string
  size?: "sm" | "md"
  className?: string
  disabled?: boolean
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  size = "md",
  className,
  disabled,
}: SegmentedProps<T>) {
  function onKeyDown(event: React.KeyboardEvent, index: number) {
    const delta =
      event.key === "ArrowRight" || event.key === "ArrowDown"
        ? 1
        : event.key === "ArrowLeft" || event.key === "ArrowUp"
          ? -1
          : 0
    if (!delta) return
    event.preventDefault()
    const next = options[(index + delta + options.length) % options.length]
    onChange(next.value)
    const group = (event.currentTarget as HTMLElement).parentElement
    requestAnimationFrame(() =>
      group?.querySelector<HTMLElement>(`[data-value="${next.value}"]`)?.focus(),
    )
  }
  return (
    <div
      role="radiogroup"
      aria-label={label}
      className={cn(
        "inline-flex w-full rounded-xl border border-border bg-surface-2 p-1",
        className,
      )}
    >
      {options.map((option, index) => {
        const selected = option.value === value
        return (
          <button
            key={option.value}
            type="button"
            role="radio"
            aria-checked={selected}
            data-value={option.value}
            tabIndex={selected ? 0 : -1}
            disabled={disabled}
            onClick={() => onChange(option.value)}
            onKeyDown={(e) => onKeyDown(e, index)}
            className={cn(
              "flex flex-1 items-center justify-center gap-1.5 rounded-lg font-semibold transition-colors disabled:opacity-50",
              size === "sm" ? "h-8 px-2 text-xs" : "h-9 px-3 text-sm",
              selected
                ? "bg-surface text-fg shadow-sm ring-1 ring-border"
                : "text-fg-muted hover:text-fg",
            )}
          >
            {option.icon}
            {option.label}
          </button>
        )
      })}
    </div>
  )
}

interface SliderProps {
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  label: string
  format?: (value: number) => string
  hint?: string
  disabled?: boolean
}

export function Slider({
  value,
  onChange,
  min,
  max,
  step,
  label,
  format,
  hint,
  disabled,
}: SliderProps) {
  const id = useId()
  const percent = ((value - min) / (max - min)) * 100
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between gap-3">
        <label htmlFor={id} className="text-sm font-medium text-fg">
          {label}
        </label>
        <span className="tabular rounded-md bg-surface-3 px-2 py-0.5 text-xs font-semibold text-fg">
          {format ? format(value) : value}
        </span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-valuetext={format ? format(value) : String(value)}
        className="h-2 w-full cursor-pointer appearance-none rounded-full bg-surface-3 accent-[var(--primary)] disabled:opacity-50 [&::-moz-range-thumb]:size-4 [&::-moz-range-thumb]:rounded-full [&::-moz-range-thumb]:border-2 [&::-moz-range-thumb]:border-[var(--surface)] [&::-moz-range-thumb]:bg-[var(--primary)] [&::-webkit-slider-thumb]:size-4.5 [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:border-2 [&::-webkit-slider-thumb]:border-[var(--surface)] [&::-webkit-slider-thumb]:bg-[var(--primary)] [&::-webkit-slider-thumb]:shadow"
        style={{
          background: `linear-gradient(to right, var(--primary) ${percent}%, var(--surface-3) ${percent}%)`,
        }}
      />
      {hint ? <p className="text-xs text-fg-subtle">{hint}</p> : null}
    </div>
  )
}
