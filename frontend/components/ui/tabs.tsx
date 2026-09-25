"use client"

import { cn } from "@/lib/cn"

export interface TabItem<T extends string> {
  value: T
  label: string
  icon?: React.ReactNode
  badge?: React.ReactNode
}

interface TabListProps<T extends string> {
  items: TabItem<T>[]
  value: T
  onChange: (value: T) => void
  label: string
  idPrefix: string
  orientation?: "horizontal" | "vertical"
  className?: string
  variant?: "pill" | "underline"
  stretch?: boolean
}

export function TabList<T extends string>({
  items,
  value,
  onChange,
  label,
  idPrefix,
  orientation = "horizontal",
  className,
  variant = "pill",
  stretch = false,
}: TabListProps<T>) {
  function onKeyDown(event: React.KeyboardEvent, index: number) {
    let next = -1
    if (event.key === "ArrowRight" || event.key === "ArrowDown") next = (index + 1) % items.length
    if (event.key === "ArrowLeft" || event.key === "ArrowUp")
      next = (index - 1 + items.length) % items.length
    if (event.key === "Home") next = 0
    if (event.key === "End") next = items.length - 1
    if (next < 0) return
    event.preventDefault()
    onChange(items[next].value)
    document.getElementById(`${idPrefix}-tab-${items[next].value}`)?.focus()
  }

  return (
    <div
      role="tablist"
      aria-label={label}
      aria-orientation={orientation}
      className={cn(
        !className?.split(" ").includes("grid") && "flex",
        orientation === "vertical"
          ? "flex-col gap-1"
          : "gap-1 overflow-x-auto overscroll-x-contain",
        variant === "underline" && orientation === "horizontal" && "border-b border-border",
        className,
      )}
    >
      {items.map((item, index) => {
        const selected = item.value === value
        return (
          <button
            key={item.value}
            id={`${idPrefix}-tab-${item.value}`}
            type="button"
            role="tab"
            aria-selected={selected}
            aria-controls={`${idPrefix}-panel-${item.value}`}
            tabIndex={selected ? 0 : -1}
            onClick={() => onChange(item.value)}
            onKeyDown={(e) => onKeyDown(e, index)}
            className={cn(
              "flex shrink-0 items-center gap-2 text-sm font-semibold whitespace-nowrap transition-colors [&>svg]:size-4",
              stretch && "flex-1 justify-center",
              variant === "pill" &&
                cn(
                  "rounded-xl px-3.5 py-2.5",
                  selected
                    ? "bg-primary-soft text-primary-soft-fg"
                    : "text-fg-muted hover:bg-surface-3 hover:text-fg",
                ),
              variant === "underline" &&
                cn(
                  "-mb-px border-b-2 px-3 py-2.5",
                  selected
                    ? "border-primary text-fg"
                    : "border-transparent text-fg-muted hover:text-fg",
                ),
            )}
          >
            {item.icon}
            {item.label}
            {item.badge}
          </button>
        )
      })}
    </div>
  )
}

export function TabPanel({
  idPrefix,
  value,
  children,
  className,
}: {
  idPrefix: string
  value: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div
      role="tabpanel"
      id={`${idPrefix}-panel-${value}`}
      aria-labelledby={`${idPrefix}-tab-${value}`}
      tabIndex={0}
      className={cn("outline-none", className)}
    >
      {children}
    </div>
  )
}
