import { cn } from "@/lib/cn"

export function TableWrap({
  className,
  children,
  label,
  compact,
}: {
  className?: string
  children: React.ReactNode
  label?: string
  compact?: boolean
}) {
  return (
    <div
      role="region"
      aria-label={label}
      tabIndex={label ? 0 : undefined}
      className={cn("overflow-x-auto rounded-2xl border border-border bg-surface", className)}
    >
      <table
        className={cn(
          "w-full border-collapse text-left text-sm",
          compact ? "[&_td]:!px-3 [&_td]:!py-2.5 [&_th]:!px-3" : "min-w-[640px]",
        )}
      >
        {children}
      </table>
    </div>
  )
}

export function THead({ children }: { children: React.ReactNode }) {
  return (
    <thead className="bg-surface-2 text-xs font-semibold tracking-wider text-fg-subtle uppercase [&_th]:border-b [&_th]:border-border [&_th]:px-4 [&_th]:py-3 [&_th]:whitespace-nowrap">
      {children}
    </thead>
  )
}

export function TBody({ children }: { children: React.ReactNode }) {
  return (
    <tbody className="[&_td]:border-b [&_td]:border-border [&_td]:px-4 [&_td]:py-3.5 [&_tr]:transition-colors [&_tr:hover]:bg-primary-soft/60 [&_tr:last-child_td]:border-b-0 [&_tr:nth-child(even)]:bg-bg-alt/60">
      {children}
    </tbody>
  )
}
