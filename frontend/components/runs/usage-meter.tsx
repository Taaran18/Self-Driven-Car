import { cn } from "@/lib/cn"

export function UsageMeter({
  label,
  used,
  limit,
  hint,
}: {
  label: string
  used: number
  limit: number
  hint?: string
}) {
  const left = Math.max(0, limit - used)
  const ratio = limit ? Math.min(1, used / limit) : 0
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <p className="text-sm font-semibold text-fg">{label}</p>
        <p className="tabular text-sm text-fg-muted">
          <strong className="text-fg">{left}</strong> of {limit} left
        </p>
      </div>
      <div
        className="mt-2 h-2.5 overflow-hidden rounded-full bg-surface-3"
        role="meter"
        aria-label={label}
        aria-valuemin={0}
        aria-valuemax={limit}
        aria-valuenow={used}
        aria-valuetext={`${used} of ${limit} used, ${left} left`}
      >
        <div
          className={cn(
            "h-full rounded-full transition-[width] duration-500",
            ratio >= 0.8 ? "bg-danger" : "bg-primary",
          )}
          style={{ width: `${ratio * 100}%` }}
        />
      </div>
      {hint ? <p className="mt-1.5 text-xs text-fg-subtle">{hint}</p> : null}
    </div>
  )
}
