import { cn } from "@/lib/cn"
import type { RunStatus } from "@/lib/types"

type Tone = "neutral" | "primary" | "success" | "warning" | "danger" | "violet"

const tones: Record<Tone, string> = {
  neutral: "bg-surface-3 text-fg-muted",
  primary: "bg-primary-soft text-primary-soft-fg",
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
  danger: "bg-danger-soft text-danger",
  violet: "bg-violet-soft text-violet",
}

export function Badge({
  tone = "neutral",
  className,
  dot,
  children,
}: {
  tone?: Tone
  className?: string
  dot?: boolean
  children: React.ReactNode
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap",
        tones[tone],
        className,
      )}
    >
      {dot ? <span aria-hidden className="size-1.5 rounded-full bg-current" /> : null}
      {children}
    </span>
  )
}

const statusMeta: Record<RunStatus, { label: string; tone: Tone }> = {
  running: { label: "Running", tone: "primary" },
  completed: { label: "Completed", tone: "success" },
  stopped: { label: "Stopped", tone: "neutral" },
  interrupted: { label: "Interrupted", tone: "warning" },
  failed: { label: "Failed", tone: "danger" },
}

export const statusOptions = Object.entries(statusMeta).map(([value, meta]) => ({
  value: value as RunStatus,
  label: meta.label,
}))

export function StatusBadge({ status }: { status: RunStatus }) {
  const meta = statusMeta[status]
  return (
    <Badge tone={meta.tone} dot>
      {meta.label}
    </Badge>
  )
}

const reasonLabels: Record<string, string> = {
  solved: "A car reached the finish line",
  max_generations: "Reached the generation limit",
  user: "Stopped by you",
  disconnected: "Browser tab closed or lost connection",
  replaced: "Replaced by a run in another tab",
  idle: "Paused for too long",
  time_limit: "Reached the time limit",
  server_restart: "The server restarted",
  server_shutdown: "The server shut down",
  engine_error: "The simulation hit an error",
  data_deleted: "Your data was deleted",
}

export function describeReason(reason: string | null | undefined): string {
  if (!reason) return "—"
  return reasonLabels[reason] ?? reason.replace(/_/g, " ")
}
