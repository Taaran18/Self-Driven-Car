import { CircleAlert, LoaderCircle } from "lucide-react"
import { cn } from "@/lib/cn"
import { Button } from "./button"

export function Skeleton({ className }: { className?: string }) {
  return <div aria-hidden className={cn("animate-pulse rounded-lg bg-surface-3", className)} />
}

export function Spinner({ label = "Loading", className }: { label?: string; className?: string }) {
  return (
    <span
      role="status"
      className={cn("inline-flex items-center gap-2 text-sm text-fg-muted", className)}
    >
      <LoaderCircle aria-hidden className="size-4 animate-spin" />
      {label}
    </span>
  )
}

interface EmptyStateProps {
  icon: React.ReactNode
  title: string
  description: React.ReactNode
  action?: React.ReactNode
  className?: string
}

export function EmptyState({ icon, title, description, action, className }: EmptyStateProps) {
  return (
    <div className={cn("flex flex-col items-center px-6 py-14 text-center", className)}>
      <div className="flex size-14 items-center justify-center rounded-2xl bg-primary-soft text-primary-soft-fg [&>svg]:size-6">
        {icon}
      </div>
      <h3 className="mt-5 text-lg font-bold text-fg">{title}</h3>
      <div className="mt-2 max-w-md text-sm leading-relaxed text-fg-muted">{description}</div>
      {action ? <div className="mt-6 flex flex-wrap justify-center gap-3">{action}</div> : null}
    </div>
  )
}

export function ErrorState({
  title = "We Couldn't Load This",
  message,
  onRetry,
  className,
}: {
  title?: string
  message: string
  onRetry?: () => void
  className?: string
}) {
  return (
    <div
      role="alert"
      className={cn("flex flex-col items-center px-6 py-12 text-center", className)}
    >
      <div className="flex size-12 items-center justify-center rounded-2xl bg-danger-soft text-danger">
        <CircleAlert className="size-6" />
      </div>
      <h3 className="mt-4 text-lg font-bold text-fg">{title}</h3>
      <p className="mt-2 max-w-md text-sm text-fg-muted">{message}</p>
      {onRetry ? (
        <Button variant="outline" className="mt-5" onClick={onRetry}>
          Try Again
        </Button>
      ) : null}
    </div>
  )
}

export function InlineAlert({
  tone = "info",
  title,
  children,
  className,
  icon,
}: {
  tone?: "info" | "warning" | "danger" | "success"
  title?: string
  children: React.ReactNode
  className?: string
  icon?: React.ReactNode
}) {
  const tones = {
    info: "border-primary/25 bg-primary-soft text-primary-soft-fg",
    warning: "border-warning/30 bg-warning-soft text-warning",
    danger: "border-danger/30 bg-danger-soft text-danger",
    success: "border-success/30 bg-success-soft text-success",
  }
  return (
    <div className={cn("flex gap-3 rounded-xl border px-4 py-3 text-sm", tones[tone], className)}>
      {icon ? <span className="mt-0.5 shrink-0 [&>svg]:size-4">{icon}</span> : null}
      <div className="min-w-0">
        {title ? <p className="font-semibold">{title}</p> : null}
        <div className={cn("text-fg-muted", title && "mt-0.5")}>{children}</div>
      </div>
    </div>
  )
}
