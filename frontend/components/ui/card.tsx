import { cn } from "@/lib/cn"

export function Card({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      className={cn(
        "rounded-2xl border border-border bg-surface shadow-[var(--shadow-card)]",
        className,
      )}
      {...props}
    />
  )
}

interface CardHeaderProps {
  title: string
  description?: React.ReactNode
  icon?: React.ReactNode
  action?: React.ReactNode
  className?: string
  as?: "h2" | "h3"
}

export function CardHeader({
  title,
  description,
  icon,
  action,
  className,
  as: Heading = "h2",
}: CardHeaderProps) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-start justify-between gap-3 border-b border-border px-5 py-4",
        className,
      )}
    >
      <div className="flex min-w-0 items-start gap-3">
        {icon ? (
          <div className="flex size-9 shrink-0 items-center justify-center rounded-lg bg-primary-soft text-primary-soft-fg [&>svg]:size-[18px]">
            {icon}
          </div>
        ) : null}
        <div className="min-w-0">
          <Heading className="text-base font-bold text-fg">{title}</Heading>
          {description ? <div className="mt-0.5 text-sm text-fg-muted">{description}</div> : null}
        </div>
      </div>
      {action ? <div className="flex shrink-0 items-center gap-2">{action}</div> : null}
    </div>
  )
}

interface StatCardProps {
  label: string
  value: React.ReactNode
  hint?: React.ReactNode
  icon?: React.ReactNode
  tone?: "primary" | "success" | "warning" | "violet" | "danger"
  className?: string
}

const toneClasses = {
  primary: "bg-primary-soft text-primary-soft-fg",
  success: "bg-success-soft text-success",
  warning: "bg-warning-soft text-warning",
  violet: "bg-violet-soft text-violet",
  danger: "bg-danger-soft text-danger",
}

export function StatCard({ label, value, hint, icon, tone = "primary", className }: StatCardProps) {
  return (
    <Card className={cn("relative overflow-hidden p-4 sm:p-5", className)}>
      <div className="flex items-start justify-between gap-3">
        <p className="text-xs font-semibold tracking-wider text-fg-subtle uppercase">{label}</p>
        {icon ? (
          <span
            className={cn(
              "flex size-8 items-center justify-center rounded-lg [&>svg]:size-4",
              toneClasses[tone],
            )}
          >
            {icon}
          </span>
        ) : null}
      </div>
      <p className="tabular mt-2 font-display text-2xl font-bold text-fg sm:text-3xl">{value}</p>
      {hint ? <p className="mt-1 truncate text-xs text-fg-muted">{hint}</p> : null}
    </Card>
  )
}
