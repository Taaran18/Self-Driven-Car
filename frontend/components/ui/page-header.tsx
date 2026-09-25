import { cn } from "@/lib/cn"

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: React.ReactNode
  actions?: React.ReactNode
  className?: string
  size?: "md" | "lg"
}

export function PageHeader({
  eyebrow,
  title,
  description,
  actions,
  className,
  size = "md",
}: PageHeaderProps) {
  return (
    <header className={cn("mx-auto flex max-w-3xl flex-col items-center text-center", className)}>
      {eyebrow ? (
        <p className="mb-3 inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary-soft px-3 py-1 text-xs font-semibold tracking-wider text-primary-soft-fg uppercase">
          {eyebrow}
        </p>
      ) : null}
      <h1
        className={cn(
          "font-bold tracking-tight text-fg",
          size === "lg" ? "text-4xl sm:text-5xl lg:text-6xl" : "text-3xl sm:text-4xl lg:text-5xl",
        )}
      >
        {title}
      </h1>
      {description ? (
        <div className="mt-4 max-w-2xl text-base leading-relaxed text-fg-muted sm:text-lg">
          {description}
        </div>
      ) : null}
      {actions ? (
        <div className="mt-6 flex flex-wrap items-center justify-center gap-3">{actions}</div>
      ) : null}
    </header>
  )
}

export function SectionHeading({
  eyebrow,
  title,
  description,
  className,
}: {
  eyebrow?: string
  title: string
  description?: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn("mx-auto max-w-3xl text-center", className)}>
      {eyebrow ? (
        <p className="text-sm font-semibold tracking-wider text-primary uppercase">{eyebrow}</p>
      ) : null}
      <h2 className="mt-2 text-3xl font-bold tracking-tight text-fg sm:text-4xl lg:text-5xl">
        {title}
      </h2>
      {description ? (
        <div className="mt-4 text-base leading-relaxed text-fg-muted sm:text-lg">{description}</div>
      ) : null}
    </div>
  )
}
