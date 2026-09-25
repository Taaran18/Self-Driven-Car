import Link from "next/link"
import { cn } from "@/lib/cn"
import { site } from "@/lib/site"

export function LogoMark({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 32 32" aria-hidden className={cn("size-8 shrink-0", className)}>
      <rect width="32" height="32" rx="9" fill="var(--primary)" />
      <path
        d="M9.5 10.5a9 9 0 0 1 13 0M12 13.2a5.4 5.4 0 0 1 8 0"
        fill="none"
        stroke="var(--primary-fg)"
        strokeWidth="1.8"
        strokeLinecap="round"
        opacity="0.75"
      />
      <path
        d="M16 15.2c1.9 0 3.2 1.4 3.2 3.4v4.6a1.6 1.6 0 0 1-1.6 1.6h-3.2a1.6 1.6 0 0 1-1.6-1.6v-4.6c0-2 1.3-3.4 3.2-3.4Z"
        fill="var(--primary-fg)"
      />
    </svg>
  )
}

export function Logo({ className, compact }: { className?: string; compact?: boolean }) {
  return (
    <Link
      href="/"
      className={cn(
        "inline-flex items-center gap-2.5 rounded-lg font-display font-bold text-fg",
        className,
      )}
      aria-label={`${site.name} home`}
    >
      <LogoMark />
      {compact ? null : <span className="text-lg tracking-tight">{site.name}</span>}
    </Link>
  )
}
