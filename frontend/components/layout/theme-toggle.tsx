"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "@/components/theme-provider"
import { cn } from "@/lib/cn"

export function ThemeToggle({ className, withLabel }: { className?: string; withLabel?: boolean }) {
  const { theme, toggleTheme } = useTheme()
  const next = theme === "dark" ? "light" : "dark"
  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
      className={cn(
        "inline-flex h-10 items-center gap-2.5 rounded-xl text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg",
        withLabel ? "w-full px-3" : "w-10 justify-center",
        className,
      )}
    >
      <span className="relative flex size-5 shrink-0 items-center justify-center">
        <Sun className="absolute size-5 scale-100 rotate-0 transition-all dark:scale-0 dark:-rotate-90" />
        <Moon className="absolute size-5 scale-0 rotate-90 transition-all dark:scale-100 dark:rotate-0" />
      </span>
      {withLabel ? (
        <span className="truncate text-sm font-medium">
          <span className="dark:hidden">Light Mode</span>
          <span className="hidden dark:inline">Dark Mode</span>
        </span>
      ) : null}
    </button>
  )
}
