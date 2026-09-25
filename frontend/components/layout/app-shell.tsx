"use client"

import {
  BookOpen,
  CarFront,
  History,
  House,
  LayoutDashboard,
  Menu as MenuIcon,
  Pin,
  PinOff,
  Settings,
  X,
} from "lucide-react"
import Link from "next/link"
import { usePathname } from "next/navigation"
import { useCallback, useEffect, useRef, useState, useSyncExternalStore } from "react"
import { cn } from "@/lib/cn"
import { site } from "@/lib/site"
import { useUsage } from "@/lib/usage"
import { LogoMark } from "./logo"
import { SiteFooter } from "./site-footer"
import { ThemeToggle } from "./theme-toggle"

const PIN_KEY = "sdc-sidebar-pinned"
const COLLAPSED = 76
const EXPANDED = 264

const workspace = [
  { href: "/simulator", label: "Simulator", icon: CarFront },
  { href: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { href: "/runs", label: "Training Runs", icon: History },
  { href: "/settings", label: "Settings", icon: Settings },
]

const learn = [
  { href: "/how-it-works", label: "How It Works", icon: BookOpen },
  { href: "/", label: "Home", icon: House },
]

const pinListeners = new Set<() => void>()

function readPinned() {
  try {
    return localStorage.getItem(PIN_KEY) === "true"
  } catch {
    return false
  }
}

function subscribePinned(callback: () => void) {
  pinListeners.add(callback)
  return () => pinListeners.delete(callback)
}

function writePinned(value: boolean) {
  try {
    localStorage.setItem(PIN_KEY, String(value))
  } catch {}
  pinListeners.forEach((l) => l())
}

function UsageRing({ expanded, onNavigate }: { expanded: boolean; onNavigate?: () => void }) {
  const usage = useUsage()
  const ratio = usage.runs_per_day ? usage.left_today / usage.runs_per_day : 0
  const radius = 15
  const circumference = 2 * Math.PI * radius
  const low = ratio <= 0.2
  return (
    <Link
      href="/settings?tab=usage"
      onClick={onNavigate}
      className="flex items-center gap-3 rounded-xl bg-surface-2 p-2 transition-colors hover:bg-surface-3"
      aria-label={
        usage.live
          ? `Free trial: ${usage.left_today} of ${usage.runs_per_day} runs left today`
          : `Free trial: ${usage.runs_per_day} runs a day`
      }
    >
      <span className="relative flex size-9 shrink-0 items-center justify-center">
        <svg viewBox="0 0 36 36" className="size-9 -rotate-90" aria-hidden>
          <circle
            cx="18"
            cy="18"
            r={radius}
            fill="none"
            strokeWidth="3.5"
            className="stroke-border"
          />
          <circle
            cx="18"
            cy="18"
            r={radius}
            fill="none"
            strokeWidth="3.5"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={circumference * (1 - ratio)}
            className={cn(
              "transition-[stroke-dashoffset] duration-500",
              low ? "stroke-danger" : "stroke-primary",
            )}
          />
        </svg>
        <span className="tabular absolute text-[11px] font-bold text-fg">{usage.left_today}</span>
      </span>
      <span
        className={cn(
          "min-w-0 transition-opacity duration-150",
          expanded ? "opacity-100" : "opacity-0",
        )}
      >
        <span className="block truncate text-sm font-semibold text-fg">Free Trial</span>
        <span className="block truncate text-xs text-fg-subtle">
          {usage.live
            ? `${usage.left_today} of ${usage.runs_per_day} runs left today`
            : `${usage.runs_per_day} runs a day`}
        </span>
      </span>
    </Link>
  )
}

function NavSection({
  title,
  items,
  expanded,
  onNavigate,
}: {
  title: string
  items: typeof workspace
  expanded: boolean
  onNavigate?: () => void
}) {
  const pathname = usePathname()
  return (
    <div>
      <p
        className={cn(
          "mb-2 h-4 px-3 text-[11px] font-bold tracking-widest text-fg-subtle uppercase transition-opacity duration-150",
          expanded ? "opacity-100" : "opacity-0",
        )}
        aria-hidden={!expanded}
      >
        {title}
      </p>
      <ul className="space-y-1">
        {items.map((item) => {
          const active =
            item.href === "/"
              ? pathname === "/"
              : pathname === item.href || pathname.startsWith(`${item.href}/`)
          const Icon = item.icon
          return (
            <li key={item.href}>
              <Link
                href={item.href}
                onClick={onNavigate}
                aria-current={active ? "page" : undefined}
                aria-label={expanded ? undefined : item.label}
                className={cn(
                  "group relative flex h-11 items-center gap-3 rounded-xl px-3 text-sm font-semibold transition-colors",
                  active
                    ? "bg-primary-soft text-primary-soft-fg"
                    : "text-fg-muted hover:bg-surface-3 hover:text-fg",
                )}
              >
                {active ? (
                  <span
                    aria-hidden
                    className="absolute top-2.5 bottom-2.5 -left-3 w-1 rounded-r-full bg-primary"
                  />
                ) : null}
                <Icon aria-hidden className="size-5 shrink-0" />
                <span
                  className={cn(
                    "flex min-w-0 flex-1 items-center justify-between gap-2 truncate transition-opacity duration-150",
                    expanded ? "opacity-100" : "pointer-events-none opacity-0",
                  )}
                >
                  <span className="truncate">{item.label}</span>
                </span>
              </Link>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function SidebarBody({
  expanded,
  pinned,
  onTogglePin,
  onNavigate,
  showPin,
  onClose,
}: {
  expanded: boolean
  pinned?: boolean
  onTogglePin?: () => void
  onNavigate?: () => void
  showPin?: boolean
  onClose?: () => void
}) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex h-16 shrink-0 items-center gap-2 border-b border-border px-[22px]">
        <Link
          href="/"
          className="flex min-w-0 items-center gap-2.5 rounded-lg"
          onClick={onNavigate}
          aria-label={`${site.name} home`}
        >
          <LogoMark />
          <span
            className={cn(
              "truncate font-display text-base font-bold text-fg transition-opacity duration-150",
              expanded ? "opacity-100" : "opacity-0",
            )}
          >
            {site.name}
          </span>
        </Link>
        {showPin ? (
          <button
            type="button"
            onClick={onTogglePin}
            aria-pressed={pinned}
            aria-label={pinned ? "Unpin sidebar" : "Pin sidebar open"}
            title={pinned ? "Unpin sidebar" : "Pin sidebar open"}
            tabIndex={expanded ? 0 : -1}
            className={cn(
              "ml-auto flex size-8 shrink-0 items-center justify-center rounded-lg transition-[opacity,background-color,color] duration-150",
              pinned
                ? "bg-primary-soft text-primary-soft-fg"
                : "text-fg-subtle hover:bg-surface-3 hover:text-fg",
              expanded ? "opacity-100" : "pointer-events-none opacity-0",
            )}
          >
            {pinned ? <PinOff className="size-4" /> : <Pin className="size-4" />}
          </button>
        ) : null}
        {onClose ? (
          <button
            type="button"
            onClick={onClose}
            aria-label="Close menu"
            className="ml-auto flex size-9 items-center justify-center rounded-lg text-fg-muted hover:bg-surface-3 hover:text-fg"
          >
            <X className="size-5" />
          </button>
        ) : null}
      </div>
      <nav
        aria-label="App"
        className="flex-1 space-y-6 overflow-x-hidden overflow-y-auto px-3 py-5"
      >
        <NavSection
          title="Workspace"
          items={workspace}
          expanded={expanded}
          onNavigate={onNavigate}
        />
        <NavSection title="Learn" items={learn} expanded={expanded} onNavigate={onNavigate} />
      </nav>
      <div className="shrink-0 space-y-2 border-t border-border p-3">
        <ThemeToggle withLabel className={cn(!expanded && "[&>span:last-child]:opacity-0")} />
        <UsageRing expanded={expanded} onNavigate={onNavigate} />
      </div>
    </div>
  )
}

export function AppShell({ children }: { children: React.ReactNode }) {
  const pinned = useSyncExternalStore(subscribePinned, readPinned, () => false)
  const [hovered, setHovered] = useState(false)
  const [drawerOpen, setDrawerOpen] = useState(false)
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pathname = usePathname()
  const [lastPath, setLastPath] = useState(pathname)
  const expanded = pinned || hovered

  if (pathname !== lastPath) {
    setLastPath(pathname)
    setDrawerOpen(false)
  }

  const schedule = useCallback((value: boolean, delay: number) => {
    if (timer.current) clearTimeout(timer.current)
    timer.current = setTimeout(() => setHovered(value), delay)
  }, [])

  useEffect(() => () => void (timer.current && clearTimeout(timer.current)), [])

  useEffect(() => {
    if (!drawerOpen) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawerOpen(false)
    document.addEventListener("keydown", onKey)
    const { overflow } = document.body.style
    document.body.style.overflow = "hidden"
    return () => {
      document.removeEventListener("keydown", onKey)
      document.body.style.overflow = overflow
    }
  }, [drawerOpen])

  return (
    <div
      className="min-h-dvh bg-bg"
      style={{ ["--sidebar-width" as string]: `${expanded ? EXPANDED : COLLAPSED}px` }}
    >
      <aside
        aria-label="Sidebar"
        onMouseEnter={() => schedule(true, 90)}
        onMouseLeave={() => schedule(false, 180)}
        onFocus={() => schedule(true, 0)}
        onBlur={(e) => {
          if (!e.currentTarget.contains(e.relatedTarget as Node)) schedule(false, 0)
        }}
        className="fixed inset-y-0 left-0 z-40 hidden border-r border-border bg-surface transition-[width,box-shadow] duration-200 ease-out lg:block"
        style={{ width: expanded ? EXPANDED : COLLAPSED }}
      >
        <SidebarBody
          expanded={expanded}
          pinned={pinned}
          onTogglePin={() => writePinned(!pinned)}
          showPin
        />
      </aside>

      <div className="sticky top-0 z-30 flex h-16 items-center justify-between gap-3 border-b border-border bg-bg/85 px-4 backdrop-blur-xl sm:px-6 lg:hidden">
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={() => setDrawerOpen(true)}
            aria-label="Open menu"
            aria-expanded={drawerOpen}
            aria-controls="app-drawer"
            className="flex size-10 items-center justify-center rounded-xl text-fg-muted hover:bg-surface-3 hover:text-fg"
          >
            <MenuIcon className="size-5" />
          </button>
          <Link
            href="/"
            className="flex items-center gap-2 font-display font-bold"
            aria-label={`${site.name} home`}
          >
            <LogoMark className="size-7" />
            <span className="hidden sm:inline">{site.name}</span>
          </Link>
        </div>
        <ThemeToggle />
      </div>

      {drawerOpen ? (
        <div
          className="fixed inset-0 z-50 lg:hidden"
          role="dialog"
          aria-modal="true"
          aria-label="Menu"
          id="app-drawer"
        >
          <div
            className="absolute inset-0 animate-fade-in bg-overlay"
            onClick={() => setDrawerOpen(false)}
            aria-hidden
          />
          <div className="absolute inset-y-0 left-0 w-[min(84vw,300px)] animate-drawer-in border-r border-border bg-surface shadow-[var(--shadow-pop)]">
            <SidebarBody
              expanded
              onNavigate={() => setDrawerOpen(false)}
              onClose={() => setDrawerOpen(false)}
            />
          </div>
        </div>
      ) : null}

      <div className="flex min-h-dvh flex-col transition-[padding] duration-200 ease-out lg:pl-[var(--sidebar-width)]">
        <main id="main" className="flex-1">
          {children}
        </main>
        <SiteFooter compact className={pathname === "/simulator" ? "lg:hidden" : undefined} />
      </div>
    </div>
  )
}
