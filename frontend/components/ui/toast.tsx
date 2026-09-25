"use client"

import { CircleAlert, CircleCheck, Info, TriangleAlert, X } from "lucide-react"
import { createContext, useCallback, useContext, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/cn"

type Tone = "success" | "error" | "info" | "warning"

interface Toast {
  id: number
  title: string
  description?: string
  tone: Tone
}

interface ToastContextValue {
  toast: (toast: Omit<Toast, "id" | "tone"> & { tone?: Tone; duration?: number }) => void
  success: (title: string, description?: string) => void
  error: (title: string, description?: string) => void
  info: (title: string, description?: string) => void
}

const ToastContext = createContext<ToastContextValue | null>(null)

const toneStyles: Record<Tone, { icon: typeof Info; className: string }> = {
  success: { icon: CircleCheck, className: "text-success" },
  error: { icon: CircleAlert, className: "text-danger" },
  info: { icon: Info, className: "text-primary" },
  warning: { icon: TriangleAlert, className: "text-warning" },
}

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const nextId = useRef(1)
  const timers = useRef(new Map<number, ReturnType<typeof setTimeout>>())

  const dismiss = useCallback((id: number) => {
    setToasts((current) => current.filter((t) => t.id !== id))
    const timer = timers.current.get(id)
    if (timer) clearTimeout(timer)
    timers.current.delete(id)
  }, [])

  const toast = useCallback<ToastContextValue["toast"]>(
    ({ title, description, tone = "info", duration = 5000 }) => {
      const id = nextId.current++
      setToasts((current) => [...current.slice(-3), { id, title, description, tone }])
      timers.current.set(
        id,
        setTimeout(() => dismiss(id), duration),
      )
    },
    [dismiss],
  )

  const value = useMemo<ToastContextValue>(
    () => ({
      toast,
      success: (title, description) => toast({ title, description, tone: "success" }),
      error: (title, description) => toast({ title, description, tone: "error", duration: 7000 }),
      info: (title, description) => toast({ title, description, tone: "info" }),
    }),
    [toast],
  )

  return (
    <ToastContext.Provider value={value}>
      {children}
      <div
        aria-live="polite"
        aria-relevant="additions"
        className="pointer-events-none fixed inset-x-0 bottom-0 z-[80] flex flex-col items-center gap-2 p-4 sm:items-end sm:p-6"
      >
        {toasts.map((t) => {
          const { icon: Icon, className } = toneStyles[t.tone]
          return (
            <div
              key={t.id}
              role={t.tone === "error" ? "alert" : "status"}
              className="pointer-events-auto flex w-full max-w-sm animate-toast-in items-start gap-3 rounded-xl border border-border bg-surface p-4 shadow-[var(--shadow-pop)]"
            >
              <Icon aria-hidden className={cn("mt-0.5 size-5 shrink-0", className)} />
              <div className="min-w-0 flex-1">
                <p className="text-sm font-semibold text-fg">{t.title}</p>
                {t.description ? (
                  <p className="mt-1 text-sm text-fg-muted">{t.description}</p>
                ) : null}
              </div>
              <button
                type="button"
                onClick={() => dismiss(t.id)}
                className="-m-1 rounded-md p-1 text-fg-subtle transition hover:bg-surface-3 hover:text-fg"
                aria-label="Dismiss notification"
              >
                <X className="size-4" />
              </button>
            </div>
          )
        })}
      </div>
    </ToastContext.Provider>
  )
}

export function useToast() {
  const context = useContext(ToastContext)
  if (!context) throw new Error("useToast must be used inside ToastProvider")
  return context
}
