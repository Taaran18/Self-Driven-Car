"use client"

import { TriangleAlert, X } from "lucide-react"
import { useEffect, useId, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { cn } from "@/lib/cn"
import { Button } from "./button"
import { Field, Input } from "./field"

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]), select:not([disabled]), [tabindex]:not([tabindex="-1"])'

interface DialogProps {
  open: boolean
  onClose: () => void
  title: string
  description?: React.ReactNode
  children?: React.ReactNode
  footer?: React.ReactNode
  size?: "sm" | "md" | "lg"
  dismissible?: boolean
  icon?: React.ReactNode
  tone?: "default" | "danger"
}

export function Dialog({
  open,
  onClose,
  title,
  description,
  children,
  footer,
  size = "md",
  dismissible = true,
  icon,
  tone = "default",
}: DialogProps) {
  const titleId = useId()
  const descriptionId = useId()
  const panelRef = useRef<HTMLDivElement>(null)
  const onCloseRef = useRef(onClose)
  const dismissibleRef = useRef(dismissible)

  useEffect(() => {
    onCloseRef.current = onClose
    dismissibleRef.current = dismissible
  })

  useEffect(() => {
    if (!open) return
    const previous = document.activeElement as HTMLElement | null
    const { overflow, paddingRight } = document.body.style
    const scrollbar = window.innerWidth - document.documentElement.clientWidth
    document.body.style.overflow = "hidden"
    if (scrollbar > 0) document.body.style.paddingRight = `${scrollbar}px`

    const frame = requestAnimationFrame(() => {
      const panel = panelRef.current
      const autofocus = panel?.querySelector<HTMLElement>("[data-autofocus]")
      const first = panel?.querySelectorAll<HTMLElement>(FOCUSABLE)[0]
      ;(autofocus ?? first ?? panel)?.focus()
    })

    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && dismissibleRef.current) {
        event.stopPropagation()
        onCloseRef.current()
      }
      if (event.key === "Tab" && panelRef.current) {
        const items = Array.from(panelRef.current.querySelectorAll<HTMLElement>(FOCUSABLE))
        if (!items.length) return
        const first = items[0]
        const last = items[items.length - 1]
        if (event.shiftKey && document.activeElement === first) {
          event.preventDefault()
          last.focus()
        } else if (!event.shiftKey && document.activeElement === last) {
          event.preventDefault()
          first.focus()
        }
      }
    }
    document.addEventListener("keydown", onKeyDown)
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener("keydown", onKeyDown)
      document.body.style.overflow = overflow
      document.body.style.paddingRight = paddingRight
      previous?.focus?.()
    }
  }, [open])

  if (!open) return null

  return createPortal(
    <div className="fixed inset-0 z-[70] flex items-end justify-center p-0 sm:items-center sm:p-6">
      <div
        aria-hidden
        className="absolute inset-0 animate-fade-in bg-overlay backdrop-blur-[2px]"
        onClick={() => dismissible && onClose()}
      />
      <div
        ref={panelRef}
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={description ? descriptionId : undefined}
        tabIndex={-1}
        className={cn(
          "relative flex max-h-[calc(100dvh-2rem)] w-full animate-dialog-in flex-col overflow-hidden rounded-t-2xl border border-border bg-surface shadow-[var(--shadow-pop)] outline-none sm:rounded-2xl",
          size === "sm" && "sm:max-w-md",
          size === "md" && "sm:max-w-lg",
          size === "lg" && "sm:max-w-2xl",
        )}
      >
        <div className="flex items-start gap-4 px-6 pt-6">
          {icon ? (
            <div
              className={cn(
                "flex size-11 shrink-0 items-center justify-center rounded-xl",
                tone === "danger"
                  ? "bg-danger-soft text-danger"
                  : "bg-primary-soft text-primary-soft-fg",
              )}
            >
              {icon}
            </div>
          ) : null}
          <div className="min-w-0 flex-1 pt-0.5">
            <h2 id={titleId} className="text-xl font-bold text-fg">
              {title}
            </h2>
            {description ? (
              <div id={descriptionId} className="mt-1.5 text-sm leading-relaxed text-fg-muted">
                {description}
              </div>
            ) : null}
          </div>
          {dismissible ? (
            <button
              type="button"
              onClick={onClose}
              className="-mt-1 -mr-2 rounded-lg p-2 text-fg-subtle transition hover:bg-surface-3 hover:text-fg"
              aria-label="Close dialog"
            >
              <X className="size-5" />
            </button>
          ) : null}
        </div>
        {children ? <div className="overflow-y-auto px-6 pt-5">{children}</div> : null}
        {footer ? (
          <div className="mt-6 flex flex-col-reverse gap-2 border-t border-border bg-surface-2 px-6 py-4 sm:flex-row sm:justify-end">
            {footer}
          </div>
        ) : (
          <div className="h-6" />
        )}
      </div>
    </div>,
    document.body,
  )
}

interface ConfirmDialogProps {
  open: boolean
  onClose: () => void
  onConfirm: () => Promise<void> | void
  title: string
  description: React.ReactNode
  confirmLabel: string
  cancelLabel?: string
  tone?: "default" | "danger"
  requireText?: string
  icon?: React.ReactNode
  children?: React.ReactNode
  canConfirm?: boolean
  error?: string | null
}

export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  description,
  confirmLabel,
  cancelLabel = "Cancel",
  tone = "danger",
  requireText,
  icon,
  children,
  canConfirm = true,
  error,
}: ConfirmDialogProps) {
  const [typed, setTyped] = useState("")
  const [busy, setBusy] = useState(false)
  const [wasOpen, setWasOpen] = useState(open)

  if (open !== wasOpen) {
    setWasOpen(open)
    if (open) setTyped("")
  }

  const matches = !requireText || typed.trim() === requireText
  const disabled = !matches || !canConfirm

  async function handleConfirm() {
    if (disabled) return
    setBusy(true)
    try {
      await onConfirm()
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog
      open={open}
      onClose={() => !busy && onClose()}
      dismissible={!busy}
      title={title}
      description={description}
      tone={tone}
      icon={icon ?? (tone === "danger" ? <TriangleAlert className="size-5" /> : undefined)}
      footer={
        <>
          <Button variant="outline" onClick={onClose} disabled={busy}>
            {cancelLabel}
          </Button>
          <Button
            variant={tone === "danger" ? "danger" : "primary"}
            onClick={handleConfirm}
            disabled={disabled}
            loading={busy}
          >
            {confirmLabel}
          </Button>
        </>
      }
    >
      {children || requireText || error ? (
        <form
          className="space-y-4"
          onSubmit={(event) => {
            event.preventDefault()
            void handleConfirm()
          }}
        >
          {children}
          {requireText ? (
            <Field label={`Type ${requireText} to confirm`}>
              {({ id, describedBy }) => (
                <Input
                  id={id}
                  aria-describedby={describedBy}
                  value={typed}
                  onChange={(e) => setTyped(e.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                  data-autofocus
                  placeholder={requireText}
                />
              )}
            </Field>
          ) : null}
          {error ? (
            <p
              role="alert"
              className="rounded-lg bg-danger-soft px-3 py-2 text-sm font-medium text-danger"
            >
              {error}
            </p>
          ) : null}
          <button type="submit" hidden aria-hidden tabIndex={-1} />
        </form>
      ) : null}
    </Dialog>
  )
}
