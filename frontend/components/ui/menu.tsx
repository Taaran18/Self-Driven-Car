"use client"

import { useEffect, useId, useState } from "react"
import { createPortal } from "react-dom"
import { cn } from "@/lib/cn"
import { popoverStyle, usePopover } from "./use-popover"

export interface MenuItem {
  label: string
  icon?: React.ReactNode
  onSelect: () => void
  tone?: "default" | "danger"
  disabled?: boolean
}

interface MenuProps {
  items: MenuItem[]
  label: string
  children: React.ReactNode
  triggerClassName?: string
  align?: "start" | "end"
}

export function Menu({ items, label, children, triggerClassName, align = "end" }: MenuProps) {
  const id = useId()
  const { open, setOpen, position, triggerRef, panelRef } = usePopover<HTMLButtonElement>({
    matchWidth: false,
    align,
    minWidth: 200,
  })
  const [active, setActive] = useState(0)

  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => {
      const el = panelRef.current?.querySelector<HTMLElement>(`[data-index="${active}"]`)
      el?.focus()
    })
    return () => cancelAnimationFrame(frame)
  }, [open, active, panelRef])

  function close(focus = true) {
    setOpen(false)
    if (focus) triggerRef.current?.focus()
  }

  function onKeyDown(event: React.KeyboardEvent) {
    if (event.key === "ArrowDown") {
      event.preventDefault()
      setActive((a) => (a + 1) % items.length)
    } else if (event.key === "ArrowUp") {
      event.preventDefault()
      setActive((a) => (a - 1 + items.length) % items.length)
    } else if (event.key === "Escape") {
      event.preventDefault()
      close()
    } else if (event.key === "Tab") {
      close(false)
    }
  }

  return (
    <>
      <button
        ref={triggerRef}
        type="button"
        className={triggerClassName}
        onClick={() => {
          setActive(0)
          setOpen(!open)
        }}
        onKeyDown={(event) => {
          if (event.key === "ArrowDown" || event.key === "Enter" || event.key === " ") {
            event.preventDefault()
            setActive(0)
            setOpen(true)
          }
        }}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? id : undefined}
        aria-label={label}
      >
        {children}
      </button>
      {open
        ? createPortal(
            <div
              ref={panelRef}
              id={id}
              role="menu"
              aria-label={label}
              onKeyDown={onKeyDown}
              style={popoverStyle(position)}
              className="fixed z-[90] min-w-[200px] animate-fade-in rounded-xl border border-border bg-surface p-1.5 shadow-[var(--shadow-pop)]"
            >
              {items.map((item, index) => (
                <button
                  key={item.label}
                  type="button"
                  role="menuitem"
                  data-index={index}
                  tabIndex={index === active ? 0 : -1}
                  disabled={item.disabled}
                  onPointerMove={() => setActive(index)}
                  onClick={() => {
                    close()
                    item.onSelect()
                  }}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-sm font-medium transition-colors outline-none focus:bg-surface-3 disabled:opacity-50",
                    item.tone === "danger" ? "text-danger focus:bg-danger-soft" : "text-fg",
                  )}
                >
                  {item.icon ? <span className="shrink-0 [&>svg]:size-4">{item.icon}</span> : null}
                  {item.label}
                </button>
              ))}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
