"use client"

import { Check, ChevronDown } from "lucide-react"
import { useEffect, useId, useRef, useState } from "react"
import { createPortal } from "react-dom"
import { cn } from "@/lib/cn"
import { popoverStyle, usePopover } from "./use-popover"

export interface SelectOption<T extends string> {
  value: T
  label: string
  description?: string
  icon?: React.ReactNode
  disabled?: boolean
}

interface SelectProps<T extends string> {
  value: T
  onChange: (value: T) => void
  options: SelectOption<T>[]
  id?: string
  label?: string
  placeholder?: string
  disabled?: boolean
  invalid?: boolean
  size?: "sm" | "md"
  className?: string
  describedBy?: string
  prefix?: React.ReactNode
}

export function Select<T extends string>({
  value,
  onChange,
  options,
  id,
  label,
  placeholder = "Select an option",
  disabled,
  invalid,
  size = "md",
  className,
  describedBy,
  prefix,
}: SelectProps<T>) {
  const generatedId = useId()
  const buttonId = id ?? generatedId
  const listId = `${buttonId}-listbox`
  const { open, setOpen, position, triggerRef, panelRef } = usePopover<HTMLButtonElement>()
  const [active, setActive] = useState(0)
  const typeahead = useRef({ text: "", time: 0 })
  const selectedIndex = options.findIndex((o) => o.value === value)
  const selected = options[selectedIndex]

  useEffect(() => {
    if (!open) return
    const frame = requestAnimationFrame(() => panelRef.current?.focus({ preventScroll: true }))
    return () => cancelAnimationFrame(frame)
  }, [open, panelRef])

  useEffect(() => {
    if (!open) return
    panelRef.current
      ?.querySelector(`[data-index="${active}"]`)
      ?.scrollIntoView({ block: "nearest" })
  }, [active, open, panelRef])

  function openList() {
    if (disabled) return
    setActive(Math.max(0, selectedIndex))
    setOpen(true)
  }

  function close(focusTrigger = true) {
    setOpen(false)
    if (focusTrigger) triggerRef.current?.focus()
  }

  function choose(index: number) {
    const option = options[index]
    if (!option || option.disabled) return
    onChange(option.value)
    close()
  }

  function move(delta: number) {
    if (!options.length) return
    let next = active
    for (let i = 0; i < options.length; i++) {
      next = (next + delta + options.length) % options.length
      if (!options[next].disabled) break
    }
    setActive(next)
  }

  function onTriggerKey(event: React.KeyboardEvent) {
    if (["ArrowDown", "ArrowUp", "Enter", " "].includes(event.key)) {
      event.preventDefault()
      openList()
    }
  }

  function onListKey(event: React.KeyboardEvent) {
    switch (event.key) {
      case "ArrowDown":
        event.preventDefault()
        move(1)
        break
      case "ArrowUp":
        event.preventDefault()
        move(-1)
        break
      case "Home":
        event.preventDefault()
        setActive(0)
        break
      case "End":
        event.preventDefault()
        setActive(options.length - 1)
        break
      case "Enter":
      case " ":
        event.preventDefault()
        choose(active)
        break
      case "Escape":
        event.preventDefault()
        close()
        break
      case "Tab":
        close(false)
        break
      default:
        if (event.key.length === 1) {
          const now = event.timeStamp
          const buffer =
            now - typeahead.current.time < 600 ? typeahead.current.text + event.key : event.key
          typeahead.current = { text: buffer.toLowerCase(), time: now }
          const match = options.findIndex((o) =>
            o.label.toLowerCase().startsWith(typeahead.current.text),
          )
          if (match >= 0) setActive(match)
        }
    }
  }

  return (
    <>
      <button
        ref={triggerRef}
        id={buttonId}
        type="button"
        role="combobox"
        aria-haspopup="listbox"
        aria-expanded={open}
        aria-controls={open ? listId : undefined}
        aria-label={label}
        aria-invalid={invalid || undefined}
        aria-describedby={describedBy}
        disabled={disabled}
        onClick={() => (open ? close() : openList())}
        onKeyDown={onTriggerKey}
        className={cn(
          "group flex w-full items-center gap-2 rounded-xl border border-border bg-surface text-left text-sm text-fg transition-[border-color,box-shadow] outline-none hover:border-border-strong focus-visible:border-primary focus-visible:ring-4 focus-visible:ring-primary/15 disabled:cursor-not-allowed disabled:opacity-60 aria-[invalid=true]:border-danger",
          size === "sm" ? "h-9 px-3" : "h-11 px-3.5",
          open && "border-primary ring-4 ring-primary/15",
          className,
        )}
      >
        {prefix ? <span className="shrink-0 text-fg-subtle">{prefix}</span> : null}
        {selected?.icon ? <span className="shrink-0">{selected.icon}</span> : null}
        <span className={cn("min-w-0 flex-1 truncate", !selected && "text-fg-subtle")}>
          {selected?.label ?? placeholder}
        </span>
        <ChevronDown
          aria-hidden
          className={cn(
            "size-4 shrink-0 text-fg-subtle transition-transform",
            open && "rotate-180",
          )}
        />
      </button>
      {open
        ? createPortal(
            <div
              ref={panelRef}
              id={listId}
              role="listbox"
              tabIndex={-1}
              aria-labelledby={buttonId}
              aria-activedescendant={`${listId}-${active}`}
              onKeyDown={onListKey}
              style={popoverStyle(position)}
              className="fixed z-[90] animate-fade-in overflow-y-auto overscroll-contain rounded-xl border border-border bg-surface p-1.5 shadow-[var(--shadow-pop)] outline-none"
            >
              {options.map((option, index) => {
                const isSelected = option.value === value
                return (
                  <div
                    key={option.value}
                    id={`${listId}-${index}`}
                    data-index={index}
                    role="option"
                    aria-selected={isSelected}
                    aria-disabled={option.disabled || undefined}
                    onPointerMove={() => setActive(index)}
                    onClick={() => choose(index)}
                    className={cn(
                      "flex items-start gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-colors",
                      index === active && "bg-surface-3",
                      isSelected ? "text-fg" : "text-fg-muted",
                      option.disabled && "cursor-not-allowed opacity-50",
                    )}
                  >
                    {option.icon ? <span className="mt-0.5 shrink-0">{option.icon}</span> : null}
                    <span className="min-w-0 flex-1">
                      <span className={cn("block", isSelected && "font-semibold")}>
                        {option.label}
                      </span>
                      {option.description ? (
                        <span className="mt-0.5 block text-xs text-fg-subtle">
                          {option.description}
                        </span>
                      ) : null}
                    </span>
                    <Check
                      aria-hidden
                      className={cn(
                        "mt-0.5 size-4 shrink-0 text-primary",
                        isSelected ? "opacity-100" : "opacity-0",
                      )}
                    />
                  </div>
                )
              })}
            </div>,
            document.body,
          )
        : null}
    </>
  )
}
