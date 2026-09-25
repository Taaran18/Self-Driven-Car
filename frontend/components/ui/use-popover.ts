"use client"

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react"

interface Position {
  top: number
  left: number
  width: number
  maxHeight: number
  placement: "bottom" | "top"
}

export function usePopover<T extends HTMLElement>({
  matchWidth = true,
  align = "start",
  minWidth = 180,
}: { matchWidth?: boolean; align?: "start" | "end"; minWidth?: number } = {}) {
  const [open, setOpen] = useState(false)
  const [position, setPosition] = useState<Position | null>(null)
  const triggerRef = useRef<T | null>(null)
  const panelRef = useRef<HTMLDivElement | null>(null)

  const update = useCallback(() => {
    const trigger = triggerRef.current
    if (!trigger) return
    const rect = trigger.getBoundingClientRect()
    const gap = 6
    const viewportH = window.innerHeight
    const viewportW = window.innerWidth
    const spaceBelow = viewportH - rect.bottom - gap - 8
    const spaceAbove = rect.top - gap - 8
    const panelHeight = panelRef.current?.scrollHeight ?? 280
    const placement =
      spaceBelow < Math.min(panelHeight, 240) && spaceAbove > spaceBelow ? "top" : "bottom"
    const width = matchWidth
      ? Math.max(rect.width, minWidth)
      : Math.max(minWidth, panelRef.current?.offsetWidth ?? minWidth)
    let left = align === "end" ? rect.right - width : rect.left
    left = Math.min(Math.max(8, left), viewportW - width - 8)
    setPosition({
      top: placement === "bottom" ? rect.bottom + gap : rect.top - gap,
      left,
      width,
      maxHeight: Math.max(160, Math.min(360, placement === "bottom" ? spaceBelow : spaceAbove)),
      placement,
    })
  }, [align, matchWidth, minWidth])

  useLayoutEffect(() => {
    if (!open) return
    update()
    const frame = requestAnimationFrame(update)
    return () => cancelAnimationFrame(frame)
  }, [open, update])

  useEffect(() => {
    if (!open) return
    const onPointer = (event: PointerEvent) => {
      const target = event.target as Node
      if (triggerRef.current?.contains(target) || panelRef.current?.contains(target)) return
      setOpen(false)
    }
    const onScroll = (event: Event) => {
      if (panelRef.current?.contains(event.target as Node)) return
      update()
    }
    document.addEventListener("pointerdown", onPointer)
    window.addEventListener("resize", update)
    window.addEventListener("scroll", onScroll, true)
    return () => {
      document.removeEventListener("pointerdown", onPointer)
      window.removeEventListener("resize", update)
      window.removeEventListener("scroll", onScroll, true)
    }
  }, [open, update])

  return { open, setOpen, position, triggerRef, panelRef }
}

export function popoverStyle(
  position: {
    top: number
    left: number
    width: number
    maxHeight: number
    placement: string
  } | null,
) {
  if (!position) return { visibility: "hidden" as const, top: 0, left: 0 }
  return {
    top: position.top,
    left: position.left,
    width: position.width,
    maxHeight: position.maxHeight,
    transform: position.placement === "top" ? "translateY(-100%)" : undefined,
  }
}
