"use client"

import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/cn"

interface RevealProps extends React.HTMLAttributes<HTMLElement> {
  as?: "div" | "section" | "li" | "article" | "header"
  delay?: number
  from?: "up" | "left" | "right" | "scale"
  once?: boolean
}

export function Reveal({
  as: Tag = "div",
  delay = 0,
  from = "up",
  once = true,
  className,
  style,
  children,
  ...props
}: RevealProps) {
  const ref = useRef<HTMLElement>(null)
  const [visible, setVisible] = useState(false)

  useEffect(() => {
    const node = ref.current
    if (!node) return
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setVisible(true)
          if (once) observer.disconnect()
        } else if (!once) {
          setVisible(false)
        }
      },
      { rootMargin: "0px 0px -10% 0px", threshold: 0.12 },
    )
    observer.observe(node)
    return () => observer.disconnect()
  }, [once])

  return (
    <Tag
      ref={ref as React.Ref<never>}
      data-reveal={from === "up" ? "" : from}
      data-visible={visible}
      className={cn(className)}
      style={{ ...style, ["--reveal-delay" as string]: `${delay}ms` }}
      {...props}
    >
      {children}
    </Tag>
  )
}
