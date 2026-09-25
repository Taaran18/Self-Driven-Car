"use client"

import { Check, Copy, FileCode2 } from "lucide-react"
import { useEffect, useMemo, useRef, useState } from "react"
import { cn } from "@/lib/cn"
import { tokenizeLine } from "@/lib/highlight"

interface CodeBlockProps {
  code: string
  file?: string
  startLine?: number
  highlight?: number[]
  annotations?: Record<number, React.ReactNode>
  className?: string
  maxHeight?: number
  scrollToHighlight?: boolean
  label?: string
}

export function CodeBlock({
  code,
  file,
  startLine = 1,
  highlight,
  annotations,
  className,
  maxHeight,
  scrollToHighlight,
  label,
}: CodeBlockProps) {
  const [copied, setCopied] = useState(false)
  const lines = useMemo(() => code.split("\n").map(tokenizeLine), [code])
  const scrollRef = useRef<HTMLDivElement>(null)
  const firstHighlight = highlight && highlight.length ? Math.min(...highlight) : null

  useEffect(() => {
    if (!scrollToHighlight || firstHighlight === null) return
    const container = scrollRef.current
    const row = container?.querySelector<HTMLElement>(`[data-line="${firstHighlight}"]`)
    if (!container || !row) return
    const top = row.offsetTop - container.clientHeight / 3
    container.scrollTo({ top: Math.max(0, top), behavior: "smooth" })
  }, [firstHighlight, scrollToHighlight])

  async function copy() {
    try {
      await navigator.clipboard.writeText(code)
      setCopied(true)
      setTimeout(() => setCopied(false), 1600)
    } catch {}
  }

  return (
    <figure
      className={cn("overflow-hidden rounded-2xl border border-border bg-code-bg", className)}
    >
      <figcaption className="flex items-center justify-between gap-3 border-b border-border bg-surface-2 px-4 py-2.5">
        <span className="flex min-w-0 items-center gap-2 font-mono text-xs text-fg-muted">
          <FileCode2 aria-hidden className="size-4 shrink-0 text-primary" />
          <span className="truncate">
            {file ?? label ?? "python"}
            {file ? <span className="text-fg-subtle"> · line {startLine}</span> : null}
          </span>
        </span>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs font-semibold text-fg-muted transition hover:bg-surface-3 hover:text-fg"
          aria-label="Copy code"
        >
          {copied ? <Check className="size-3.5 text-success" /> : <Copy className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </figcaption>
      <div
        ref={scrollRef}
        className="relative overflow-auto"
        style={maxHeight ? { maxHeight } : undefined}
      >
        <pre className="min-w-max py-3 font-mono text-[13px] leading-6">
          <code>
            {lines.map((tokens, i) => {
              const lineNo = startLine + i
              const active = highlight?.includes(i)
              return (
                <div
                  key={i}
                  data-line={i}
                  className={cn(
                    "flex pr-4 transition-colors duration-200",
                    active && "bg-code-line shadow-[inset_3px_0_0_var(--primary)]",
                  )}
                >
                  <span
                    aria-hidden
                    className={cn(
                      "sticky left-0 w-12 shrink-0 bg-code-bg pr-4 text-right text-fg-subtle/70 select-none",
                      active && "bg-transparent text-primary",
                    )}
                  >
                    {lineNo}
                  </span>
                  <span className="whitespace-pre">
                    {tokens.map((token, j) => (
                      <span key={j} className={`code-token-${token.kind}`}>
                        {token.text}
                      </span>
                    ))}
                    {tokens.length === 0 ? " " : null}
                  </span>
                  {annotations?.[i] ? (
                    <span className="ml-4 inline-flex items-center rounded-md bg-primary-soft px-1.5 text-xs font-semibold whitespace-nowrap text-primary-soft-fg">
                      {annotations[i]}
                    </span>
                  ) : null}
                </div>
              )
            })}
          </code>
        </pre>
      </div>
    </figure>
  )
}
