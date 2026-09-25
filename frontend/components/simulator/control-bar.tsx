"use client"

import { Gauge, Pause, Play, SkipForward, Square, StepForward } from "lucide-react"
import { useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { ConfirmDialog } from "@/components/ui/dialog"
import { Select } from "@/components/ui/select"
import type { Phase } from "@/lib/simulation/controller"
import type { Speed } from "@/lib/types"
import { SPEED_OPTIONS } from "./options"

interface ControlBarProps {
  phase: Phase
  speed: Speed
  confirmStop: boolean
  generationsDone: number
  onStart: () => void
  starting: boolean
  onPause: () => void
  onResume: () => void
  onStep: () => void
  onSkip: () => void
  onStop: () => void
  onSpeed: (speed: Speed) => void
}

export function ControlBar({
  phase,
  speed,
  confirmStop,
  generationsDone,
  onStart,
  starting,
  onPause,
  onResume,
  onStep,
  onSkip,
  onStop,
  onSpeed,
}: ControlBarProps) {
  const [confirming, setConfirming] = useState(false)
  const active = phase === "running" || phase === "paused"

  useEffect(() => {
    if (!active) return
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement
      if (
        target.closest("input, textarea, [role=combobox], [role=listbox], [role=dialog], button, a")
      )
        return
      if (event.key === " ") {
        event.preventDefault()
        if (phase === "running") onPause()
        else onResume()
      } else if (event.key === ".") {
        event.preventDefault()
        onStep()
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [active, phase, onPause, onResume, onStep])

  function requestStop() {
    if (confirmStop) setConfirming(true)
    else onStop()
  }

  return (
    <div className="flex flex-wrap items-center justify-between gap-3 border-t border-border bg-surface-2 px-3 py-3 sm:px-4">
      <div className="flex flex-wrap items-center gap-2">
        {phase === "running" ? (
          <Button onClick={onPause} aria-keyshortcuts="Space">
            <Pause className="size-4" /> Pause
          </Button>
        ) : phase === "paused" ? (
          <Button onClick={onResume} aria-keyshortcuts="Space">
            <Play className="size-4" /> Resume
          </Button>
        ) : (
          <Button onClick={onStart} loading={starting} loadingText="Starting">
            <Play className="size-4" /> Start Training
          </Button>
        )}
        <Button
          variant="outline"
          onClick={onStep}
          disabled={!active}
          aria-keyshortcuts="."
          title="Run exactly one tick, then pause"
        >
          <StepForward className="size-4" /> Step
        </Button>
        <Button
          variant="outline"
          onClick={onSkip}
          disabled={!active}
          title="End this generation now and breed the next one"
        >
          <SkipForward className="size-4" />
          <span className="hidden sm:inline">Skip Generation</span>
          <span className="sm:hidden">Skip</span>
        </Button>
        <Button variant="danger-outline" onClick={requestStop} disabled={!active}>
          <Square className="size-4" /> Stop
        </Button>
      </div>
      <div className="flex w-full items-center gap-2 sm:w-56">
        <Select
          value={speed}
          onChange={onSpeed}
          options={SPEED_OPTIONS}
          disabled={!active}
          label="Simulation speed"
          prefix={<Gauge className="size-4" />}
        />
      </div>
      <p className="hidden w-full text-xs text-fg-subtle lg:block">
        Shortcuts:{" "}
        <kbd className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono">Space</kbd>{" "}
        pause or resume ·{" "}
        <kbd className="rounded border border-border bg-surface px-1.5 py-0.5 font-mono">.</kbd>{" "}
        step one tick
      </p>
      <ConfirmDialog
        open={confirming}
        onClose={() => setConfirming(false)}
        onConfirm={() => {
          onStop()
          setConfirming(false)
        }}
        title="Stop This Run?"
        description={
          <>
            Training ends now. The {generationsDone} generation{generationsDone === 1 ? "" : "s"}{" "}
            finished so far stay in your run history, and this run still counts toward today&apos;s
            limit.
          </>
        }
        confirmLabel="Stop Run"
        cancelLabel="Keep Training"
      />
    </div>
  )
}
