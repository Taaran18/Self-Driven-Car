"use client"

import { useEffect, useRef, useState } from "react"
import type { Phase, SimulationController } from "@/lib/simulation/controller"

const AWAY_MS = 5000

export function useAwayPause(controller: SimulationController, phase: Phase) {
  const [askToContinue, setAskToContinue] = useState(false)
  const phaseRef = useRef(phase)
  const pausedWhileAway = useRef(false)

  useEffect(() => {
    phaseRef.current = phase
  })

  useEffect(() => {
    let timer: ReturnType<typeof setTimeout> | null = null
    const onVisibility = () => {
      if (document.hidden) {
        if (phaseRef.current !== "running") return
        timer = setTimeout(() => {
          if (document.hidden && phaseRef.current === "running") {
            pausedWhileAway.current = true
            controller.pause()
          }
        }, AWAY_MS)
      } else {
        if (timer) clearTimeout(timer)
        timer = null
        if (pausedWhileAway.current) {
          pausedWhileAway.current = false
          if (phaseRef.current === "paused") setAskToContinue(true)
        }
      }
    }
    document.addEventListener("visibilitychange", onVisibility)
    return () => {
      document.removeEventListener("visibilitychange", onVisibility)
      if (timer) clearTimeout(timer)
    }
  }, [controller])

  const open = askToContinue && phase === "paused"
  return { open, close: () => setAskToContinue(false) }
}
