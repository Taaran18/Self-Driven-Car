"use client"

import { useEffect, useState, useSyncExternalStore } from "react"
import { SimulationController } from "@/lib/simulation/controller"

export function useSimulation() {
  const [controller] = useState(() => new SimulationController())
  const snapshot = useSyncExternalStore(
    controller.subscribe,
    controller.getSnapshot,
    controller.getSnapshot,
  )

  useEffect(() => {
    controller.attach()
    return () => controller.dispose()
  }, [controller])

  useEffect(() => {
    const active =
      snapshot.phase === "running" || snapshot.phase === "paused" || snapshot.phase === "starting"
    if (!active) return
    const onBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault()
    }
    window.addEventListener("beforeunload", onBeforeUnload)
    return () => window.removeEventListener("beforeunload", onBeforeUnload)
  }, [snapshot.phase])

  return { controller, snapshot }
}
