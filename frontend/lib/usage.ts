"use client"

import { api } from "@/lib/api"
import { createStoredState } from "@/lib/store"

export interface Usage {
  runs_per_day: number
  runs_per_week: number
  used_today: number
  used_this_week: number
  left_today: number
  left_this_week: number
  day_resets_at: string
  week_resets_at: string
  live?: boolean
}

export const STATIC_USAGE: Usage = {
  runs_per_day: 5,
  runs_per_week: 20,
  used_today: 0,
  used_this_week: 0,
  left_today: 5,
  left_this_week: 20,
  day_resets_at: "",
  week_resets_at: "",
  live: false,
}

const store = createStoredState<Usage>("sdc.usage.v2", STATIC_USAGE, (value) => {
  const usage = value as Usage
  if (!usage || typeof usage.left_today !== "number") return STATIC_USAGE
  if (usage.day_resets_at && new Date(usage.day_resets_at).getTime() <= Date.now()) {
    const weekExpired =
      usage.week_resets_at && new Date(usage.week_resets_at).getTime() <= Date.now()
    const leftWeek = weekExpired ? usage.runs_per_week : usage.left_this_week
    return {
      ...STATIC_USAGE,
      runs_per_day: usage.runs_per_day,
      runs_per_week: usage.runs_per_week,
      left_today: Math.min(usage.runs_per_day, leftWeek),
      left_this_week: leftWeek,
      used_this_week: weekExpired ? 0 : usage.used_this_week,
    }
  }
  return usage
})

export const useUsage = store.useValue

export function saveUsage(usage: Omit<Usage, "live"> | null | undefined) {
  if (usage) store.write({ ...usage, live: true })
}

export async function refreshUsage(): Promise<Usage> {
  const usage = await api.get<Usage>("/api/me/usage")
  saveUsage(usage)
  return { ...usage, live: true }
}

export function clearUsage() {
  store.clear()
}
