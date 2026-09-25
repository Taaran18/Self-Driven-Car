"use client"

import { createStoredState } from "@/lib/store"
import type { Preferences } from "@/lib/types"

export const DEFAULT_PREFERENCES: Preferences = {
  default_speed: "1",
  default_population: 50,
  default_generations: 50,
  show_sensors: true,
  show_network: true,
  show_code: true,
  confirm_before_stop: true,
  guided_steps: true,
}

const store = createStoredState<Preferences>("sdc.preferences", DEFAULT_PREFERENCES, (value) => ({
  ...DEFAULT_PREFERENCES,
  ...(value as Partial<Preferences>),
}))

export const usePreferences = store.useValue

export function updatePreferences(patch: Partial<Preferences>) {
  store.write({ ...store.read(), ...patch })
}

export function resetPreferences() {
  store.clear()
}
