"use client"

import { useSyncExternalStore } from "react"

export function createStoredState<T>(
  key: string,
  fallback: T,
  validate: (value: unknown) => T = (v) => v as T,
) {
  const listeners = new Set<() => void>()
  let cache: T | undefined
  let cacheRaw: string | null | undefined

  function read(): T {
    let raw: string | null = null
    try {
      raw = window.localStorage.getItem(key)
    } catch {}
    if (cache !== undefined && raw === cacheRaw) return cache
    cacheRaw = raw
    try {
      cache = raw ? validate(JSON.parse(raw)) : fallback
    } catch {
      cache = fallback
    }
    return cache
  }

  function write(value: T) {
    try {
      window.localStorage.setItem(key, JSON.stringify(value))
    } catch {}
    cache = value
    try {
      cacheRaw = window.localStorage.getItem(key)
    } catch {
      cacheRaw = undefined
    }
    listeners.forEach((l) => l())
  }

  function clear() {
    try {
      window.localStorage.removeItem(key)
    } catch {}
    cache = undefined
    cacheRaw = undefined
    listeners.forEach((l) => l())
  }

  function subscribe(listener: () => void) {
    listeners.add(listener)
    const onStorage = (event: StorageEvent) => event.key === key && listener()
    window.addEventListener("storage", onStorage)
    return () => {
      listeners.delete(listener)
      window.removeEventListener("storage", onStorage)
    }
  }

  function useValue(): T {
    return useSyncExternalStore(subscribe, read, () => fallback)
  }

  return { read, write, clear, subscribe, useValue }
}
