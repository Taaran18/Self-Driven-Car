"use client"

import { useCallback, useEffect, useState } from "react"
import { api, errorMessage } from "@/lib/api"

interface State<T> {
  key: string | null
  data?: T
  error?: string
}

export function useApi<T>(path: string | null) {
  const [nonce, setNonce] = useState(0)
  const [state, setState] = useState<State<T>>({ key: null })
  const key = path ? `${path}#${nonce}` : null

  useEffect(() => {
    if (!path || !key) return
    const controller = new AbortController()
    api.get<T>(path, { signal: controller.signal }).then(
      (data) => setState({ key, data }),
      (error) => {
        if (controller.signal.aborted) return
        setState((s) => ({ key, data: s.data, error: errorMessage(error) }))
      },
    )
    return () => controller.abort()
  }, [path, key])

  const reload = useCallback(() => setNonce((n) => n + 1), [])
  const mutate = useCallback((update: (data: T | undefined) => T | undefined) => {
    setState((s) => ({ ...s, data: update(s.data) }))
  }, [])

  return {
    data: state.data,
    error: state.key === key ? state.error : undefined,
    loading: state.key !== key,
    reload,
    mutate,
  }
}
