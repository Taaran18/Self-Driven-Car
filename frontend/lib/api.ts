import { site } from "@/lib/site"

export class ApiError extends Error {
  status: number
  code: string
  fields: Record<string, string>
  details: Record<string, unknown>

  constructor(
    status: number,
    code: string,
    message: string,
    fields: Record<string, string> = {},
    details: Record<string, unknown> = {},
  ) {
    super(message)
    this.status = status
    this.code = code
    this.fields = fields
    this.details = details
  }
}

const VISITOR_KEY = "sdc.visitor"
let memoryVisitor: string | null = null

function createId(): string {
  const cryptoApi = globalThis.crypto
  if (typeof cryptoApi?.randomUUID === "function") return cryptoApi.randomUUID()
  const bytes = new Uint8Array(16)
  cryptoApi.getRandomValues(bytes)
  bytes[6] = (bytes[6] & 0x0f) | 0x40
  bytes[8] = (bytes[8] & 0x3f) | 0x80
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function visitorId(): string {
  try {
    let id = window.localStorage.getItem(VISITOR_KEY)
    if (!id) {
      id = createId()
      window.localStorage.setItem(VISITOR_KEY, id)
    }
    return id
  } catch {
    memoryVisitor ||= createId()
    return memoryVisitor
  }
}

export function resetVisitor() {
  memoryVisitor = null
  try {
    window.localStorage.removeItem(VISITOR_KEY)
  } catch {}
}

type WakeListener = (waking: boolean) => void
const wakeListeners = new Set<WakeListener>()

export function onServerWaking(listener: WakeListener) {
  wakeListeners.add(listener)
  return () => {
    wakeListeners.delete(listener)
  }
}

function setWaking(value: boolean) {
  wakeListeners.forEach((l) => l(value))
}

type Method = "GET" | "POST" | "PUT" | "PATCH" | "DELETE"

interface RequestOptions {
  body?: unknown
  signal?: AbortSignal
  retries?: number
  headers?: Record<string, string>
}

const RETRYABLE = new Set([502, 503, 504])
const DELAYS = [1000, 2000, 3000, 5000, 6000]
const NETWORK_MESSAGE =
  "We couldn't reach the simulation server. It may still be waking up. Check your connection and try again."

function sleep(ms: number, signal?: AbortSignal) {
  return new Promise<void>((resolve, reject) => {
    const timer = setTimeout(resolve, ms)
    signal?.addEventListener("abort", () => {
      clearTimeout(timer)
      reject(new DOMException("Aborted", "AbortError"))
    })
  })
}

async function parseError(response: Response): Promise<ApiError> {
  let payload: {
    error?: {
      code?: string
      message?: string
      details?: Record<string, unknown> & { fields?: Record<string, string> }
    }
  } = {}
  try {
    payload = await response.json()
  } catch {}
  const error = payload.error
  if (error?.message) {
    return new ApiError(
      response.status,
      error.code ?? "error",
      error.message,
      error.details?.fields ?? {},
      error.details ?? {},
    )
  }
  if (RETRYABLE.has(response.status))
    return new ApiError(response.status, "unavailable", NETWORK_MESSAGE)
  return new ApiError(
    response.status,
    "error",
    "Something went wrong on our side. Try again in a moment.",
  )
}

export async function request<T>(
  method: Method,
  path: string,
  options: RequestOptions = {},
): Promise<T> {
  const { body, signal, headers } = options
  const maxRetries = options.retries ?? (method === "GET" ? DELAYS.length : 2)
  let attempt = 0
  try {
    while (true) {
      try {
        const response = await fetch(`${site.apiUrl}${path}`, {
          method,
          signal,
          headers: {
            "X-Visitor-Id": visitorId(),
            ...(body === undefined ? {} : { "Content-Type": "application/json" }),
            ...headers,
          },
          body: body === undefined ? undefined : JSON.stringify(body),
        })
        if (response.ok) {
          if (response.status === 204) return undefined as T
          return (await response.json()) as T
        }
        if (RETRYABLE.has(response.status) && attempt < maxRetries) {
          setWaking(true)
          await sleep(DELAYS[Math.min(attempt, DELAYS.length - 1)], signal)
          attempt += 1
          continue
        }
        throw await parseError(response)
      } catch (error) {
        if (error instanceof ApiError) throw error
        if (error instanceof DOMException && error.name === "AbortError") throw error
        if (attempt < maxRetries) {
          setWaking(true)
          await sleep(DELAYS[Math.min(attempt, DELAYS.length - 1)], signal)
          attempt += 1
          continue
        }
        throw new ApiError(0, "network", NETWORK_MESSAGE)
      }
    }
  } finally {
    if (attempt > 0) setWaking(false)
  }
}

export const api = {
  get: <T>(path: string, options?: RequestOptions) => request<T>("GET", path, options),
  post: <T>(path: string, body?: unknown, options?: RequestOptions) =>
    request<T>("POST", path, { ...options, body: body ?? {} }),
  patch: <T>(path: string, body: unknown) => request<T>("PATCH", path, { body }),
  delete: <T>(path: string) => request<T>("DELETE", path),
}

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Something went wrong. Try again in a moment."
}

export async function downloadJson(path: string, filename: string) {
  const data = await api.get<unknown>(path)
  const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" })
  const url = URL.createObjectURL(blob)
  const link = document.createElement("a")
  link.href = url
  link.download = filename
  document.body.appendChild(link)
  link.click()
  link.remove()
  setTimeout(() => URL.revokeObjectURL(url), 1000)
}
