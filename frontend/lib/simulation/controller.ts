import { api, ApiError, errorMessage, onServerWaking } from "@/lib/api"
import { site } from "@/lib/site"
import type { NetworkShape, RunConfig, Speed } from "@/lib/types"
import { saveUsage, type Usage } from "@/lib/usage"
import type {
  CarRow,
  CodeStep,
  EndedMessage,
  FrameMessage,
  GenerationMessage,
  GenerationStartMessage,
  RetiredCar,
  ServerMessage,
  StartedMessage,
  Trace,
} from "./types"

export type Phase =
  "idle" | "waking" | "connecting" | "starting" | "running" | "paused" | "ended" | "error"

export interface FrameSummary {
  tick: number
  alive: number
  finished: number
  stepped: boolean
  trace: Trace
}

export interface SimSnapshot {
  phase: Phase
  speed: Speed
  run: StartedMessage | null
  code: CodeStep[] | null
  generation: GenerationStartMessage | null
  frame: FrameSummary | null
  network: NetworkShape | null
  history: GenerationMessage[]
  bestEver: number | null
  ended: EndedMessage | null
  error: { code: string; message: string } | null
  notice: string | null
  stepSeq: number
}

export interface RenderCar {
  id: number
  x: number
  y: number
  rotation: number
  braking: boolean
}

export interface RetiredEffect extends RetiredCar {
  at: number
  kind: "crash" | "finish"
}

export interface RenderState {
  road: { left: Float32Array; right: Float32Array; version: number; width: number } | null
  finishY: number | null
  previous: Map<number, RenderCar>
  current: Map<number, RenderCar>
  receivedAt: number
  interval: number
  focusId: number | null
  trace: Trace | null
  effects: RetiredEffect[]
  resetCamera: boolean
}

const START_ERRORS = new Set([
  "trial_limit",
  "capacity",
  "invalid_config",
  "start_failed",
  "ticket_invalid",
  "missing_visitor",
  "rate_limited",
])

const initialSnapshot: SimSnapshot = {
  phase: "idle",
  speed: "1",
  run: null,
  code: null,
  generation: null,
  frame: null,
  network: null,
  history: [],
  bestEver: null,
  ended: null,
  error: null,
  notice: null,
  stepSeq: 0,
}

function toCars(rows: CarRow[]): Map<number, RenderCar> {
  const map = new Map<number, RenderCar>()
  for (const [id, x, y, rotation, braking] of rows)
    map.set(id, { id, x, y, rotation, braking: braking > 0 })
  return map
}

export class SimulationController {
  snapshot: SimSnapshot = initialSnapshot
  render: RenderState = {
    road: null,
    finishY: null,
    previous: new Map(),
    current: new Map(),
    receivedAt: 0,
    interval: 33,
    focusId: null,
    trace: null,
    effects: [],
    resetCamera: true,
  }
  private listeners = new Set<() => void>()
  private ws: WebSocket | null = null
  private pendingConfig: RunConfig | null = null
  private helloReceived = false
  private frameDirty: FrameSummary | null = null
  private flushTimer: ReturnType<typeof setInterval> | null = null
  private unsubscribeWaking: (() => void) | null = null
  private disposed = false

  subscribe = (listener: () => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  getSnapshot = () => this.snapshot

  private set(patch: Partial<SimSnapshot>) {
    this.snapshot = { ...this.snapshot, ...patch }
    this.listeners.forEach((l) => l())
  }

  get busy() {
    return ["waking", "connecting", "starting", "running", "paused"].includes(this.snapshot.phase)
  }

  async play(config: RunConfig) {
    if (this.busy) return
    this.pendingConfig = config
    this.set({ error: null, notice: null, ended: null, speed: config.speed })
    if (this.ws && this.ws.readyState === WebSocket.OPEN && this.helloReceived) {
      this.sendStart()
      return
    }
    await this.connect()
  }

  private async connect() {
    this.set({ phase: "connecting" })
    this.unsubscribeWaking?.()
    this.unsubscribeWaking = onServerWaking((waking) => {
      if (this.snapshot.phase === "connecting" || this.snapshot.phase === "waking") {
        this.set({ phase: waking ? "waking" : "connecting" })
      }
    })
    let ticket: string
    try {
      const response = await api.post<{ ticket: string; usage: Usage }>(
        "/api/simulation/ticket",
        {},
        { retries: 5 },
      )
      ticket = response.ticket
      saveUsage(response.usage)
    } catch (error) {
      if (error instanceof ApiError && error.code === "trial_limit") {
        saveUsage(error.details.usage as Usage)
      }
      this.fail(error instanceof ApiError ? error.code : "network", errorMessage(error))
      return
    } finally {
      this.unsubscribeWaking?.()
      this.unsubscribeWaking = null
    }
    if (this.disposed) return

    const url = `${site.apiUrl.replace(/^http/, "ws")}/ws/simulation?ticket=${encodeURIComponent(ticket)}`
    const ws = new WebSocket(url)
    this.ws = ws
    this.helloReceived = false
    ws.onmessage = (event) => {
      try {
        this.handle(JSON.parse(event.data) as ServerMessage)
      } catch {}
    }
    ws.onclose = () => {
      if (this.ws !== ws) return
      this.ws = null
      this.helloReceived = false
      this.stopFlush()
      const phase = this.snapshot.phase
      if (
        phase === "running" ||
        phase === "paused" ||
        phase === "starting" ||
        phase === "connecting"
      ) {
        this.fail(
          "connection_lost",
          "We lost the connection to the simulation server, so this run stopped. Check your connection and start a new run.",
        )
      }
    }
    ws.onerror = () => {}
  }

  private sendStart() {
    if (!this.pendingConfig) return
    this.set({ phase: "starting" })
    this.send({ type: "start", config: this.pendingConfig })
  }

  private fail(code: string, message: string) {
    this.stopFlush()
    this.set({ phase: "error", error: { code, message } })
  }

  private send(message: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) this.ws.send(JSON.stringify(message))
  }

  pause() {
    this.send({ type: "pause" })
  }

  resume() {
    this.send({ type: "resume" })
  }

  step() {
    this.send({ type: "step" })
  }

  skip() {
    this.send({ type: "skip" })
  }

  stop() {
    this.send({ type: "stop" })
  }

  setSpeed(speed: Speed) {
    this.set({ speed })
    this.send({ type: "speed", value: speed })
  }

  reset() {
    if (this.busy) return
    this.render.road = null
    this.render.current = new Map()
    this.render.previous = new Map()
    this.render.effects = []
    this.render.trace = null
    this.render.resetCamera = true
    this.set({ ...initialSnapshot, code: this.snapshot.code, speed: this.snapshot.speed })
  }

  attach() {
    this.disposed = false
  }

  dispose() {
    this.disposed = true
    this.stopFlush()
    this.unsubscribeWaking?.()
    const ws = this.ws
    this.ws = null
    ws?.close()
  }

  private startFlush() {
    if (this.flushTimer) return
    this.flushTimer = setInterval(() => {
      if (this.frameDirty) {
        const frame = this.frameDirty
        this.frameDirty = null
        this.set({ frame })
      }
    }, 160)
  }

  private stopFlush() {
    if (this.flushTimer) clearInterval(this.flushTimer)
    this.flushTimer = null
    if (this.frameDirty) {
      const frame = this.frameDirty
      this.frameDirty = null
      this.set({ frame })
    }
  }

  private handle(message: ServerMessage) {
    switch (message.type) {
      case "hello":
        this.helloReceived = true
        this.set({ code: message.code })
        this.sendStart()
        break
      case "started":
        saveUsage(message.usage)
        this.render.road = null
        this.render.effects = []
        this.render.resetCamera = true
        this.set({
          phase: "running",
          run: message,
          history: [],
          bestEver: null,
          generation: null,
          frame: null,
          network: null,
          stepSeq: 0,
        })
        this.startFlush()
        break
      case "status":
        if (
          this.snapshot.phase === "running" ||
          this.snapshot.phase === "paused" ||
          this.snapshot.phase === "starting"
        ) {
          this.set({ phase: message.state, speed: message.speed })
        }
        break
      case "generation_start":
        this.render.finishY = message.finish_y
        this.render.current = new Map()
        this.render.previous = new Map()
        this.render.resetCamera = true
        this.set({ generation: message })
        break
      case "road":
        this.render.road = {
          left: new Float32Array(message.left),
          right: new Float32Array(message.right),
          version: message.version,
          width: message.width,
        }
        this.render.finishY = message.finish_y
        break
      case "network": {
        const { type: _type, ...network } = message
        void _type
        this.set({ network })
        break
      }
      case "frame":
        this.onFrame(message)
        break
      case "generation": {
        const best = this.snapshot.bestEver
        this.set({
          history: [...this.snapshot.history, message],
          bestEver: best === null ? message.best_fitness : Math.max(best, message.best_fitness),
        })
        break
      }
      case "ended":
        this.stopFlush()
        saveUsage(message.usage)
        this.set({ phase: "ended", ended: message })
        break
      case "error":
        if (
          START_ERRORS.has(message.code) &&
          !["running", "paused"].includes(this.snapshot.phase)
        ) {
          if (message.code === "trial_limit" && message.details?.usage)
            saveUsage(message.details.usage as Usage)
          this.fail(message.code, message.message)
        } else {
          this.set({ error: { code: message.code, message: message.message } })
        }
        break
      case "idle":
        this.set({ notice: message.message })
        break
    }
  }

  private onFrame(message: FrameMessage) {
    const now = performance.now()
    const r = this.render
    const delta = now - r.receivedAt
    if (r.receivedAt && delta < 400)
      r.interval = r.interval * 0.8 + Math.min(200, Math.max(16, delta)) * 0.2
    r.previous = r.current
    r.current = toCars(message.cars)
    r.receivedAt = now
    r.focusId = message.trace.genome_id
    r.trace = message.trace
    for (const car of message.eliminated) r.effects.push({ ...car, at: now, kind: "crash" })
    for (const car of message.crossed) r.effects.push({ ...car, at: now, kind: "finish" })
    if (r.effects.length > 80) r.effects.splice(0, r.effects.length - 80)
    const summary: FrameSummary = {
      tick: message.tick,
      alive: message.alive,
      finished: message.finished,
      stepped: message.stepped,
      trace: message.trace,
    }
    if (message.stepped) {
      this.frameDirty = null
      this.set({ frame: summary, stepSeq: this.snapshot.stepSeq + 1 })
    } else {
      this.frameDirty = summary
    }
  }
}
