"use client"

import { useEffect, useRef } from "react"
import type { RenderCar, SimulationController } from "@/lib/simulation/controller"

const CAR_LENGTH = 120
const CAR_WIDTH = 69
const SPRITES = ["yellow", "red", "blue", "green"]
const EFFECT_MS = 900

interface Palette {
  ground: string
  road: string
  edge: string
  lane: string
  primary: string
  danger: string
  success: string
  warning: string
  fg: string
}

function readPalette(): Palette {
  const style = getComputedStyle(document.documentElement)
  const v = (name: string) => style.getPropertyValue(name).trim()
  return {
    ground: v("--track-ground"),
    road: v("--track-road"),
    edge: v("--track-edge"),
    lane: v("--track-lane"),
    primary: v("--primary"),
    danger: v("--danger"),
    success: v("--success"),
    warning: v("--warning"),
    fg: v("--fg"),
  }
}

function lerp(a: number, b: number, t: number) {
  return a + (b - a) * t
}

function lerpAngle(a: number, b: number, t: number) {
  const diff = ((((b - a) % 360) + 540) % 360) - 180
  return a + diff * t
}

export function TrackCanvas({
  controller,
  showSensors,
  label,
}: {
  controller: SimulationController
  showSensors: boolean
  label: string
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const sensorsRef = useRef(showSensors)

  useEffect(() => {
    sensorsRef.current = showSensors
  }, [showSensors])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const context = canvas.getContext("2d", { alpha: false })
    if (!context) return
    const ctx: CanvasRenderingContext2D = context

    const images = SPRITES.map((name) => {
      const img = new Image()
      img.src = `/cars/${name}.png`
      return img
    })
    const brakes = new Image()
    brakes.src = "/cars/brakes.png"

    let palette = readPalette()
    const themeObserver = new MutationObserver(() => {
      palette = readPalette()
    })
    themeObserver.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    })

    let width = 0
    let height = 0
    let dpr = 1
    const resize = () => {
      const rect = canvas.getBoundingClientRect()
      dpr = Math.min(window.devicePixelRatio || 1, 2)
      width = rect.width
      height = rect.height
      canvas.width = Math.max(1, Math.round(width * dpr))
      canvas.height = Math.max(1, Math.round(height * dpr))
    }
    const resizeObserver = new ResizeObserver(resize)
    resizeObserver.observe(canvas)
    resize()

    const camera = { x: 0, y: 0 }
    let last = performance.now()
    let frame = 0
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw)
      const dt = Math.min(0.1, (now - last) / 1000)
      last = now
      const r = controller.render
      const scale = Math.min(height / 1500, width / 820)
      const anchorY = height * 0.7

      const t = Math.min(1, Math.max(0, (now - r.receivedAt) / r.interval))
      const cars: RenderCar[] = []
      for (const car of r.current.values()) {
        const prev = r.previous.get(car.id)
        cars.push(
          prev
            ? {
                ...car,
                x: lerp(prev.x, car.x, t),
                y: lerp(prev.y, car.y, t),
                rotation: lerpAngle(prev.rotation, car.rotation, t),
              }
            : car,
        )
      }
      const focus =
        cars.find((c) => c.id === r.focusId) ??
        cars.reduce<RenderCar | null>((best, c) => (!best || c.y < best.y ? c : best), null)
      if (focus) {
        if (r.resetCamera || reduceMotion) {
          camera.x = focus.x
          camera.y = focus.y
          r.resetCamera = false
        } else {
          const k = 1 - Math.exp(-dt * 7)
          camera.x += (focus.x - camera.x) * k
          camera.y += (focus.y - camera.y) * k
        }
      }

      const sx = (x: number) => (x - camera.x) * scale + width / 2
      const sy = (y: number) => (y - camera.y) * scale + anchorY

      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.fillStyle = palette.ground
      ctx.fillRect(0, 0, width, height)

      const road = r.road
      if (road && road.left.length >= 4) {
        const n = road.left.length / 2
        ctx.beginPath()
        for (let i = 0; i < n; i++) {
          const x = sx(road.left[i * 2])
          const y = sy(road.left[i * 2 + 1])
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        for (let i = n - 1; i >= 0; i--)
          ctx.lineTo(sx(road.right[i * 2]), sy(road.right[i * 2 + 1]))
        ctx.closePath()
        ctx.fillStyle = palette.road
        ctx.fill()

        ctx.lineJoin = "round"
        ctx.lineWidth = Math.max(2, 4 * scale)
        ctx.strokeStyle = palette.edge
        for (const side of [road.left, road.right]) {
          ctx.beginPath()
          for (let i = 0; i < n; i++) {
            const x = sx(side[i * 2])
            const y = sy(side[i * 2 + 1])
            if (i === 0) ctx.moveTo(x, y)
            else ctx.lineTo(x, y)
          }
          ctx.stroke()
        }

        ctx.setLineDash([22 * scale, 22 * scale])
        ctx.lineWidth = Math.max(1.5, 3 * scale)
        ctx.strokeStyle = palette.lane
        ctx.beginPath()
        for (let i = 0; i < n; i++) {
          const x = sx((road.left[i * 2] + road.right[i * 2]) / 2)
          const y = sy((road.left[i * 2 + 1] + road.right[i * 2 + 1]) / 2)
          if (i === 0) ctx.moveTo(x, y)
          else ctx.lineTo(x, y)
        }
        ctx.stroke()
        ctx.setLineDash([])

        if (r.finishY !== null) {
          let best = -1
          let bestDist = Infinity
          for (let i = 0; i < n; i++) {
            const d = Math.abs(road.left[i * 2 + 1] - r.finishY)
            if (d < bestDist) {
              bestDist = d
              best = i
            }
          }
          if (best >= 0 && bestDist < 40) {
            const x1 = sx(road.left[best * 2])
            const x2 = sx(road.right[best * 2])
            const y = sy(r.finishY)
            const cell = Math.max(6, 14 * scale)
            const cols = Math.max(2, Math.ceil((x2 - x1) / cell))
            const w = (x2 - x1) / cols
            for (let row = 0; row < 2; row++) {
              for (let col = 0; col < cols; col++) {
                ctx.fillStyle = (row + col) % 2 === 0 ? "#ffffff" : "#0b1117"
                ctx.fillRect(x1 + col * w, y - cell + row * cell, w + 0.5, cell)
              }
            }
          }
        }
      } else {
        const roadWidth = 200 * scale
        ctx.fillStyle = palette.road
        ctx.fillRect(width / 2 - roadWidth / 2, 0, roadWidth, height)
        ctx.fillStyle = palette.edge
        ctx.fillRect(width / 2 - roadWidth / 2 - 2, 0, 3, height)
        ctx.fillRect(width / 2 + roadWidth / 2 - 1, 0, 3, height)
      }

      for (let i = r.effects.length - 1; i >= 0; i--) {
        const effect = r.effects[i]
        const age = now - effect.at
        if (age > EFFECT_MS) {
          r.effects.splice(i, 1)
          continue
        }
        const alpha = 1 - age / EFFECT_MS
        const x = sx(effect.x)
        const y = sy(effect.y)
        ctx.globalAlpha = alpha * 0.9
        const size = 16 * Math.max(0.6, scale)
        ctx.lineWidth = 3
        if (effect.kind === "crash") {
          ctx.strokeStyle = palette.danger
          ctx.beginPath()
          ctx.moveTo(x - size, y - size)
          ctx.lineTo(x + size, y + size)
          ctx.moveTo(x + size, y - size)
          ctx.lineTo(x - size, y + size)
          ctx.stroke()
        } else {
          ctx.strokeStyle = palette.success
          ctx.beginPath()
          ctx.arc(x, y, size * (1 + age / EFFECT_MS), 0, Math.PI * 2)
          ctx.stroke()
        }
        ctx.globalAlpha = 1
      }

      const traced = cars.find((c) => c.id === r.trace?.genome_id)
      if (traced && sensorsRef.current && r.trace) {
        const { distances, range } = r.trace.sense
        const cx = sx(traced.x)
        const cy = sy(traced.y)
        distances.forEach((distance, k) => {
          const angle = ((traced.rotation + 45 * k) * Math.PI) / 180
          const ex = cx + Math.sin(angle) * distance * scale
          const ey = cy - Math.cos(angle) * distance * scale
          const closeness = 1 - distance / range
          const color =
            closeness > 0.6 ? palette.danger : closeness > 0.3 ? palette.warning : palette.success
          ctx.strokeStyle = color
          ctx.globalAlpha = 0.85
          ctx.lineWidth = 1.5
          ctx.beginPath()
          ctx.moveTo(cx, cy)
          ctx.lineTo(ex, ey)
          ctx.stroke()
          if (distance < range - 0.5) {
            ctx.fillStyle = color
            ctx.beginPath()
            ctx.arc(ex, ey, 3.5, 0, Math.PI * 2)
            ctx.fill()
          }
          ctx.globalAlpha = 1
        })
      }

      const w = CAR_WIDTH * scale
      const l = CAR_LENGTH * scale
      for (const car of cars) {
        const x = sx(car.x)
        const y = sy(car.y)
        if (x < -l || x > width + l || y < -l || y > height + l) continue
        const isFocus = focus?.id === car.id
        ctx.save()
        ctx.translate(x, y)
        if (isFocus) {
          ctx.strokeStyle = palette.primary
          ctx.lineWidth = 2.5
          ctx.globalAlpha = 0.9
          ctx.beginPath()
          ctx.arc(0, 0, l * 0.62, 0, Math.PI * 2)
          ctx.stroke()
          ctx.globalAlpha = 1
        } else if (r.focusId !== null) {
          ctx.globalAlpha = 0.72
        }
        ctx.rotate(((car.rotation + 90) * Math.PI) / 180)
        const sprite = images[Math.abs(car.id) % images.length]
        if (sprite.complete && sprite.naturalWidth) ctx.drawImage(sprite, -l / 2, -w / 2, l, w)
        else {
          ctx.fillStyle = palette.primary
          ctx.fillRect(-l / 2, -w / 2, l, w)
        }
        if (car.braking && brakes.complete && brakes.naturalWidth)
          ctx.drawImage(brakes, -l / 2, -w / 2, l, w)
        ctx.restore()
      }
    }
    frame = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(frame)
      resizeObserver.disconnect()
      themeObserver.disconnect()
    }
  }, [controller])

  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label={label}
      className="absolute inset-0 block size-full"
    />
  )
}
