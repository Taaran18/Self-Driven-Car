"use client"

import { Brain, Gauge, Move, Radar, Scale } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { cn } from "@/lib/cn"
import {
  DecideIllustration,
  MoveIllustration,
  NetworkIllustration,
  ScoreIllustration,
  SensorIllustration,
} from "./illustrations"

const steps = [
  {
    icon: Radar,
    title: "Sense",
    heading: "Eight Rays Measure the Road",
    text: "Every car casts eight rays around itself, like a simple lidar. Each ray reports how close the nearest wall is. The car also knows its own speed, which makes nine numbers in total.",
    Illustration: SensorIllustration,
  },
  {
    icon: Brain,
    title: "Think",
    heading: "A Tiny Neural Network Processes Them",
    text: "The nine numbers flow into the car's own neural network. Every connection has a weight that strengthens or weakens a signal. Out come four numbers, one for each possible action.",
    Illustration: NetworkIllustration,
  },
  {
    icon: Gauge,
    title: "Decide",
    heading: "Numbers Become Driving Actions",
    text: "An action only happens if its number passes 0.5 and beats its opposite. So Accelerate competes with Brake, and Left competes with Right. Nobody wrote a rule like “turn when a wall is close.”",
    Illustration: DecideIllustration,
  },
  {
    icon: Move,
    title: "Move",
    heading: "Simple Physics Moves the Car",
    text: "Turning changes the heading by two degrees. The pedal speeds the car up while friction slows it down. The car then moves forward along its heading, thirty times every second.",
    Illustration: MoveIllustration,
  },
  {
    icon: Scale,
    title: "Score",
    heading: "Distance Becomes Fitness",
    text: "The further a car drives, the higher its fitness. Hitting a wall, falling far behind, reversing, or stalling ends its turn. Fitness is the only feedback evolution gets.",
    Illustration: ScoreIllustration,
  },
]

export function LearnSteps() {
  const [active, setActive] = useState(0)
  const refs = useRef<(HTMLElement | null)[]>([])

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0]
        if (visible) setActive(Number((visible.target as HTMLElement).dataset.index))
      },
      { rootMargin: "-35% 0px -45% 0px", threshold: [0, 0.25, 0.5, 1] },
    )
    refs.current.forEach((node) => node && observer.observe(node))
    return () => observer.disconnect()
  }, [])

  const ActiveIllustration = steps[active].Illustration

  return (
    <div className="mx-auto mt-14 grid max-w-7xl gap-10 lg:grid-cols-2 lg:gap-16">
      <div className="hidden lg:block">
        <div className="sticky top-28">
          <div className="rounded-3xl border border-border bg-surface p-8 shadow-[var(--shadow-card)]">
            <div className="mb-6 flex items-center gap-2">
              {steps.map((step, i) => (
                <span
                  key={step.title}
                  className={cn(
                    "h-1.5 flex-1 rounded-full transition-colors duration-300",
                    i <= active ? "bg-primary" : "bg-surface-3",
                  )}
                />
              ))}
            </div>
            <p className="text-sm font-semibold tracking-wider text-primary uppercase">
              Step {active + 1} of 5 · {steps[active].title}
            </p>
            <div key={active} className="mt-4 animate-dialog-in">
              <ActiveIllustration className="mx-auto h-[320px] w-full max-w-[360px]" />
            </div>
          </div>
        </div>
      </div>
      <ol className="space-y-6 lg:space-y-[28vh] lg:py-[12vh]">
        {steps.map((step, i) => (
          <li
            key={step.title}
            ref={(node) => {
              refs.current[i] = node
            }}
            data-index={i}
            className={cn(
              "rounded-3xl border p-6 transition-[border-color,background-color,opacity] duration-300 sm:p-8",
              i === active
                ? "border-primary/40 bg-surface shadow-[var(--shadow-card)]"
                : "border-border bg-surface/60 lg:opacity-60",
            )}
          >
            <div className="flex items-center gap-3">
              <span className="flex size-11 items-center justify-center rounded-xl bg-primary-soft text-primary-soft-fg">
                <step.icon className="size-5" />
              </span>
              <span className="text-sm font-bold tracking-wider text-fg-subtle uppercase">
                {i + 1}. {step.title}
              </span>
            </div>
            <h3 className="mt-4 text-2xl font-bold text-fg">{step.heading}</h3>
            <p className="mt-3 leading-relaxed text-fg-muted">{step.text}</p>
            <step.Illustration className="mx-auto mt-6 h-56 w-full max-w-xs lg:hidden" />
          </li>
        ))}
      </ol>
    </div>
  )
}
