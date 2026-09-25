import Image from "next/image"
import { NetworkIllustration } from "./illustrations"

export function HeroVisual() {
  return (
    <div className="relative mx-auto mt-16 max-w-7xl" aria-hidden>
      <div className="absolute -inset-x-10 -top-10 -bottom-10 -z-10 rounded-[3rem] bg-[radial-gradient(ellipse_at_center,var(--primary-soft),transparent_70%)]" />
      <div className="grid gap-4 rounded-3xl border border-border bg-surface/80 p-3 shadow-[var(--shadow-pop)] backdrop-blur sm:p-4 md:grid-cols-[1.3fr_1fr]">
        <div className="relative h-72 overflow-hidden rounded-2xl bg-[var(--track-ground)] sm:h-96">
          <div className="absolute inset-y-0 left-1/2 w-48 -translate-x-1/2 border-x-4 border-[var(--track-edge)] bg-[var(--track-road)] sm:w-56" />
          <div className="road-dash absolute inset-y-0 left-1/2 w-1.5 -translate-x-1/2" />
          <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 animate-float-car">
            <div className="relative">
              {[-135, -90, -45, 0, 45, 90, 135, 180].map((angle) => (
                <span
                  key={angle}
                  className="absolute top-1/2 left-1/2 h-0.5 w-24 origin-left bg-gradient-to-r from-success/90 to-transparent sm:w-28"
                  style={{ transform: `rotate(${angle - 90}deg)` }}
                />
              ))}
              <div
                className="absolute top-1/2 left-1/2 h-0.5 w-32 origin-left animate-sweep bg-gradient-to-r from-primary to-transparent"
                style={{ transformOrigin: "0 50%" }}
              />
              <Image
                src="/cars/blue.png"
                alt=""
                width={120}
                height={69}
                priority
                className="relative w-16 rotate-90 sm:w-20"
              />
            </div>
          </div>
          <div className="absolute bottom-3 left-3 rounded-lg bg-surface/90 px-2.5 py-1 text-xs font-semibold text-fg backdrop-blur">
            8 sensors · 9 inputs
          </div>
        </div>
        <div className="flex flex-col rounded-2xl border border-border bg-surface-2 p-4">
          <p className="text-xs font-semibold tracking-wider text-fg-subtle uppercase">
            The Car&apos;s Brain
          </p>
          <NetworkIllustration className="mx-auto mt-2 h-56 w-full max-w-xs sm:h-64" />
          <div className="mt-auto grid grid-cols-3 gap-2 text-center text-xs">
            <span className="rounded-lg bg-success-soft px-2 py-1.5 font-semibold text-success">
              Sense
            </span>
            <span className="rounded-lg bg-violet-soft px-2 py-1.5 font-semibold text-violet">
              Think
            </span>
            <span className="rounded-lg bg-warning-soft px-2 py-1.5 font-semibold text-warning">
              Act
            </span>
          </div>
        </div>
      </div>
    </div>
  )
}
