"use client"

import { ChevronDown, Play, RotateCcw, SlidersHorizontal } from "lucide-react"
import Link from "next/link"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardHeader } from "@/components/ui/card"
import { Segmented, Slider } from "@/components/ui/controls"
import { Field, Input } from "@/components/ui/field"
import { Select } from "@/components/ui/select"
import { cn } from "@/lib/cn"
import { formatDateTime } from "@/lib/format"
import type { RunConfig } from "@/lib/types"
import { useUsage } from "@/lib/usage"
import {
  CURVE_OPTIONS,
  GENERATION_OPTIONS,
  LENGTH_OPTIONS,
  POPULATION_OPTIONS,
  SPEED_OPTIONS,
  TRACK_MODE_OPTIONS,
  WIDTH_OPTIONS,
} from "./options"

interface RunSetupProps {
  config: RunConfig
  onChange: (config: RunConfig) => void
  onStart: () => void
  onReset: () => void
  busy: boolean
  busyLabel: string
  bare?: boolean
}

export function RunSetup({
  config,
  onChange,
  onStart,
  onReset,
  busy,
  busyLabel,
  bare,
}: RunSetupProps) {
  const [advanced, setAdvanced] = useState(false)
  const usage = useUsage()
  const outOfRuns = usage.live === true && usage.left_today <= 0
  const set = <K extends keyof RunConfig>(key: K, value: RunConfig[K]) =>
    onChange({ ...config, [key]: value })

  const form = (
    <form
      className={cn("space-y-5", !bare && "p-5")}
      onSubmit={(event) => {
        event.preventDefault()
        if (!busy && !outOfRuns) onStart()
      }}
    >
      <Field label="Run Name" optional hint="Helps you find this run later in your history.">
        {({ id, describedBy }) => (
          <Input
            id={id}
            aria-describedby={describedBy}
            value={config.name ?? ""}
            onChange={(e) => set("name", e.target.value)}
            maxLength={80}
            placeholder="Training Run"
          />
        )}
      </Field>
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1 2xl:grid-cols-2">
        <Field label="Cars per Generation">
          {({ id }) => (
            <Select
              id={id}
              value={String(config.population_size)}
              onChange={(v) => set("population_size", Number(v))}
              options={POPULATION_OPTIONS}
            />
          )}
        </Field>
        <Field label="Generation Limit">
          {({ id }) => (
            <Select
              id={id}
              value={String(config.max_generations)}
              onChange={(v) => set("max_generations", Number(v))}
              options={GENERATION_OPTIONS}
            />
          )}
        </Field>
      </div>
      <div className="space-y-1.5">
        <p className="text-sm font-medium text-fg">Track Length</p>
        <Segmented
          value={config.track_length}
          onChange={(v) => set("track_length", v)}
          options={LENGTH_OPTIONS}
          label="Track length"
        />
      </div>
      <Field label="Starting Speed">
        {({ id }) => (
          <Select
            id={id}
            value={config.speed}
            onChange={(v) => set("speed", v)}
            options={SPEED_OPTIONS}
          />
        )}
      </Field>

      <div className="rounded-2xl border border-border">
        <button
          type="button"
          onClick={() => setAdvanced((v) => !v)}
          aria-expanded={advanced}
          aria-controls="advanced-settings"
          className="flex w-full items-center justify-between gap-3 rounded-2xl px-4 py-3 text-left text-sm font-semibold text-fg hover:bg-surface-2"
        >
          <span>
            Advanced Settings
            <span className="ml-2 font-normal text-fg-subtle">Track shape, mutation, and seed</span>
          </span>
          <ChevronDown
            className={cn("size-4 text-fg-subtle transition-transform", advanced && "rotate-180")}
          />
        </button>
        {advanced ? (
          <div id="advanced-settings" className="space-y-5 border-t border-border p-4">
            <div className="space-y-1.5">
              <p className="text-sm font-medium text-fg">Track Width</p>
              <Segmented
                value={config.track_width}
                onChange={(v) => set("track_width", v)}
                options={WIDTH_OPTIONS}
                label="Track width"
              />
            </div>
            <div className="space-y-1.5">
              <p className="text-sm font-medium text-fg">Curves</p>
              <Segmented
                value={config.track_curviness}
                onChange={(v) => set("track_curviness", v)}
                options={CURVE_OPTIONS}
                label="Track curves"
              />
            </div>
            <Field label="Track Variety">
              {({ id }) => (
                <Select
                  id={id}
                  value={config.track_mode}
                  onChange={(v) => set("track_mode", v)}
                  options={TRACK_MODE_OPTIONS}
                />
              )}
            </Field>
            <Field
              label="Track Seed"
              optional
              hint="Use the same seed to get the same sequence of tracks."
            >
              {({ id, describedBy }) => (
                <Input
                  id={id}
                  aria-describedby={describedBy}
                  inputMode="numeric"
                  value={config.track_seed ?? ""}
                  onChange={(e) => {
                    const digits = e.target.value.replace(/\D/g, "").slice(0, 9)
                    set("track_seed", digits ? Number(digits) : null)
                  }}
                  placeholder="Random"
                />
              )}
            </Field>
            <Slider
              label="Weight Mutation Chance"
              value={config.weight_mutation_rate}
              onChange={(v) => set("weight_mutation_rate", v)}
              min={0}
              max={1}
              step={0.05}
              format={(v) => `${Math.round(v * 100)}%`}
              hint="How often a child's connection weights get nudged."
            />
            <Slider
              label="New Connection Chance"
              value={config.add_connection_rate}
              onChange={(v) => set("add_connection_rate", v)}
              min={0}
              max={1}
              step={0.05}
              format={(v) => `${Math.round(v * 100)}%`}
              hint="How often a child grows a new connection between neurons."
            />
            <Slider
              label="New Neuron Chance"
              value={config.add_node_rate}
              onChange={(v) => set("add_node_rate", v)}
              min={0}
              max={1}
              step={0.05}
              format={(v) => `${Math.round(v * 100)}%`}
              hint="How often a child grows a new hidden neuron."
            />
            <Button variant="ghost" size="sm" onClick={onReset}>
              <RotateCcw className="size-4" /> Restore Defaults
            </Button>
          </div>
        ) : null}
      </div>

      <div className="space-y-3 border-t border-border pt-5">
        <Button
          type="submit"
          size="lg"
          className="w-full"
          loading={busy}
          loadingText={busyLabel}
          disabled={outOfRuns}
        >
          <Play className="size-5" /> Start Training
        </Button>
        <p className="text-center text-xs leading-relaxed text-fg-subtle" aria-live="polite">
          {outOfRuns ? (
            <>
              You&apos;ve used today&apos;s {usage.runs_per_day} free runs.{" "}
              {usage.day_resets_at ? `More at ${formatDateTime(usage.day_resets_at)}.` : null}{" "}
              <Link
                href="/settings?tab=usage"
                className="font-semibold text-primary hover:underline"
              >
                See Usage
              </Link>
            </>
          ) : usage.live ? (
            <>
              Uses 1 of your free runs · {usage.left_today} left today, {usage.left_this_week} this
              week.
            </>
          ) : (
            <>
              Free trial: {usage.runs_per_day} runs a day, {usage.runs_per_week} a week. Each start
              uses one run.
            </>
          )}
        </p>
      </div>
    </form>
  )

  if (bare) return form

  return (
    <Card>
      <CardHeader
        title="Set Up a Training Run"
        description="Pick how many cars learn together and how long they train."
        icon={<SlidersHorizontal />}
      />
      {form}
    </Card>
  )
}
