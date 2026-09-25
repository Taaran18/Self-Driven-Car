"use client"

import {
  Check,
  Copy,
  Database,
  Download,
  Gauge,
  Moon,
  Palette,
  RefreshCw,
  RotateCcw,
  SlidersHorizontal,
  Sun,
  Trash2,
  TriangleAlert,
} from "lucide-react"
import { usePathname, useRouter, useSearchParams } from "next/navigation"
import { useEffect, useState } from "react"
import { UsageMeter } from "@/components/runs/usage-meter"
import { SPEED_OPTIONS } from "@/components/simulator/options"
import { useTheme } from "@/components/theme-provider"
import { Button } from "@/components/ui/button"
import { Card, CardHeader } from "@/components/ui/card"
import { Switch } from "@/components/ui/controls"
import { ConfirmDialog } from "@/components/ui/dialog"
import { InlineAlert } from "@/components/ui/feedback"
import { Field } from "@/components/ui/field"
import { PageHeader } from "@/components/ui/page-header"
import { Select } from "@/components/ui/select"
import { TabList, TabPanel } from "@/components/ui/tabs"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { useToast } from "@/components/ui/toast"
import { api, downloadJson, errorMessage, resetVisitor } from "@/lib/api"
import { cn } from "@/lib/cn"
import { formatDateTime } from "@/lib/format"
import { resetPreferences, updatePreferences, usePreferences } from "@/lib/preferences"
import { clearUsage, refreshUsage, useUsage } from "@/lib/usage"

type Tab = "appearance" | "simulation" | "usage" | "data"
const TABS: { value: Tab; label: string; icon: React.ReactNode }[] = [
  { value: "appearance", label: "Appearance", icon: <Palette /> },
  { value: "simulation", label: "Simulation", icon: <SlidersHorizontal /> },
  { value: "usage", label: "Usage", icon: <Gauge /> },
  { value: "data", label: "Data & Privacy", icon: <Database /> },
]

export function SettingsView() {
  const params = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  const requested = params.get("tab") as Tab | null
  const tab: Tab = TABS.some((t) => t.value === requested) ? (requested as Tab) : "appearance"

  function setTab(next: Tab) {
    router.replace(`${pathname}?tab=${next}`, { scroll: false })
  }

  return (
    <div>
      <section className="bg-bg px-4 pt-10 pb-10 sm:px-6 lg:px-8">
        <PageHeader
          eyebrow="Preferences"
          title="Settings"
          description="Adjust how the lab looks and behaves, check your free usage, and manage your data."
        />
      </section>
      <section className="border-t border-border bg-bg-alt px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto grid max-w-[1600px] gap-6 lg:grid-cols-[280px_minmax(0,1fr)] xl:gap-8">
          <Card className="h-fit min-w-0 p-2 lg:sticky lg:top-6">
            <TabList
              items={TABS}
              value={tab}
              onChange={setTab}
              label="Settings sections"
              idPrefix="settings"
              className="grid grid-cols-2 sm:flex lg:flex-col lg:overflow-visible"
            />
          </Card>
          <TabPanel idPrefix="settings" value={tab} className="min-w-0 space-y-6">
            {tab === "appearance" ? <AppearanceSection /> : null}
            {tab === "simulation" ? <SimulationSection /> : null}
            {tab === "usage" ? <UsageSection /> : null}
            {tab === "data" ? <DataSection /> : null}
          </TabPanel>
        </div>
      </section>
    </div>
  )
}

function AppearanceSection() {
  const { theme, setTheme } = useTheme()
  const options = [
    {
      value: "dark" as const,
      label: "Dark",
      description: "Pure black background. Easy on the eyes at night.",
      icon: Moon,
    },
    {
      value: "light" as const,
      label: "Light",
      description: "Clean white background for bright rooms.",
      icon: Sun,
    },
  ]
  return (
    <Card>
      <CardHeader
        title="Theme"
        description="Choose how the lab looks on this device."
        icon={<Palette />}
      />
      <div role="radiogroup" aria-label="Theme" className="grid gap-4 p-5 sm:grid-cols-2">
        {options.map((option) => {
          const selected = theme === option.value
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={selected}
              onClick={() => setTheme(option.value)}
              className={cn(
                "group rounded-2xl border-2 p-3 text-left transition-colors",
                selected
                  ? "border-primary bg-primary-soft"
                  : "border-border hover:border-border-strong",
              )}
            >
              <div
                className={cn(
                  "flex h-28 flex-col gap-2 rounded-xl border p-3",
                  option.value === "dark"
                    ? "border-neutral-800 bg-black"
                    : "border-neutral-200 bg-white",
                )}
                aria-hidden
              >
                <div
                  className={cn(
                    "h-2.5 w-1/2 rounded-full",
                    option.value === "dark" ? "bg-neutral-700" : "bg-neutral-300",
                  )}
                />
                <div
                  className={cn(
                    "h-2.5 w-3/4 rounded-full",
                    option.value === "dark" ? "bg-neutral-800" : "bg-neutral-200",
                  )}
                />
                <div
                  className={cn(
                    "mt-auto h-6 w-20 rounded-lg",
                    option.value === "dark" ? "bg-cyan-400" : "bg-cyan-700",
                  )}
                />
              </div>
              <div className="mt-3 flex items-center justify-between gap-3 px-1">
                <span className="flex items-center gap-2 font-semibold text-fg">
                  <option.icon className="size-4" /> {option.label}
                </span>
                <span
                  className={cn(
                    "flex size-5 items-center justify-center rounded-full border-2",
                    selected ? "border-primary bg-primary text-primary-fg" : "border-border-strong",
                  )}
                >
                  {selected ? <Check className="size-3" /> : null}
                </span>
              </div>
              <p className="mt-1 px-1 text-sm text-fg-muted">{option.description}</p>
            </button>
          )
        })}
      </div>
    </Card>
  )
}

function SimulationSection() {
  const prefs = usePreferences()
  const toast = useToast()
  const [confirming, setConfirming] = useState(false)

  return (
    <>
      <Card>
        <CardHeader
          title="Run Defaults"
          description="Used to fill in the setup form each time you open the simulator."
          icon={<SlidersHorizontal />}
        />
        <div className="grid gap-5 p-5 sm:grid-cols-3">
          <Field label="Starting Speed">
            {({ id }) => (
              <Select
                id={id}
                value={prefs.default_speed}
                onChange={(v) => updatePreferences({ default_speed: v })}
                options={SPEED_OPTIONS}
              />
            )}
          </Field>
          <Field label="Cars per Generation">
            {({ id }) => (
              <Select
                id={id}
                value={String(prefs.default_population)}
                onChange={(v) => updatePreferences({ default_population: Number(v) })}
                options={[20, 30, 50, 80, 100, 150].map((n) => ({
                  value: String(n),
                  label: `${n} Cars`,
                }))}
              />
            )}
          </Field>
          <Field label="Generation Limit">
            {({ id }) => (
              <Select
                id={id}
                value={String(prefs.default_generations)}
                onChange={(v) => updatePreferences({ default_generations: Number(v) })}
                options={[10, 25, 50, 100, 200].map((n) => ({
                  value: String(n),
                  label: `${n} Generations`,
                }))}
              />
            )}
          </Field>
        </div>
      </Card>
      <Card>
        <CardHeader
          title="What to Show"
          description="Turn panels on or off to focus on what you're learning."
          icon={<Gauge />}
        />
        <div className="divide-y divide-border px-5">
          {(
            [
              ["show_sensors", "Sensor Rays", "Draw the leading car's 8 sensor rays on the track."],
              [
                "show_network",
                "Network Diagram",
                "Show the leading car's neural network with live activations.",
              ],
              ["show_code", "Code Walkthrough", "Show the live Python code panel below the track."],
              [
                "guided_steps",
                "Guided Step Walkthrough",
                "After each Step, walk the code panel through all five stages.",
              ],
              [
                "confirm_before_stop",
                "Confirm Before Stopping",
                "Ask before stopping a run that's still training.",
              ],
            ] as const
          ).map(([key, label, description]) => (
            <div key={key} className="py-4">
              <Switch
                label={label}
                description={description}
                checked={prefs[key]}
                onChange={(v) => updatePreferences({ [key]: v })}
              />
            </div>
          ))}
        </div>
      </Card>
      <div className="flex justify-end">
        <Button variant="outline" onClick={() => setConfirming(true)}>
          <RotateCcw className="size-4" /> Restore Default Settings
        </Button>
      </div>
      <ConfirmDialog
        open={confirming}
        onClose={() => setConfirming(false)}
        tone="default"
        icon={<RotateCcw className="size-5" />}
        title="Restore Default Settings?"
        description="Your run defaults and display options go back to how they were on your first visit. Your theme and run history stay the same."
        confirmLabel="Restore Defaults"
        onConfirm={() => {
          resetPreferences()
          setConfirming(false)
          toast.success("Settings Restored", "Everything is back to the defaults.")
        }}
      />
    </>
  )
}

function UsageSection() {
  const usage = useUsage()
  const toast = useToast()
  const [trialId, setTrialId] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    let cancelled = false
    api.get<{ id: string }>("/api/me").then(
      (me) => {
        if (!cancelled) setTrialId(me.id)
      },
      () => {},
    )
    refreshUsage().catch((err) => {
      if (!cancelled) setError(errorMessage(err))
    })
    return () => {
      cancelled = true
    }
  }, [])

  async function refresh() {
    setLoading(true)
    setError(null)
    try {
      await refreshUsage()
      toast.success("Usage Updated")
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <>
      <Card>
        <CardHeader
          title="Free Trial Usage"
          description="Limits are counted per network (IP address), so everyone on the same Wi-Fi shares them."
          icon={<Gauge />}
          action={
            <Button
              variant="outline"
              size="sm"
              onClick={refresh}
              loading={loading}
              loadingText="Refreshing"
            >
              <RefreshCw className="size-4" /> Refresh
            </Button>
          }
        />
        <div className="space-y-6 p-5">
          {error ? (
            <InlineAlert tone="warning" title="Showing Saved Numbers">
              {error}
            </InlineAlert>
          ) : null}
          <UsageMeter
            label="Runs Today"
            used={usage.used_today}
            limit={usage.runs_per_day}
            hint={
              usage.day_resets_at
                ? `Resets ${formatDateTime(usage.day_resets_at)} (your time)`
                : "Resets at midnight UTC"
            }
          />
          <UsageMeter
            label="Runs This Week"
            used={usage.used_this_week}
            limit={usage.runs_per_week}
            hint={
              usage.week_resets_at
                ? `Resets ${formatDateTime(usage.week_resets_at)} (your time)`
                : "Resets Monday at midnight UTC"
            }
          />
        </div>
      </Card>
      <Card>
        <CardHeader title="How the Limits Work" icon={<SlidersHorizontal />} />
        <div className="p-5">
          <TableWrap label="Trial limits" className="min-w-0">
            <THead>
              <tr>
                <th scope="col">Limit</th>
                <th scope="col">Amount</th>
                <th scope="col">Resets</th>
              </tr>
            </THead>
            <TBody>
              <tr>
                <td className="font-semibold text-fg">Runs per day</td>
                <td className="tabular text-fg-muted">{usage.runs_per_day}</td>
                <td className="text-fg-muted">Every day at midnight UTC</td>
              </tr>
              <tr>
                <td className="font-semibold text-fg">Runs per week</td>
                <td className="tabular text-fg-muted">{usage.runs_per_week}</td>
                <td className="text-fg-muted">Every Monday at midnight UTC</td>
              </tr>
              <tr>
                <td className="font-semibold text-fg">Simulations at once</td>
                <td className="tabular text-fg-muted">1 per browser</td>
                <td className="text-fg-muted">Starting in a new tab stops the old one</td>
              </tr>
            </TBody>
          </TableWrap>
          <p className="mt-4 text-sm text-fg-muted">
            A run counts when it starts. If it fails because of a server error before finishing its
            first generation, it&apos;s given back.
          </p>
        </div>
      </Card>
      <Card>
        <CardHeader
          title="Your Trial ID"
          description="Anonymous. Share it only if you need help with your runs."
          icon={<Database />}
        />
        <div className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center">
          <code className="min-w-0 flex-1 truncate rounded-xl border border-border bg-surface-2 px-4 py-3 font-mono text-sm text-fg">
            {trialId ?? "Loading…"}
          </code>
          <Button
            variant="outline"
            disabled={!trialId}
            onClick={async () => {
              if (!trialId) return
              try {
                await navigator.clipboard.writeText(trialId)
                setCopied(true)
                setTimeout(() => setCopied(false), 1500)
              } catch {}
            }}
          >
            {copied ? <Check className="size-4 text-success" /> : <Copy className="size-4" />}
            {copied ? "Copied" : "Copy ID"}
          </Button>
        </div>
      </Card>
    </>
  )
}

function DataSection() {
  const toast = useToast()
  const router = useRouter()
  const [exporting, setExporting] = useState(false)
  const [confirmHistory, setConfirmHistory] = useState(false)
  const [confirmEverything, setConfirmEverything] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function exportData() {
    setExporting(true)
    try {
      await downloadJson("/api/me/export", "self-driven-car-export.json")
      toast.success("Export Ready", "Your runs were downloaded as a JSON file.")
    } catch (err) {
      toast.error("We Couldn't Export Your Data", errorMessage(err))
    } finally {
      setExporting(false)
    }
  }

  return (
    <>
      <Card>
        <CardHeader
          title="Export Your Data"
          description="Download every run, generation, and champion network as JSON."
          icon={<Download />}
        />
        <div className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-sm text-fg-muted">
            Useful for backups or analysing your runs in another tool.
          </p>
          <Button onClick={exportData} loading={exporting} loadingText="Preparing">
            <Download className="size-4" /> Export Data
          </Button>
        </div>
      </Card>
      <Card className="border-danger/40">
        <CardHeader
          title="Danger Zone"
          description="These actions can't be undone."
          icon={<TriangleAlert />}
          className="[&_div>div:first-child]:bg-danger-soft [&_div>div:first-child]:text-danger"
        />
        <div className="divide-y divide-border">
          <div className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="font-semibold text-fg">Delete Run History</p>
              <p className="mt-1 text-sm text-fg-muted">
                Remove all saved runs. A run that&apos;s training right now is kept.
              </p>
            </div>
            <Button variant="danger-outline" onClick={() => setConfirmHistory(true)}>
              <Trash2 className="size-4" /> Delete History
            </Button>
          </div>
          <div className="flex flex-col gap-3 p-5 sm:flex-row sm:items-center sm:justify-between">
            <div>
              <p className="font-semibold text-fg">Delete Everything</p>
              <p className="mt-1 text-sm text-fg-muted">
                Remove your runs and trial ID, and reset this browser&apos;s settings. You&apos;ll
                start fresh with a new trial ID.
              </p>
            </div>
            <Button variant="danger" onClick={() => setConfirmEverything(true)}>
              <Trash2 className="size-4" /> Delete Everything
            </Button>
          </div>
        </div>
      </Card>
      <ConfirmDialog
        open={confirmHistory}
        onClose={() => {
          setConfirmHistory(false)
          setError(null)
        }}
        title="Delete All Runs?"
        description="Every saved run, with all of its generations and champion networks, will be permanently deleted. Your free-run count isn't reset."
        confirmLabel="Delete All Runs"
        error={error}
        onConfirm={async () => {
          try {
            const result = await api.delete<{ deleted: number }>("/api/me/runs")
            setConfirmHistory(false)
            setError(null)
            toast.success(
              "Run History Deleted",
              `${result.deleted} run${result.deleted === 1 ? " was" : "s were"} removed.`,
            )
          } catch (err) {
            setError(errorMessage(err))
          }
        }}
      />
      <ConfirmDialog
        open={confirmEverything}
        onClose={() => {
          setConfirmEverything(false)
          setError(null)
        }}
        title="Delete Everything?"
        description={
          <>
            This permanently deletes your runs and trial ID, stops any run in progress, and resets
            this browser&apos;s settings. Usage records stay until they expire so the free limits
            stay fair.
          </>
        }
        requireText="DELETE"
        confirmLabel="Delete Everything"
        error={error}
        onConfirm={async () => {
          try {
            await api.delete("/api/me")
            resetVisitor()
            resetPreferences()
            clearUsage()
            setConfirmEverything(false)
            toast.success("Everything Was Deleted", "You're starting fresh with a new trial ID.")
            router.push("/")
          } catch (err) {
            setError(errorMessage(err))
          }
        }}
      />
    </>
  )
}
