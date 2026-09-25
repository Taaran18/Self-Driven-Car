"use client"

import { Activity, Globe, KeyRound, LogOut, RefreshCw, Save, Users } from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardHeader, StatCard } from "@/components/ui/card"
import { EmptyState, InlineAlert, Skeleton } from "@/components/ui/feedback"
import { Field, PasswordInput } from "@/components/ui/field"
import { PageHeader } from "@/components/ui/page-header"
import { Select } from "@/components/ui/select"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { api, errorMessage } from "@/lib/api"
import { describeUserAgent, formatDateTime, formatNumber, formatRelative } from "@/lib/format"

interface Report {
  generated_at: string
  window_days: number
  limits: { runs_per_day: number; runs_per_week: number }
  totals: {
    runs: number
    visitors: number
    ips: number
    runs_saved: number
    visitors_all_time: number
  }
  simulations: { active: number; max: number }
  ips: {
    ip: string
    runs: number
    runs_today: number
    runs_this_week: number
    visitors: number
    first_run_at: string | null
    last_run_at: string | null
  }[]
  visitors: {
    visitor_id: string
    runs: number
    runs_today: number
    runs_this_week: number
    last_run_at: string | null
    ip: string
    user_agent: string | null
  }[]
}

const TOKEN_KEY = "sdc.admin"
const windows = [
  { value: "1", label: "Last 24 Hours" },
  { value: "7", label: "Last 7 Days" },
  { value: "30", label: "Last 30 Days" },
  { value: "90", label: "Last 90 Days" },
]

function readToken() {
  try {
    return sessionStorage.getItem(TOKEN_KEY)
  } catch {
    return null
  }
}

export function AdminView() {
  const [token, setToken] = useState<string | null>(null)
  const [input, setInput] = useState("")
  const [days, setDays] = useState("7")
  const [report, setReport] = useState<Report | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)

  const load = useCallback(async (value: string, window: string) => {
    setLoading(true)
    setError(null)
    try {
      const data = await api.get<Report>(`/api/admin/usage?days=${window}`, {
        headers: { Authorization: `Bearer ${value}` },
        retries: 3,
      })
      setReport(data)
      setToken(value)
      try {
        sessionStorage.setItem(TOKEN_KEY, value)
      } catch {}
    } catch (err) {
      setError(errorMessage(err))
      setReport(null)
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    const saved = readToken()
    if (saved) {
      const frame = requestAnimationFrame(() => void load(saved, "7"))
      return () => cancelAnimationFrame(frame)
    }
  }, [load])

  function signOut() {
    try {
      sessionStorage.removeItem(TOKEN_KEY)
    } catch {}
    setToken(null)
    setReport(null)
    setInput("")
  }

  return (
    <div>
      <section className="bg-bg px-4 pt-10 pb-10 sm:px-6 lg:px-8">
        <PageHeader
          eyebrow="Admin"
          title="Usage Report"
          description="Who is using the simulator, from which networks, and how much."
        />
      </section>
      <section className="border-t border-border bg-bg-alt px-4 py-10 sm:px-6 lg:px-8">
        <div className="mx-auto max-w-[1600px] space-y-6">
          {!token ? (
            <Card className="mx-auto max-w-md">
              <CardHeader
                title="Enter Admin Token"
                description="The ADMIN_TOKEN value from your backend environment."
                icon={<KeyRound />}
              />
              <form
                className="space-y-4 p-5"
                onSubmit={(e) => {
                  e.preventDefault()
                  if (input.trim()) void load(input.trim(), days)
                }}
              >
                {error ? <InlineAlert tone="danger">{error}</InlineAlert> : null}
                <Field label="Admin Token">
                  {({ id }) => (
                    <PasswordInput
                      id={id}
                      value={input}
                      onChange={(e) => setInput(e.target.value)}
                      autoComplete="off"
                    />
                  )}
                </Field>
                <Button
                  type="submit"
                  className="w-full"
                  loading={loading}
                  loadingText="Checking"
                  disabled={!input.trim()}
                >
                  <Save className="size-4" /> Open Report
                </Button>
                <p className="text-xs text-fg-subtle">
                  The token is kept only for this browser tab.
                </p>
              </form>
            </Card>
          ) : (
            <>
              <Card className="flex flex-wrap items-center justify-between gap-3 p-4">
                <div className="w-52">
                  <Select
                    value={days}
                    onChange={(v) => {
                      setDays(v)
                      void load(token, v)
                    }}
                    options={windows}
                    label="Report window"
                  />
                </div>
                <div className="flex items-center gap-2">
                  {report ? (
                    <span className="text-xs text-fg-subtle">
                      Updated {formatRelative(report.generated_at)}
                    </span>
                  ) : null}
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => void load(token, days)}
                    loading={loading}
                    loadingText="Refreshing"
                  >
                    <RefreshCw className="size-4" /> Refresh
                  </Button>
                  <Button variant="ghost" size="sm" onClick={signOut}>
                    <LogOut className="size-4" /> Lock
                  </Button>
                </div>
              </Card>
              {error ? <InlineAlert tone="danger">{error}</InlineAlert> : null}
              {!report ? (
                <Skeleton className="h-64 rounded-2xl" />
              ) : (
                <>
                  <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
                    <StatCard
                      label="Runs Started"
                      value={formatNumber(report.totals.runs)}
                      hint={`In the last ${report.window_days} day${report.window_days === 1 ? "" : "s"}`}
                      icon={<Activity />}
                    />
                    <StatCard
                      label="Visitors"
                      value={formatNumber(report.totals.visitors)}
                      hint={`${formatNumber(report.totals.visitors_all_time)} all time`}
                      icon={<Users />}
                      tone="violet"
                    />
                    <StatCard
                      label="Networks (IPs)"
                      value={formatNumber(report.totals.ips)}
                      hint={`Limit ${report.limits.runs_per_day}/day, ${report.limits.runs_per_week}/week`}
                      icon={<Globe />}
                      tone="warning"
                    />
                    <StatCard
                      label="Running Now"
                      value={`${report.simulations.active} / ${report.simulations.max}`}
                      hint={`${formatNumber(report.totals.runs_saved)} runs saved`}
                      icon={<Activity />}
                      tone="success"
                    />
                  </div>
                  <div className="space-y-3">
                    <h2 className="text-xl font-bold text-fg">By Network (IP Address)</h2>
                    {report.ips.length ? (
                      <TableWrap label="Usage by IP address">
                        <THead>
                          <tr>
                            <th scope="col">IP Address</th>
                            <th scope="col" className="text-right">
                              Today
                            </th>
                            <th scope="col" className="text-right">
                              This Week
                            </th>
                            <th scope="col" className="text-right">
                              In Window
                            </th>
                            <th scope="col" className="text-right">
                              Visitors
                            </th>
                            <th scope="col" className="text-right">
                              Last Run
                            </th>
                          </tr>
                        </THead>
                        <TBody>
                          {report.ips.map((row) => (
                            <tr
                              key={row.ip}
                              className={
                                row.runs_today >= report.limits.runs_per_day
                                  ? "!bg-warning-soft"
                                  : undefined
                              }
                            >
                              <td className="font-mono text-sm text-fg">{row.ip}</td>
                              <td className="tabular text-right font-semibold text-fg">
                                {row.runs_today} / {report.limits.runs_per_day}
                              </td>
                              <td className="tabular text-right text-fg-muted">
                                {row.runs_this_week} / {report.limits.runs_per_week}
                              </td>
                              <td className="tabular text-right text-fg-muted">{row.runs}</td>
                              <td className="tabular text-right text-fg-muted">{row.visitors}</td>
                              <td
                                className="text-right text-fg-muted"
                                title={
                                  row.last_run_at ? formatDateTime(row.last_run_at) : undefined
                                }
                              >
                                {row.last_run_at ? formatRelative(row.last_run_at) : "—"}
                              </td>
                            </tr>
                          ))}
                        </TBody>
                      </TableWrap>
                    ) : (
                      <Card>
                        <EmptyState
                          icon={<Globe />}
                          title="No Runs in This Window"
                          description="Nobody has started a run in the selected period."
                        />
                      </Card>
                    )}
                  </div>
                  <div className="space-y-3">
                    <h2 className="text-xl font-bold text-fg">By Visitor</h2>
                    {report.visitors.length ? (
                      <TableWrap label="Usage by visitor">
                        <THead>
                          <tr>
                            <th scope="col">Trial ID</th>
                            <th scope="col">IP Address</th>
                            <th scope="col">Device</th>
                            <th scope="col" className="text-right">
                              Today
                            </th>
                            <th scope="col" className="text-right">
                              This Week
                            </th>
                            <th scope="col" className="text-right">
                              In Window
                            </th>
                            <th scope="col" className="text-right">
                              Last Run
                            </th>
                          </tr>
                        </THead>
                        <TBody>
                          {report.visitors.map((row) => {
                            const ua = describeUserAgent(row.user_agent)
                            return (
                              <tr key={row.visitor_id}>
                                <td className="font-mono text-xs text-fg" title={row.visitor_id}>
                                  {row.visitor_id.slice(0, 8)}…
                                </td>
                                <td className="font-mono text-sm text-fg-muted">{row.ip}</td>
                                <td className="text-fg-muted">
                                  {ua.browser} on {ua.device}
                                </td>
                                <td className="tabular text-right font-semibold text-fg">
                                  {row.runs_today}
                                </td>
                                <td className="tabular text-right text-fg-muted">
                                  {row.runs_this_week}
                                </td>
                                <td className="tabular text-right text-fg-muted">{row.runs}</td>
                                <td className="text-right text-fg-muted">
                                  {row.last_run_at ? formatRelative(row.last_run_at) : "—"}
                                </td>
                              </tr>
                            )
                          })}
                        </TBody>
                      </TableWrap>
                    ) : null}
                  </div>
                </>
              )}
            </>
          )}
        </div>
      </section>
    </div>
  )
}
