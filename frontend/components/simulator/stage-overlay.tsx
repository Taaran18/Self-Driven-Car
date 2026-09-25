import { CircleAlert, Flag, LoaderCircle, MousePointerClick, Trophy } from "lucide-react"
import Link from "next/link"
import { describeReason } from "@/components/ui/badge"
import { Button, ButtonLink } from "@/components/ui/button"
import { formatDateTime, formatDuration, formatFitness } from "@/lib/format"
import type { SimSnapshot } from "@/lib/simulation/controller"
import { useUsage } from "@/lib/usage"

function Panel({ children }: { children: React.ReactNode }) {
  return (
    <div className="absolute inset-0 z-10 flex items-center justify-center bg-bg/55 p-4 backdrop-blur-[3px]">
      <div className="w-full max-w-md animate-dialog-in rounded-2xl border border-border bg-surface p-6 text-center shadow-[var(--shadow-pop)]">
        {children}
      </div>
    </div>
  )
}

export function StageOverlay({
  snapshot,
  onRetry,
  onNewRun,
}: {
  snapshot: SimSnapshot
  onRetry: () => void
  onNewRun: () => void
}) {
  const usage = useUsage()
  const { phase } = snapshot

  if (phase === "idle") {
    return (
      <Panel>
        <span className="mx-auto flex size-12 items-center justify-center rounded-2xl bg-primary-soft text-primary-soft-fg">
          <MousePointerClick className="size-6" />
        </span>
        <h2 className="mt-4 text-2xl font-bold text-fg">Ready When You Are</h2>
        <p className="mt-2 text-sm leading-relaxed text-fg-muted">
          Choose your settings, then press <strong className="text-fg">Start Training</strong>. The
          first generation drives almost randomly. Watch how quickly that changes.
        </p>
      </Panel>
    )
  }

  if (phase === "connecting" || phase === "waking" || phase === "starting") {
    const title =
      phase === "waking"
        ? "Waking Up the Simulation Server"
        : phase === "starting"
          ? "Building the First Generation"
          : "Connecting"
    const text =
      phase === "waking"
        ? "The server sleeps when nobody is using it, to keep costs down. It usually wakes in 5 to 30 seconds."
        : phase === "starting"
          ? "Creating random neural networks and laying down a fresh track."
          : "Reserving a simulation slot for you."
    return (
      <Panel>
        <LoaderCircle className="mx-auto size-10 animate-spin text-primary" aria-hidden />
        <h2 className="mt-4 text-xl font-bold text-fg" role="status">
          {title}
        </h2>
        <p className="mt-2 text-sm leading-relaxed text-fg-muted">{text}</p>
      </Panel>
    )
  }

  if (phase === "error" && snapshot.error) {
    const { code, message } = snapshot.error
    const title =
      code === "trial_limit"
        ? "You've Used Your Free Runs"
        : code === "capacity"
          ? "All Simulation Slots Are Busy"
          : code === "connection_lost"
            ? "Connection Lost"
            : "We Couldn't Start the Run"
    return (
      <Panel>
        <span className="mx-auto flex size-12 items-center justify-center rounded-2xl bg-danger-soft text-danger">
          <CircleAlert className="size-6" />
        </span>
        <h2 className="mt-4 text-xl font-bold text-fg">{title}</h2>
        <p className="mt-2 text-sm leading-relaxed text-fg-muted">{message}</p>
        {code === "trial_limit" && usage.day_resets_at ? (
          <p className="mt-2 text-sm font-medium text-fg">
            Next run available {formatDateTime(usage.day_resets_at)}.
          </p>
        ) : null}
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {code === "trial_limit" ? (
            <>
              <ButtonLink href="/runs" variant="outline">
                Review Past Runs
              </ButtonLink>
              <ButtonLink href="/how-it-works">Read How It Works</ButtonLink>
            </>
          ) : (
            <Button onClick={onRetry}>Try Again</Button>
          )}
        </div>
      </Panel>
    )
  }

  if (phase === "ended" && snapshot.ended) {
    const ended = snapshot.ended
    const solved = ended.reason === "solved"
    const title = solved
      ? "A Car Reached the Finish Line"
      : ended.status === "completed"
        ? "Training Complete"
        : ended.status === "stopped"
          ? "Run Stopped"
          : ended.status === "failed"
            ? "The Run Hit an Error"
            : "Run Interrupted"
    return (
      <Panel>
        <span
          className={`mx-auto flex size-12 items-center justify-center rounded-2xl ${solved ? "bg-success-soft text-success" : "bg-primary-soft text-primary-soft-fg"}`}
        >
          {solved ? <Trophy className="size-6" /> : <Flag className="size-6" />}
        </span>
        <h2 className="mt-4 text-2xl font-bold text-fg">{title}</h2>
        <p className="mt-2 text-sm text-fg-muted">{describeReason(ended.reason)}.</p>
        <dl className="mt-5 grid grid-cols-3 gap-2 rounded-xl bg-surface-2 p-3 text-left">
          <div>
            <dt className="text-xs text-fg-subtle">Generations</dt>
            <dd className="tabular text-lg font-bold text-fg">{ended.generations_completed}</dd>
          </div>
          <div>
            <dt className="text-xs text-fg-subtle">Best Fitness</dt>
            <dd className="tabular text-lg font-bold text-fg">
              {formatFitness(ended.best_fitness)}
            </dd>
          </div>
          <div>
            <dt className="text-xs text-fg-subtle">Time</dt>
            <dd className="tabular text-lg font-bold text-fg">
              {formatDuration(ended.elapsed_seconds)}
            </dd>
          </div>
        </dl>
        {ended.refunded ? (
          <p className="mt-3 text-xs text-fg-subtle">
            This run didn&apos;t count toward your daily limit.
          </p>
        ) : null}
        <div className="mt-5 flex flex-wrap justify-center gap-2">
          {ended.run_id ? (
            <ButtonLink href={`/runs/${ended.run_id}`} variant="outline">
              View Run Details
            </ButtonLink>
          ) : null}
          <Button onClick={onNewRun}>Start Another Run</Button>
        </div>
        <p className="mt-4 text-xs text-fg-subtle">
          {usage.live
            ? `${usage.left_today} of ${usage.runs_per_day} free runs left today. `
            : null}
          <Link href="/settings?tab=usage" className="font-semibold text-primary hover:underline">
            See Usage
          </Link>
        </p>
      </Panel>
    )
  }

  return null
}
