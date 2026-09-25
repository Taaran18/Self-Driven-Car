"use client"

import { Pencil, Trash2 } from "lucide-react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { ConfirmDialog, Dialog } from "@/components/ui/dialog"
import { Field, Input } from "@/components/ui/field"
import { useToast } from "@/components/ui/toast"
import { api, errorMessage } from "@/lib/api"
import type { RunSummary } from "@/lib/types"

export function RenameRunDialog({
  run,
  onClose,
  onRenamed,
}: {
  run: RunSummary | null
  onClose: () => void
  onRenamed: (run: RunSummary) => void
}) {
  const toast = useToast()
  const [name, setName] = useState("")
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [lastId, setLastId] = useState<string | null>(null)

  if (run && run.id !== lastId) {
    setLastId(run.id)
    setName(run.name)
    setError(null)
  }

  async function save() {
    if (!run) return
    const trimmed = name.trim()
    if (!trimmed) {
      setError("Enter a name for this run.")
      return
    }
    if (trimmed === run.name) {
      onClose()
      return
    }
    setBusy(true)
    try {
      const updated = await api.patch<RunSummary>(`/api/runs/${run.id}`, { name: trimmed })
      onRenamed(updated)
      toast.success("Run Renamed", `Saved as “${updated.name}”.`)
      onClose()
    } catch (err) {
      setError(errorMessage(err))
    } finally {
      setBusy(false)
    }
  }

  return (
    <Dialog
      open={Boolean(run)}
      onClose={() => !busy && onClose()}
      dismissible={!busy}
      title="Rename Run"
      description="Give this run a name you'll recognize later."
      icon={<Pencil className="size-5" />}
      size="sm"
      footer={
        <>
          <Button variant="outline" onClick={onClose} disabled={busy}>
            Cancel
          </Button>
          <Button onClick={save} loading={busy} loadingText="Saving">
            Save Name
          </Button>
        </>
      }
    >
      <form
        onSubmit={(e) => {
          e.preventDefault()
          void save()
        }}
      >
        <Field label="Run Name" error={error ?? undefined}>
          {({ id, describedBy, invalid }) => (
            <Input
              id={id}
              aria-describedby={describedBy}
              invalid={invalid}
              value={name}
              onChange={(e) => {
                setName(e.target.value)
                setError(null)
              }}
              maxLength={80}
              data-autofocus
            />
          )}
        </Field>
      </form>
    </Dialog>
  )
}

export function DeleteRunDialog({
  run,
  onClose,
  onDeleted,
}: {
  run: RunSummary | null
  onClose: () => void
  onDeleted: (run: RunSummary) => void
}) {
  const toast = useToast()
  const [error, setError] = useState<string | null>(null)

  return (
    <ConfirmDialog
      open={Boolean(run)}
      onClose={() => {
        setError(null)
        onClose()
      }}
      title="Delete This Run?"
      icon={<Trash2 className="size-5" />}
      description={
        <>
          <strong className="text-fg">{run?.name}</strong> and all {run?.generations_completed ?? 0}{" "}
          of its generations will be permanently deleted. This can&apos;t be undone, and it
          doesn&apos;t give back a free run.
        </>
      }
      confirmLabel="Delete Run"
      error={error}
      onConfirm={async () => {
        if (!run) return
        try {
          await api.delete(`/api/runs/${run.id}`)
          onDeleted(run)
          toast.success("Run Deleted", `“${run.name}” was removed from your history.`)
          setError(null)
          onClose()
        } catch (err) {
          setError(errorMessage(err))
        }
      }}
    />
  )
}
