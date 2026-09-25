"use client"

import { RotateCcw, TriangleAlert } from "lucide-react"
import { Button, ButtonLink } from "@/components/ui/button"

export default function Error({
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <main
      id="main"
      className="flex min-h-[70dvh] flex-col items-center justify-center bg-bg px-4 py-16 text-center"
    >
      <span className="flex size-16 items-center justify-center rounded-2xl bg-danger-soft text-danger">
        <TriangleAlert className="size-8" />
      </span>
      <h1 className="mt-6 text-4xl font-bold tracking-tight text-fg">Something Went Wrong</h1>
      <p className="mt-4 max-w-md text-lg text-fg-muted">
        This page hit an unexpected problem. Your runs are safe on the server. Try loading the page
        again.
      </p>
      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <Button onClick={reset}>
          <RotateCcw className="size-4" /> Try Again
        </Button>
        <ButtonLink href="/" variant="outline">
          Go to the Home Page
        </ButtonLink>
      </div>
    </main>
  )
}
