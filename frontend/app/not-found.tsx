import { ArrowRight, Signpost } from "lucide-react"
import type { Metadata } from "next"
import { Logo } from "@/components/layout/logo"
import { ButtonLink } from "@/components/ui/button"

export const metadata: Metadata = {
  title: "Page Not Found",
  robots: { index: false, follow: false },
}

export default function NotFound() {
  return (
    <main
      id="main"
      className="flex min-h-dvh flex-col items-center justify-center bg-bg px-4 py-16 text-center"
    >
      <Logo />
      <span className="mt-12 flex size-16 items-center justify-center rounded-2xl bg-warning-soft text-warning">
        <Signpost className="size-8" />
      </span>
      <p className="mt-6 font-mono text-sm font-semibold text-fg-subtle">Error 404</p>
      <h1 className="mt-2 text-4xl font-bold tracking-tight text-fg sm:text-5xl">
        This Road Doesn&apos;t Exist
      </h1>
      <p className="mt-4 max-w-md text-lg text-fg-muted">
        The page you&apos;re looking for may have moved, or the link might be mistyped.
      </p>
      <div className="mt-8 flex flex-col gap-3 sm:flex-row">
        <ButtonLink href="/simulator">
          Open the Simulator <ArrowRight className="size-4" />
        </ButtonLink>
        <ButtonLink href="/" variant="outline">
          Go to the Home Page
        </ButtonLink>
      </div>
    </main>
  )
}
