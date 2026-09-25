import type { Metadata } from "next"
import { Suspense } from "react"
import { SettingsView } from "@/components/settings/settings-view"

export const metadata: Metadata = {
  title: "Settings",
  description: "Theme, simulator defaults, free usage, and data controls.",
  robots: { index: false, follow: false },
}

export default function SettingsPage() {
  return (
    <Suspense>
      <SettingsView />
    </Suspense>
  )
}
