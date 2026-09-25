import type { Metadata } from "next"
import { RunsView } from "@/components/runs/runs-view"

export const metadata: Metadata = {
  title: "Training Runs",
  description: "Search, rename, and review every training run you've started.",
  robots: { index: false, follow: false },
}

export default function RunsPage() {
  return <RunsView />
}
