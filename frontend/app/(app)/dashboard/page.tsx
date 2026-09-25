import type { Metadata } from "next"
import { DashboardView } from "@/components/runs/dashboard-view"

export const metadata: Metadata = {
  title: "Dashboard",
  description: "Your training runs, best fitness, and remaining free runs at a glance.",
  robots: { index: false, follow: false },
}

export default function DashboardPage() {
  return <DashboardView />
}
