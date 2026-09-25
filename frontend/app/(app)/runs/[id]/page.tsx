import type { Metadata } from "next"
import { RunDetailView } from "@/components/runs/run-detail-view"

export const metadata: Metadata = {
  title: "Run Details",
  robots: { index: false, follow: false },
}

export default async function RunDetailPage(props: PageProps<"/runs/[id]">) {
  const { id } = await props.params
  return <RunDetailView id={id} />
}
