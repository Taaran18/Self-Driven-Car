import type { MetadataRoute } from "next"
import { site } from "@/lib/site"

const pages: {
  path: string
  priority: number
  changeFrequency: "weekly" | "monthly" | "yearly"
}[] = [
  { path: "", priority: 1, changeFrequency: "weekly" },
  { path: "/simulator", priority: 0.9, changeFrequency: "weekly" },
  { path: "/how-it-works", priority: 0.8, changeFrequency: "monthly" },
  { path: "/privacy", priority: 0.3, changeFrequency: "yearly" },
  { path: "/terms", priority: 0.3, changeFrequency: "yearly" },
  { path: "/disclaimer", priority: 0.3, changeFrequency: "yearly" },
]

export default function sitemap(): MetadataRoute.Sitemap {
  const lastModified = new Date("2026-09-25")
  return pages.map((page) => ({
    url: `${site.url}${page.path}`,
    lastModified,
    changeFrequency: page.changeFrequency,
    priority: page.priority,
  }))
}
