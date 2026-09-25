import type { MetadataRoute } from "next"
import { site } from "@/lib/site"

export default function manifest(): MetadataRoute.Manifest {
  return {
    name: site.name,
    short_name: site.shortName,
    description: site.description,
    start_url: "/simulator",
    display: "standalone",
    background_color: "#000000",
    theme_color: "#000000",
    categories: ["education", "science"],
    icons: [{ src: "/icon.svg", sizes: "any", type: "image/svg+xml" }],
  }
}
