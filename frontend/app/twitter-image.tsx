import { ImageResponse } from "next/og"
import { site } from "@/lib/site"

export const alt = `${site.name}: ${site.tagline}`
export const size = { width: 1200, height: 630 }
export const contentType = "image/png"

export default function TwitterImage() {
  return new ImageResponse(
    <div
      style={{
        width: "100%",
        height: "100%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "space-between",
        padding: 72,
        background: "linear-gradient(135deg, #000000 0%, #06222a 100%)",
        color: "#f2f5f8",
        fontFamily: "sans-serif",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 20 }}>
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: 18,
            background: "#22d3ee",
            display: "flex",
          }}
        />
        <div style={{ fontSize: 36, fontWeight: 700 }}>{site.name}</div>
      </div>
      <div style={{ display: "flex", flexDirection: "column" }}>
        <div style={{ fontSize: 76, fontWeight: 800, lineHeight: 1.05 }}>
          Watch a Neural Network
        </div>
        <div style={{ fontSize: 76, fontWeight: 800, lineHeight: 1.05, color: "#22d3ee" }}>
          Learn to Drive
        </div>
        <div style={{ marginTop: 28, fontSize: 30, color: "#a2adb9" }}>
          Live neuroevolution with NEAT · Real code, step by step · Free, no sign-up
        </div>
      </div>
    </div>,
    size,
  )
}
