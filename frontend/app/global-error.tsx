"use client"

export default function GlobalError({
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <html lang="en">
      <body
        style={{
          margin: 0,
          fontFamily: "system-ui, sans-serif",
          background: "#000",
          color: "#f2f5f8",
        }}
      >
        <main
          style={{
            minHeight: "100vh",
            display: "grid",
            placeItems: "center",
            padding: 24,
            textAlign: "center",
          }}
        >
          <div>
            <h1 style={{ fontSize: 36, margin: 0 }}>Something Went Wrong</h1>
            <p style={{ color: "#a2adb9", maxWidth: 420 }}>
              The app couldn&apos;t load. Try again in a moment.
            </p>
            <button
              onClick={reset}
              style={{
                marginTop: 16,
                padding: "12px 20px",
                borderRadius: 12,
                border: 0,
                background: "#22d3ee",
                color: "#00161c",
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              Try Again
            </button>
          </div>
        </main>
      </body>
    </html>
  )
}
