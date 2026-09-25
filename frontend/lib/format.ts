const numberFormat = new Intl.NumberFormat("en-US")
const compactFormat = new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
})
const dateFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
})
const dateTimeFormat = new Intl.DateTimeFormat("en-US", {
  month: "short",
  day: "numeric",
  year: "numeric",
  hour: "numeric",
  minute: "2-digit",
})

export function formatNumber(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—"
  return digits === 0
    ? numberFormat.format(Math.round(value))
    : value.toLocaleString("en-US", {
        minimumFractionDigits: digits,
        maximumFractionDigits: digits,
      })
}

export function formatCompact(value: number | null | undefined): string {
  if (value === null || value === undefined) return "—"
  return compactFormat.format(value)
}

export function formatFitness(value: number | null | undefined): string {
  return formatNumber(value, 1)
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—"
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

export function formatDate(value: string | Date): string {
  return dateFormat.format(new Date(value))
}

export function formatDateTime(value: string | Date): string {
  return dateTimeFormat.format(new Date(value))
}

export function formatRelative(value: string | Date): string {
  const diff = (Date.now() - new Date(value).getTime()) / 1000
  if (diff < 45) return "Just now"
  if (diff < 3600) return `${Math.round(diff / 60)} min ago`
  if (diff < 86400) return `${Math.round(diff / 3600)} hr ago`
  if (diff < 86400 * 7) {
    const days = Math.round(diff / 86400)
    return `${days} day${days === 1 ? "" : "s"} ago`
  }
  return formatDate(value)
}

export function describeUserAgent(ua: string | null): {
  device: string
  browser: string
  kind: "mobile" | "tablet" | "desktop"
} {
  if (!ua) return { device: "Unknown Device", browser: "Unknown Browser", kind: "desktop" }
  const kind = /iPad|Tablet/i.test(ua)
    ? "tablet"
    : /Mobi|iPhone|Android/i.test(ua)
      ? "mobile"
      : "desktop"
  const device = /iPhone/i.test(ua)
    ? "iPhone"
    : /iPad/i.test(ua)
      ? "iPad"
      : /Android/i.test(ua)
        ? "Android"
        : /Mac OS X|Macintosh/i.test(ua)
          ? "Mac"
          : /Windows/i.test(ua)
            ? "Windows PC"
            : /Linux/i.test(ua)
              ? "Linux"
              : "Unknown Device"
  const browser = /Edg\//.test(ua)
    ? "Edge"
    : /OPR\//.test(ua)
      ? "Opera"
      : /Firefox\//.test(ua)
        ? "Firefox"
        : /Chrome\//.test(ua)
          ? "Chrome"
          : /Safari\//.test(ua)
            ? "Safari"
            : "Browser"
  return { device, browser, kind }
}

export function titleCase(value: string): string {
  const minor = new Set([
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "vs",
  ])
  return value
    .split(/\s+/)
    .map((word, i) => {
      const lower = word.toLowerCase()
      if (i > 0 && minor.has(lower)) return lower
      return lower.charAt(0).toUpperCase() + lower.slice(1)
    })
    .join(" ")
}
