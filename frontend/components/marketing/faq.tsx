import { ChevronDown } from "lucide-react"
import { FAQ } from "@/lib/content"

export function FaqList() {
  return (
    <div className="mx-auto mt-12 max-w-3xl divide-y divide-border overflow-hidden rounded-3xl border border-border bg-surface">
      {FAQ.map((item) => (
        <details key={item.question} className="group">
          <summary className="flex cursor-pointer list-none items-center justify-between gap-4 px-6 py-5 text-left font-semibold text-fg transition-colors hover:bg-surface-2 [&::-webkit-details-marker]:hidden">
            <span className="text-base sm:text-lg">{item.question}</span>
            <ChevronDown
              aria-hidden
              className="size-5 shrink-0 text-fg-subtle transition-transform group-open:rotate-180"
            />
          </summary>
          <p className="px-6 pb-6 leading-relaxed text-fg-muted">{item.answer}</p>
        </details>
      ))}
    </div>
  )
}
