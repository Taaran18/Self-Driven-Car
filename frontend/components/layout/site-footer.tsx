import Link from "next/link"
import { GithubIcon } from "@/components/icons/github"
import { cn } from "@/lib/cn"
import { legalLinks, site } from "@/lib/site"
import { Logo } from "./logo"

const columns = [
  {
    title: "Product",
    links: [
      { href: "/simulator", label: "Simulator" },
      { href: "/dashboard", label: "Dashboard" },
      { href: "/runs", label: "Training Runs" },
      { href: "/settings", label: "Settings" },
    ],
  },
  {
    title: "Learn",
    links: [
      { href: "/#how-it-learns", label: "How It Learns" },
      { href: "/how-it-works", label: "The Deep Dive" },
      { href: "/how-it-works#glossary", label: "Glossary" },
      { href: "/#faq", label: "FAQ" },
    ],
  },
  { title: "Legal", links: legalLinks },
]

export function SiteFooter({ compact, className }: { compact?: boolean; className?: string }) {
  const year = new Date().getFullYear()
  if (compact) {
    return (
      <footer className={cn("border-t border-border bg-bg-alt", className)}>
        <div className="mx-auto flex max-w-[1600px] flex-col items-center justify-between gap-4 px-4 py-6 sm:flex-row sm:px-6 lg:px-8">
          <div className="flex items-center gap-3">
            <Logo compact />
            <p className="text-sm text-fg-muted">
              © {year} {site.name}
            </p>
          </div>
          <nav aria-label="Legal" className="flex flex-wrap justify-center gap-x-5 gap-y-2">
            {legalLinks.map((link) => (
              <Link
                key={link.href}
                href={link.href}
                className="text-sm text-fg-muted hover:text-fg"
              >
                {link.label}
              </Link>
            ))}
          </nav>
        </div>
      </footer>
    )
  }
  return (
    <footer className={cn("border-t border-border bg-bg-alt", className)}>
      <div className="mx-auto grid max-w-[1440px] gap-12 px-4 py-16 sm:px-6 lg:grid-cols-[1.4fr_2fr] lg:px-10">
        <div className="max-w-sm">
          <Logo />
          <p className="mt-4 text-sm leading-relaxed text-fg-muted">
            An open learning lab that shows, step by step, how neural networks evolve to drive a car
            on their own.
          </p>
          <a
            href={site.githubUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-6 inline-flex items-center gap-2 rounded-xl border border-border bg-surface px-3.5 py-2 text-sm font-semibold text-fg transition hover:border-border-strong"
          >
            <GithubIcon className="size-4" />
            View the Source Code
          </a>
        </div>
        <div className="grid grid-cols-2 gap-8 sm:grid-cols-3">
          {columns.map((column) => (
            <nav key={column.title} aria-label={column.title}>
              <h2 className="text-sm font-bold text-fg">{column.title}</h2>
              <ul className="mt-4 space-y-3">
                {column.links.map((link) => (
                  <li key={link.href}>
                    <Link
                      href={link.href}
                      className="text-sm text-fg-muted transition-colors hover:text-fg"
                    >
                      {link.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </nav>
          ))}
        </div>
      </div>
      <div className="border-t border-border">
        <div className="mx-auto flex max-w-[1440px] flex-col items-center justify-between gap-3 px-4 py-6 text-sm text-fg-subtle sm:flex-row sm:px-6 lg:px-10">
          <p>
            © {year} {site.name}. Built for learning.
          </p>
          <p>Made with Next.js, FastAPI, and NEAT-Python.</p>
        </div>
      </div>
    </footer>
  )
}
