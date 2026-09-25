import Link from "next/link"
import { JsonLd } from "@/components/seo/json-ld"
import { PageHeader } from "@/components/ui/page-header"
import { breadcrumbSchema } from "@/lib/schema"
import { legalLinks, site } from "@/lib/site"

export interface LegalSection {
  id: string
  title: string
  body: React.ReactNode
}

export function ContactLine() {
  return site.contactEmail ? (
    <>
      email{" "}
      <a
        href={`mailto:${site.contactEmail}`}
        className="font-semibold text-primary hover:underline"
      >
        {site.contactEmail}
      </a>
    </>
  ) : (
    <>
      open an issue on the{" "}
      <a
        href={`${site.githubUrl}/issues`}
        target="_blank"
        rel="noopener noreferrer"
        className="font-semibold text-primary hover:underline"
      >
        project&apos;s GitHub repository
      </a>
    </>
  )
}

export function LegalPage({
  title,
  intro,
  path,
  sections,
}: {
  title: string
  intro: string
  path: string
  sections: LegalSection[]
}) {
  return (
    <>
      <section className="border-b border-border bg-bg px-4 pt-16 pb-12 sm:px-6 sm:pt-20 lg:px-10">
        <PageHeader eyebrow="Legal" title={title} description={intro} />
        <p className="mt-6 text-center text-sm text-fg-subtle">Last updated {site.legalUpdated}</p>
      </section>
      <div className="bg-bg-alt px-4 py-12 sm:px-6 sm:py-16 lg:px-10">
        <div className="mx-auto grid max-w-7xl gap-10 lg:grid-cols-[260px_minmax(0,1fr)]">
          <aside className="hidden lg:block">
            <nav aria-label="Sections" className="sticky top-24 space-y-1">
              <p className="mb-3 text-xs font-bold tracking-widest text-fg-subtle uppercase">
                On This Page
              </p>
              {sections.map((section) => (
                <a
                  key={section.id}
                  href={`#${section.id}`}
                  className="block rounded-lg px-3 py-2 text-sm text-fg-muted transition-colors hover:bg-surface-3 hover:text-fg"
                >
                  {section.title}
                </a>
              ))}
              <div className="mt-6 border-t border-border pt-4">
                <p className="mb-2 text-xs font-bold tracking-widest text-fg-subtle uppercase">
                  Other Policies
                </p>
                {legalLinks
                  .filter((l) => l.href !== path)
                  .map((l) => (
                    <Link
                      key={l.href}
                      href={l.href}
                      className="block rounded-lg px-3 py-2 text-sm text-fg-muted hover:bg-surface-3 hover:text-fg"
                    >
                      {l.label}
                    </Link>
                  ))}
              </div>
            </nav>
          </aside>
          <article className="space-y-6">
            {sections.map((section, i) => (
              <section
                key={section.id}
                id={section.id}
                className="scroll-mt-24 rounded-3xl border border-border bg-surface p-6 sm:p-8"
              >
                <h2 className="text-2xl font-bold text-fg">
                  <span className="mr-2 text-primary">{i + 1}.</span>
                  {section.title}
                </h2>
                <div className="mt-4 space-y-3 leading-relaxed text-fg-muted [&_li]:ml-5 [&_li]:list-disc [&_strong]:text-fg [&_ul]:space-y-2">
                  {section.body}
                </div>
              </section>
            ))}
          </article>
        </div>
      </div>
      <JsonLd
        data={breadcrumbSchema([
          { name: "Home", path: "/" },
          { name: title, path },
        ])}
      />
    </>
  )
}
