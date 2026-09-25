import {
  ArrowRight,
  BarChart3,
  BrainCircuit,
  CodeXml,
  Dna,
  GitMerge,
  ListChecks,
  Route,
  Shuffle,
  Sparkles,
  Split,
  Trophy,
} from "lucide-react"
import type { Metadata } from "next"
import { FaqList } from "@/components/marketing/faq"
import { HeroVisual } from "@/components/marketing/hero-visual"
import { LearnSteps } from "@/components/marketing/learn-steps"
import { JsonLd } from "@/components/seo/json-ld"
import { ButtonLink } from "@/components/ui/button"
import { CodeBlock } from "@/components/ui/code-block"
import { SectionHeading } from "@/components/ui/page-header"
import { Reveal } from "@/components/ui/reveal"
import { DECIDE_SOURCE, FAQ } from "@/lib/content"
import { faqSchema, webAppSchema } from "@/lib/schema"
import { site } from "@/lib/site"

export const metadata: Metadata = {
  title: { absolute: `${site.name} | ${site.tagline}` },
  description: site.description,
  alternates: { canonical: "/" },
}

const features = [
  {
    icon: Route,
    title: "A Live Track",
    text: "Dozens of cars share the road at once. Crashes, the leader, and the finish line are drawn as they happen.",
  },
  {
    icon: BrainCircuit,
    title: "The Network Inside",
    text: "See the leading car's neural network light up, including every neuron evolution has added along the way.",
  },
  {
    icon: CodeXml,
    title: "The Real Code",
    text: "Read the actual Python functions that run each step, next to the numbers they are producing right now.",
  },
  {
    icon: BarChart3,
    title: "Evolution in Numbers",
    text: "Track best and average fitness, species, and network size for every generation, and keep your history.",
  },
]

const evolution = [
  {
    icon: ListChecks,
    title: "Evaluate",
    text: "Every car gets a fitness score for how far it drove.",
  },
  {
    icon: Trophy,
    title: "Select",
    text: "The top 20% of each species become parents, and the best two in each species survive untouched.",
  },
  {
    icon: GitMerge,
    title: "Crossover",
    text: "Two parents combine their connections into a child network.",
  },
  {
    icon: Shuffle,
    title: "Mutate",
    text: "Weights get nudged, and sometimes a new connection or neuron appears.",
  },
  {
    icon: Split,
    title: "Speciate",
    text: "Similar networks are grouped, so new ideas aren't wiped out too early.",
  },
]

const facts = [
  { value: "8", label: "Distance Sensors per Car" },
  { value: "9", label: "Inputs to Every Network" },
  { value: "4", label: "Possible Actions" },
  { value: "30", label: "Decisions per Second" },
  { value: "50", label: "Cars per Generation by Default" },
  { value: "5", label: "Free Runs a Day" },
]

export default function HomePage() {
  return (
    <>
      <section className="relative overflow-hidden bg-bg px-4 pt-16 pb-24 sm:px-6 sm:pt-24 lg:px-10">
        <div className="mx-auto max-w-5xl text-center">
          <Reveal>
            <p className="inline-flex items-center gap-2 rounded-full border border-primary/25 bg-primary-soft px-3.5 py-1.5 text-sm font-semibold text-primary-soft-fg">
              <Sparkles className="size-4" /> Free Neuroevolution Lab · No Sign-Up
            </p>
          </Reveal>
          <Reveal delay={80}>
            <h1 className="mt-6 text-5xl font-bold tracking-tight text-fg sm:text-6xl lg:text-7xl">
              Watch a Neural Network <span className="text-primary">Learn to Drive</span>
            </h1>
          </Reveal>
          <Reveal delay={160}>
            <p className="mx-auto mt-6 max-w-2xl text-lg leading-relaxed text-fg-muted sm:text-xl">
              No rules, no human driving data. Just dozens of tiny brains that improve through
              evolution, while you follow every sensor reading, decision, and line of code.
            </p>
          </Reveal>
          <Reveal
            delay={240}
            className="mt-9 flex flex-col items-center justify-center gap-3 sm:flex-row"
          >
            <ButtonLink href="/simulator" size="lg" className="w-full sm:w-auto">
              Start the Simulator <ArrowRight className="size-5" />
            </ButtonLink>
            <ButtonLink
              href="#how-it-learns"
              size="lg"
              variant="outline"
              className="w-full sm:w-auto"
            >
              See How It Learns
            </ButtonLink>
          </Reveal>
          <Reveal delay={300}>
            <p className="mt-5 text-sm text-fg-subtle">
              5 free training runs a day · Works on phone, tablet, and desktop
            </p>
          </Reveal>
        </div>
        <Reveal from="scale" delay={200}>
          <HeroVisual />
        </Reveal>
      </section>

      <section className="border-y border-border bg-bg-alt px-4 py-20 sm:px-6 sm:py-28 lg:px-10">
        <div className="mx-auto max-w-[1440px]">
          <Reveal>
            <SectionHeading
              eyebrow="What You Can Explore"
              title="Everything the Algorithm Does, Out in the Open"
              description="Most AI demos hide the interesting part. Here, every stage of learning is visible and explained."
            />
          </Reveal>
          <div className="mt-14 grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
            {features.map((feature, i) => (
              <Reveal
                key={feature.title}
                delay={i * 90}
                className="group rounded-3xl border border-border bg-surface p-7 shadow-[var(--shadow-card)] transition-[border-color,transform] duration-300 hover:-translate-y-1 hover:border-primary/40"
              >
                <span className="flex size-12 items-center justify-center rounded-2xl bg-primary-soft text-primary-soft-fg transition-colors group-hover:bg-primary group-hover:text-primary-fg">
                  <feature.icon className="size-6" />
                </span>
                <h3 className="mt-6 text-xl font-bold text-fg">{feature.title}</h3>
                <p className="mt-2 leading-relaxed text-fg-muted">{feature.text}</p>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section id="how-it-learns" className="bg-bg px-4 py-20 sm:px-6 sm:py-28 lg:px-10">
        <Reveal>
          <SectionHeading
            eyebrow="Thirty Times a Second"
            title="How a Car Learns to Drive"
            description="Each car repeats the same five steps on every tick. Scroll through them to see what happens inside."
          />
        </Reveal>
        <LearnSteps />
      </section>

      <section className="border-y border-border bg-bg-alt px-4 py-20 sm:px-6 sm:py-28 lg:px-10">
        <div className="mx-auto max-w-[1440px]">
          <Reveal>
            <SectionHeading
              eyebrow="Between Generations"
              title="How Evolution Picks the Winners"
              description="When every car has stopped, NEAT turns their scores into a better next generation in five moves."
            />
          </Reveal>
          <ol className="relative mt-14 grid gap-5 md:grid-cols-5">
            <div
              aria-hidden
              className="absolute top-8 right-[10%] left-[10%] hidden h-0.5 bg-gradient-to-r from-primary/10 via-primary/50 to-primary/10 md:block"
            />
            {evolution.map((item, i) => (
              <Reveal as="li" key={item.title} delay={i * 110} className="relative text-center">
                <span className="relative mx-auto flex size-16 items-center justify-center rounded-2xl border border-primary/30 bg-surface text-primary shadow-[var(--shadow-card)]">
                  <item.icon className="size-7" />
                  <span className="absolute -top-2 -right-2 flex size-6 items-center justify-center rounded-full bg-primary text-xs font-bold text-primary-fg">
                    {i + 1}
                  </span>
                </span>
                <h3 className="mt-5 text-lg font-bold text-fg">{item.title}</h3>
                <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-fg-muted">
                  {item.text}
                </p>
              </Reveal>
            ))}
          </ol>
          <Reveal className="mt-12 text-center">
            <ButtonLink href="/how-it-works" variant="outline">
              Read the Deep Dive <ArrowRight className="size-4" />
            </ButtonLink>
          </Reveal>
        </div>
      </section>

      <section className="bg-bg px-4 py-20 sm:px-6 sm:py-28 lg:px-10">
        <div className="mx-auto grid max-w-7xl items-center gap-12 lg:grid-cols-[1fr_1.15fr]">
          <Reveal from="left" className="min-w-0">
            <p className="text-sm font-semibold tracking-wider text-primary uppercase">
              Real Code, Not a Mock-Up
            </p>
            <h2 className="mt-2 text-3xl font-bold tracking-tight text-fg sm:text-4xl lg:text-5xl">
              Read the Exact Function Making the Choice
            </h2>
            <p className="mt-5 text-lg leading-relaxed text-fg-muted">
              This is the real decision function from the simulation engine. In the simulator, it
              sits next to the live numbers it receives, so you can check every choice a car makes
              by hand.
            </p>
            <ul className="mt-6 space-y-3 text-fg-muted">
              {[
                "Nine steps cover the whole learning loop",
                "Highlighted lines follow each step as it runs",
                "Pause and Step to freeze one tick and inspect it",
              ].map((line) => (
                <li key={line} className="flex items-start gap-3">
                  <span aria-hidden className="mt-2 size-1.5 shrink-0 rounded-full bg-primary" />
                  {line}
                </li>
              ))}
            </ul>
            <ButtonLink href="/simulator" className="mt-8">
              Open the Code Walkthrough <ArrowRight className="size-4" />
            </ButtonLink>
          </Reveal>
          <Reveal from="right" delay={120} className="min-w-0">
            <CodeBlock
              code={DECIDE_SOURCE}
              file="app/simulation/engine.py"
              startLine={85}
              highlight={[2]}
            />
          </Reveal>
        </div>
      </section>

      <section className="border-y border-border bg-bg-alt px-4 py-20 sm:px-6 sm:py-24 lg:px-10">
        <div className="mx-auto max-w-[1440px]">
          <Reveal>
            <SectionHeading eyebrow="By the Numbers" title="The Numbers Behind Every Car" />
          </Reveal>
          <dl className="mt-12 grid grid-cols-2 gap-4 md:grid-cols-3 xl:grid-cols-6">
            {facts.map((fact, i) => (
              <Reveal
                key={fact.label}
                delay={i * 70}
                className="rounded-3xl border border-border bg-surface p-6 text-center"
              >
                <dd className="tabular font-display text-4xl font-bold text-primary sm:text-5xl">
                  {fact.value}
                </dd>
                <dt className="mt-2 text-sm font-medium text-fg-muted">{fact.label}</dt>
              </Reveal>
            ))}
          </dl>
        </div>
      </section>

      <section id="faq" className="bg-bg px-4 py-20 sm:px-6 sm:py-28 lg:px-10">
        <Reveal>
          <SectionHeading eyebrow="Questions" title="Frequently Asked Questions" />
        </Reveal>
        <Reveal delay={100}>
          <FaqList />
        </Reveal>
      </section>

      <section className="border-t border-border bg-bg-alt px-4 py-20 sm:px-6 sm:py-24 lg:px-10">
        <Reveal
          from="scale"
          className="relative mx-auto max-w-5xl overflow-hidden rounded-[2rem] border border-primary/30 bg-surface px-6 py-14 text-center shadow-[var(--shadow-pop)] sm:px-12"
        >
          <div
            aria-hidden
            className="absolute inset-0 -z-0 bg-[radial-gradient(circle_at_top,var(--primary-soft),transparent_65%)]"
          />
          <div className="relative">
            <Dna className="mx-auto size-10 text-primary" aria-hidden />
            <h2 className="mt-5 text-3xl font-bold tracking-tight text-fg sm:text-5xl">
              Start Your First Generation
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-lg text-fg-muted">
              It takes one click. Within a few minutes, cars that could barely leave the start line
              are usually taking corners on their own.
            </p>
            <ButtonLink href="/simulator" size="lg" className="mt-8">
              Start the Simulator <ArrowRight className="size-5" />
            </ButtonLink>
          </div>
        </Reveal>
      </section>

      <JsonLd data={[webAppSchema(), faqSchema(FAQ)]} />
    </>
  )
}
