import { ArrowRight, ExternalLink } from "lucide-react"
import type { Metadata } from "next"
import {
  DecideIllustration,
  NetworkIllustration,
  ScoreIllustration,
  SensorIllustration,
} from "@/components/marketing/illustrations"
import { JsonLd } from "@/components/seo/json-ld"
import { ButtonLink } from "@/components/ui/button"
import { PageHeader } from "@/components/ui/page-header"
import { Reveal } from "@/components/ui/reveal"
import { TableWrap, TBody, THead } from "@/components/ui/table"
import { cn } from "@/lib/cn"
import { articleSchema, breadcrumbSchema } from "@/lib/schema"

const TITLE = "How It Works: Neuroevolution Explained"
const DESCRIPTION =
  "A plain-language deep dive into how the self-driving cars sense the road, make decisions with neural networks, and improve through the NEAT genetic algorithm."

export const metadata: Metadata = {
  title: "How It Works",
  description: DESCRIPTION,
  alternates: { canonical: "/how-it-works" },
}

const toc = [
  { id: "problem", label: "The Challenge" },
  { id: "senses", label: "Senses" },
  { id: "brain", label: "The Brain" },
  { id: "actions", label: "Actions" },
  { id: "fitness", label: "Fitness" },
  { id: "evolution", label: "Evolution" },
  { id: "settings", label: "Settings to Try" },
  { id: "glossary", label: "Glossary" },
]

const settings = [
  {
    name: "Cars per Generation",
    effect: "How many different networks are tried each generation.",
    tryThis:
      "Compare 20 cars with 100 cars. More cars usually finds a good driver in fewer generations.",
  },
  {
    name: "Track Variety",
    effect: "Whether each generation drives a new road or the same one.",
    tryThis: "Use Same Track, then watch a champion fail on a new track. That is overfitting.",
  },
  {
    name: "Track Width and Curves",
    effect: "How hard the road is to follow.",
    tryThis: "Narrow and Twisty tracks need more generations and often grow hidden neurons.",
  },
  {
    name: "Weight Mutation Chance",
    effect: "How often a child's connection weights are nudged.",
    tryThis: "Set it very low and progress stalls, because children barely differ from parents.",
  },
  {
    name: "New Connection and Neuron Chance",
    effect: "How quickly networks grow new structure.",
    tryThis: "Raise the neuron chance and watch the brain diagram get more complex.",
  },
]

const glossary = [
  ["Tick", "One step of the simulation. There are 30 ticks per second at normal speed."],
  [
    "Neural Network",
    "A web of neurons joined by weighted connections that turns inputs into outputs.",
  ],
  [
    "Weight",
    "A number on a connection that strengthens, weakens, or flips the signal passing through it.",
  ],
  [
    "Bias",
    "A number added inside a neuron before its activation function. It shifts when the neuron fires.",
  ],
  [
    "Activation Function",
    "The squashing function inside each neuron. This project uses a scaled tanh, which outputs −1 to 1.",
  ],
  [
    "Genome",
    "The genetic description of one network: its neurons, its connections, and their weights.",
  ],
  [
    "Fitness",
    "The score a genome earns. Here it is roughly the distance driven, in hundreds of units.",
  ],
  [
    "Generation",
    "One round in which every car drives, gets scored, and is replaced by the next population.",
  ],
  ["Species", "A group of genomes with similar structure that mostly compete among themselves."],
  ["Elitism", "Copying the very best genomes of each species into the next generation unchanged."],
  ["Crossover", "Building a child genome by combining the genes of two parents."],
  ["Mutation", "A random change to a genome, like nudging a weight or adding a neuron."],
  [
    "Stagnation",
    "When a species stops improving. NEAT removes species that stagnate for 20 generations.",
  ],
]

function Section({
  id,
  eyebrow,
  title,
  alt,
  children,
  visual,
}: {
  id: string
  eyebrow: string
  title: string
  alt?: boolean
  children: React.ReactNode
  visual?: React.ReactNode
}) {
  return (
    <section
      id={id}
      className={cn(
        "scroll-mt-20 px-4 py-16 sm:px-6 sm:py-24 lg:px-10",
        alt ? "border-y border-border bg-bg-alt" : "bg-bg",
      )}
    >
      <div
        className={cn(
          "mx-auto grid max-w-7xl items-center gap-10",
          visual ? "lg:grid-cols-[1.2fr_1fr] lg:gap-16" : undefined,
        )}
      >
        <Reveal className="min-w-0">
          <p className="text-sm font-semibold tracking-wider text-primary uppercase">{eyebrow}</p>
          <h2 className="mt-2 text-3xl font-bold tracking-tight text-fg sm:text-4xl">{title}</h2>
          <div className="mt-5 space-y-4 text-base leading-relaxed text-fg-muted sm:text-lg [&_strong]:text-fg">
            {children}
          </div>
        </Reveal>
        {visual ? (
          <Reveal
            from="right"
            delay={120}
            className="min-w-0 rounded-3xl border border-border bg-surface p-6 shadow-[var(--shadow-card)]"
          >
            {visual}
          </Reveal>
        ) : null}
      </div>
    </section>
  )
}

export default function HowItWorksPage() {
  return (
    <>
      <section className="bg-bg px-4 pt-16 pb-12 sm:px-6 sm:pt-24 lg:px-10">
        <Reveal>
          <PageHeader
            size="lg"
            eyebrow="The Deep Dive"
            title="How the Cars Learn to Drive"
            description="Everything happening inside the simulator, from a single sensor reading to a whole species evolving. No math degree required."
          />
        </Reveal>
        <Reveal delay={120}>
          <nav
            aria-label="On this page"
            className="mx-auto mt-10 flex max-w-4xl flex-wrap justify-center gap-2"
          >
            {toc.map((item) => (
              <a
                key={item.id}
                href={`#${item.id}`}
                className="rounded-full border border-border bg-surface px-4 py-2 text-sm font-medium text-fg-muted transition-colors hover:border-primary/40 hover:text-fg"
              >
                {item.label}
              </a>
            ))}
          </nav>
        </Reveal>
      </section>

      <Section id="problem" eyebrow="1 · The Challenge" title="Driving Without Being Taught" alt>
        <p>
          Most software follows rules a programmer wrote. Here, nobody writes the rule “turn left
          when the wall on the right gets close.” Instead, each car gets a small neural network with{" "}
          <strong>random</strong> settings, and the only feedback is a score for how far it drove.
        </p>
        <p>
          That is the idea behind <strong>neuroevolution</strong>: treat each network like a living
          creature, keep the ones that do well, and let them have slightly different children.
          Repeat for enough generations and good driving emerges on its own.
        </p>
      </Section>

      <Section
        id="senses"
        eyebrow="2 · Senses"
        title="Eight Rays and a Speedometer"
        visual={<SensorIllustration className="mx-auto h-72 w-full max-w-xs" />}
      >
        <p>
          Each car casts <strong>8 rays</strong> every 45°, reaching up to 200 units. For every ray
          the engine finds the closest point where it crosses a road edge.
        </p>
        <p>
          The distance becomes a <strong>closeness</strong> value: 0 when nothing is in range and
          close to 1 when a wall is right next to the car. The ninth input is the car&apos;s speed
          divided by its top speed, so all inputs sit between 0 and 1.
        </p>
      </Section>

      <Section
        id="brain"
        eyebrow="3 · The Brain"
        title="A Tiny Neural Network"
        alt
        visual={<NetworkIllustration className="mx-auto h-72 w-full max-w-xs" />}
      >
        <p>
          The 9 inputs feed into a network with <strong>4 outputs</strong>. Every connection
          multiplies the signal by a weight. Each neuron adds up what arrives, adds its bias, and
          passes the total through a scaled <strong>tanh</strong> function, which squeezes any
          number into the range −1 to 1.
        </p>
        <p>
          Every network starts with each input wired straight to each output and no hidden neurons.
          Hidden neurons only appear later, when mutation adds them and they turn out to help.
        </p>
      </Section>

      <Section
        id="actions"
        eyebrow="4 · Actions"
        title="From Four Numbers to Four Pedals"
        visual={<DecideIllustration className="mx-auto h-72 w-full max-w-xs" />}
      >
        <p>
          The outputs map to <strong>Accelerate, Brake, Turn Left, and Turn Right</strong>. An
          action only fires when its output is above 0.5 and higher than its opposite, so a car can
          never press both pedals or steer both ways at once.
        </p>
        <p>
          Physics is kept deliberately simple. Turning rotates the car 2° per tick, acceleration
          adds 0.2 to the speed, braking removes 1, friction removes 0.1, and speed is capped at 10
          units per tick.
        </p>
      </Section>

      <Section
        id="fitness"
        eyebrow="5 · Fitness"
        title="Scoring Every Driver"
        alt
        visual={<ScoreIllustration className="mx-auto h-72 w-full max-w-xs" />}
      >
        <p>
          Each tick, a car earns fitness equal to how far it moved up the track, divided by 100. A
          car is taken off the road if it <strong>hits a wall</strong>, falls more than 200 units
          behind the leader, drives backward, or stalls. That costs it 1 point.
        </p>
        <p>
          A generation ends when every car has stopped, crossed the finish line, or run out of time.
          A run is solved when any car reaches the finish line.
        </p>
      </Section>

      <Section id="evolution" eyebrow="6 · Evolution" title="How NEAT Breeds Better Drivers">
        <p>
          The simulator uses <strong>NEAT</strong>, short for NeuroEvolution of Augmenting
          Topologies. Between generations it does five things:
        </p>
        <ol className="list-decimal space-y-2 pl-6">
          <li>
            <strong>Speciate.</strong> Group similar networks into species, so a new idea competes
            with its own kind first.
          </li>
          <li>
            <strong>Keep the elite.</strong> The best 2 networks of each species are copied into the
            next generation unchanged.
          </li>
          <li>
            <strong>Select parents.</strong> Only the top 20% of each species may reproduce.
          </li>
          <li>
            <strong>Crossover.</strong> A child inherits connections from two parents, preferring
            the fitter parent.
          </li>
          <li>
            <strong>Mutate.</strong> Weights are nudged (80% chance by default), and new connections
            (30%) or neurons (20%) may appear. Connections and neurons can also be removed.
          </li>
        </ol>
        <p>
          Species that fail to improve for 20 generations are removed, which frees up room for new
          approaches.
        </p>
      </Section>

      <section
        id="settings"
        className="scroll-mt-20 border-y border-border bg-bg-alt px-4 py-16 sm:px-6 sm:py-24 lg:px-10"
      >
        <div className="mx-auto max-w-7xl">
          <Reveal className="mx-auto max-w-3xl text-center">
            <p className="text-sm font-semibold tracking-wider text-primary uppercase">
              7 · Experiments
            </p>
            <h2 className="mt-2 text-3xl font-bold tracking-tight text-fg sm:text-4xl">
              Settings Worth Trying
            </h2>
            <p className="mt-4 text-lg text-fg-muted">
              Change one setting at a time and compare runs on your dashboard.
            </p>
          </Reveal>
          <Reveal delay={100} className="mt-10">
            <TableWrap label="Settings and what they change">
              <THead>
                <tr>
                  <th scope="col">Setting</th>
                  <th scope="col">What It Changes</th>
                  <th scope="col">Try This</th>
                </tr>
              </THead>
              <TBody>
                {settings.map((row) => (
                  <tr key={row.name}>
                    <td className="font-semibold whitespace-nowrap text-fg">{row.name}</td>
                    <td className="text-fg-muted">{row.effect}</td>
                    <td className="text-fg-muted">{row.tryThis}</td>
                  </tr>
                ))}
              </TBody>
            </TableWrap>
          </Reveal>
        </div>
      </section>

      <section id="glossary" className="scroll-mt-20 bg-bg px-4 py-16 sm:px-6 sm:py-24 lg:px-10">
        <div className="mx-auto max-w-7xl">
          <Reveal className="mx-auto max-w-3xl text-center">
            <p className="text-sm font-semibold tracking-wider text-primary uppercase">
              8 · Glossary
            </p>
            <h2 className="mt-2 text-3xl font-bold tracking-tight text-fg sm:text-4xl">
              Words You&apos;ll See in the Simulator
            </h2>
          </Reveal>
          <dl className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {glossary.map(([term, definition], i) => (
              <Reveal
                key={term}
                delay={(i % 3) * 80}
                className="rounded-2xl border border-border bg-surface p-5"
              >
                <dt className="font-bold text-fg">{term}</dt>
                <dd className="mt-1.5 text-sm leading-relaxed text-fg-muted">{definition}</dd>
              </Reveal>
            ))}
          </dl>
          <Reveal className="mt-12 flex flex-col items-center gap-4 text-center">
            <p className="text-fg-muted">Want the original sources?</p>
            <div className="flex flex-wrap justify-center gap-3">
              <a
                href="https://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-fg hover:border-border-strong"
              >
                The Original NEAT Paper (2002) <ExternalLink className="size-4" />
              </a>
              <a
                href="https://neat-python.readthedocs.io/"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 rounded-xl border border-border bg-surface px-4 py-2.5 text-sm font-semibold text-fg hover:border-border-strong"
              >
                NEAT-Python Documentation <ExternalLink className="size-4" />
              </a>
            </div>
            <ButtonLink href="/simulator" size="lg" className="mt-6">
              Try It in the Simulator <ArrowRight className="size-5" />
            </ButtonLink>
          </Reveal>
        </div>
      </section>

      <JsonLd
        data={[
          articleSchema({ title: TITLE, description: DESCRIPTION, path: "/how-it-works" }),
          breadcrumbSchema([
            { name: "Home", path: "/" },
            { name: "How It Works", path: "/how-it-works" },
          ]),
        ]}
      />
    </>
  )
}
