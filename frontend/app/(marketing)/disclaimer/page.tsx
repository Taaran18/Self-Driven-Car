import type { Metadata } from "next"
import { ContactLine, LegalPage } from "@/components/marketing/legal-page"

export const metadata: Metadata = {
  title: "Disclaimer",
  description:
    "Self-Driven Car is an educational simulation, not a real autonomous driving system.",
  alternates: { canonical: "/disclaimer" },
}

export default function DisclaimerPage() {
  return (
    <LegalPage
      title="Disclaimer"
      path="/disclaimer"
      intro="Please read this before drawing conclusions from the simulator."
      sections={[
        {
          id: "education",
          title: "For Education Only",
          body: (
            <p>
              Self-Driven Car is a teaching tool that demonstrates neuroevolution. It is{" "}
              <strong>not</strong> a self-driving system, and nothing here is designed, tested, or
              suitable for controlling a real vehicle.
            </p>
          ),
        },
        {
          id: "simplified",
          title: "A Deliberately Simplified World",
          body: (
            <ul>
              <li>
                The cars live on a flat, two-dimensional road with no traffic, pedestrians, signs,
                or weather.
              </li>
              <li>The physics is simplified, and the sensors are perfect and noise-free.</li>
              <li>A network that drives well here says nothing about real-world driving safety.</li>
            </ul>
          ),
        },
        {
          id: "results",
          title: "Results Vary",
          body: (
            <p>
              Evolution is random. The same settings can learn quickly one time and slowly the next.
              Fitness numbers are only meaningful inside this simulator.
            </p>
          ),
        },
        {
          id: "advice",
          title: "Not Professional Advice",
          body: (
            <p>
              The explanations on this site are simplified to help you learn. They are not
              engineering, safety, or academic advice. For rigorous detail, read the original
              research linked on the How It Works page.
            </p>
          ),
        },
        {
          id: "links",
          title: "External Links",
          body: (
            <p>
              Links to other websites are provided for convenience. We don&apos;t control and
              aren&apos;t responsible for their content.
            </p>
          ),
        },
        {
          id: "contact",
          title: "Contact",
          body: (
            <p>
              Spotted something inaccurate? <ContactLine />.
            </p>
          ),
        },
      ]}
    />
  )
}
