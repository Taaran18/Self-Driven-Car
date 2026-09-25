import type { Metadata } from "next"
import { JsonLd } from "@/components/seo/json-ld"
import { SimulatorView } from "@/components/simulator/simulator-view"
import { breadcrumbSchema, webAppSchema } from "@/lib/schema"

export const metadata: Metadata = {
  title: "Simulator",
  description:
    "Train a population of self-driving cars in your browser. Watch NEAT evolve their neural networks live, step through the real Python code, and track fitness by generation.",
  alternates: { canonical: "/simulator" },
}

export default function SimulatorPage() {
  return (
    <>
      <SimulatorView />
      <JsonLd
        data={[
          webAppSchema(),
          breadcrumbSchema([
            { name: "Home", path: "/" },
            { name: "Simulator", path: "/simulator" },
          ]),
        ]}
      />
    </>
  )
}
