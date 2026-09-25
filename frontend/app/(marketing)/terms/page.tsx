import type { Metadata } from "next"
import Link from "next/link"
import { ContactLine, LegalPage } from "@/components/marketing/legal-page"

export const metadata: Metadata = {
  title: "Terms of Service",
  description:
    "The rules for using the Self-Driven Car simulator, including free usage limits and acceptable use.",
  alternates: { canonical: "/terms" },
}

export default function TermsPage() {
  return (
    <LegalPage
      title="Terms of Service"
      path="/terms"
      intro="These terms cover your use of the Self-Driven Car website and simulator. By using them, you agree to these terms."
      sections={[
        {
          id: "service",
          title: "The Service",
          body: (
            <p>
              Self-Driven Car is a free educational simulator that shows how neural networks evolve
              with the NEAT algorithm. It is provided as a learning tool, not as a commercial
              product.
            </p>
          ),
        },
        {
          id: "limits",
          title: "Free Usage Limits",
          body: (
            <>
              <p>
                To keep the project affordable, each network (IP address) can start{" "}
                <strong>5 training runs per day</strong> and <strong>20 per week</strong>. The daily
                limit resets at midnight UTC and the weekly limit on Monday at midnight UTC.
              </p>
              <p>
                Only a limited number of simulations can run at the same time, and each run is
                capped in length.
              </p>
            </>
          ),
        },
        {
          id: "acceptable-use",
          title: "Acceptable Use",
          body: (
            <>
              <p>You agree not to:</p>
              <ul>
                <li>
                  Try to get around the usage limits, for example by rotating IP addresses or
                  automating requests.
                </li>
                <li>Overload, probe, or disrupt the servers or the people using them.</li>
                <li>Use the service for anything unlawful.</li>
              </ul>
              <p>We may block access that breaks these rules.</p>
            </>
          ),
        },
        {
          id: "availability",
          title: "Availability",
          body: (
            <p>
              The simulation server sleeps when idle and may take a moment to wake up. We may
              change, pause, or stop the service, or reset stored data, at any time and without
              notice.
            </p>
          ),
        },
        {
          id: "content",
          title: "Your Runs",
          body: (
            <p>
              You are responsible for anything you type into run names or notes. Your runs are only
              visible to your trial ID. See the{" "}
              <Link href="/privacy" className="font-semibold text-primary hover:underline">
                Privacy Policy
              </Link>{" "}
              for how they are stored.
            </p>
          ),
        },
        {
          id: "source",
          title: "Source Code",
          body: (
            <p>
              The source code is published on GitHub. Its use is governed by the license in that
              repository.
            </p>
          ),
        },
        {
          id: "warranty",
          title: "No Warranty",
          body: (
            <p>
              The service is provided “as is” and “as available,” without warranties of any kind.
              Simulation results are for learning only. See the{" "}
              <Link href="/disclaimer" className="font-semibold text-primary hover:underline">
                Disclaimer
              </Link>
              .
            </p>
          ),
        },
        {
          id: "liability",
          title: "Limitation of Liability",
          body: (
            <p>
              To the fullest extent allowed by law, the project and its contributors are not liable
              for any indirect, incidental, or consequential damages, or for any loss of data,
              arising from your use of the service.
            </p>
          ),
        },
        {
          id: "changes",
          title: "Changes and Contact",
          body: (
            <p>
              We may update these terms, and the date at the top of the page will show when.
              Continuing to use the service means you accept the updated terms. For questions,{" "}
              <ContactLine />.
            </p>
          ),
        },
      ]}
    />
  )
}
