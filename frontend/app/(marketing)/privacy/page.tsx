import type { Metadata } from "next"
import { ContactLine, LegalPage } from "@/components/marketing/legal-page"

export const metadata: Metadata = {
  title: "Privacy Policy",
  description:
    "What Self-Driven Car collects, why, how long it is kept, and how you can export or delete it.",
  alternates: { canonical: "/privacy" },
}

export default function PrivacyPage() {
  return (
    <LegalPage
      title="Privacy Policy"
      path="/privacy"
      intro="Self-Driven Car collects as little as it can. There are no accounts, no advertising, and no tracking cookies. This page explains exactly what is stored and why."
      sections={[
        {
          id: "summary",
          title: "The Short Version",
          body: (
            <ul>
              <li>You don&apos;t need an account, and we never ask for your name or email.</li>
              <li>We store an anonymous trial ID, your IP address, and the runs you start.</li>
              <li>
                We use this only to enforce the free limits and to show you your own run history.
              </li>
              <li>
                We don&apos;t sell your data, show ads, or use analytics or tracking services.
              </li>
            </ul>
          ),
        },
        {
          id: "collect",
          title: "What We Collect",
          body: (
            <>
              <p>
                When you press Start Training or open your dashboard, your browser talks to our
                simulation server. We keep:
              </p>
              <ul>
                <li>
                  <strong>A trial ID.</strong> A random identifier created in your browser and
                  combined with your IP address on the server. It isn&apos;t linked to your
                  identity.
                </li>
                <li>
                  <strong>Your IP address and browser type</strong>, taken from the request, and the
                  time each run started.
                </li>
                <li>
                  <strong>Your training runs:</strong> the settings you chose, any run name or notes
                  you typed, and the results of each generation.
                </li>
              </ul>
              <p>Please don&apos;t put personal information in run names or notes.</p>
            </>
          ),
        },
        {
          id: "use",
          title: "How We Use It",
          body: (
            <ul>
              <li>
                To count how many runs each network starts, so we can apply the limit of 5 per day
                and 20 per week.
              </li>
              <li>To show you your own runs, statistics, and champion networks.</li>
              <li>To understand overall usage and protect the service from abuse.</li>
            </ul>
          ),
        },
        {
          id: "browser",
          title: "What Stays in Your Browser",
          body: (
            <p>
              Your browser&apos;s local storage holds your trial ID, your theme, your simulator
              preferences, and a cached copy of your remaining runs. None of this is a cookie, and
              none of it is sent anywhere except the trial ID, which accompanies requests to our
              server. Clearing your browser data removes it.
            </p>
          ),
        },
        {
          id: "retention",
          title: "How Long We Keep It",
          body: (
            <ul>
              <li>
                Usage records (trial ID, IP address, browser type, and start time) are deleted
                automatically after 90 days.
              </li>
              <li>
                If you don&apos;t visit for 90 days, the last IP address and browser type saved with
                your trial ID are erased too.
              </li>
              <li>Training runs are kept until you delete them in Settings.</li>
            </ul>
          ),
        },
        {
          id: "sharing",
          title: "Where It Lives and Who Can See It",
          body: (
            <p>
              The website is hosted on Vercel and the simulation server on Railway, which store data
              on our behalf and may keep their own short-lived request logs. We don&apos;t share
              your data with anyone else, except where the law requires it.
            </p>
          ),
        },
        {
          id: "choices",
          title: "Your Choices",
          body: (
            <ul>
              <li>
                <strong>Export</strong> your runs as a JSON file from Settings.
              </li>
              <li>
                <strong>Delete</strong> your run history, or delete everything tied to your trial
                ID, from Settings.
              </li>
              <li>
                Usage records are kept after deletion until they expire, because they are needed to
                apply the free limits fairly.
              </li>
            </ul>
          ),
        },
        {
          id: "children",
          title: "Children",
          body: (
            <p>
              This is an educational tool and doesn&apos;t knowingly collect personal information
              from children. Because there are no accounts, no names or contact details are ever
              collected.
            </p>
          ),
        },
        {
          id: "contact",
          title: "Changes and Contact",
          body: (
            <p>
              If this policy changes, the date at the top of this page will change too. For
              questions or deletion requests, <ContactLine />.
            </p>
          ),
        },
      ]}
    />
  )
}
