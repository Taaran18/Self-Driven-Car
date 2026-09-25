const siteUrl = (process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000").replace(/\/$/, "")

export const site = {
  name: "Self-Driven Car",
  shortName: "Self-Driven",
  tagline: "Watch a Neural Network Learn to Drive",
  description:
    "An interactive neuroevolution lab. Watch cars evolve their own neural networks with NEAT, follow every sensor reading and decision live, and read the real Python code as it runs.",
  url: siteUrl,
  githubUrl: process.env.NEXT_PUBLIC_GITHUB_URL ?? "https://github.com/Taaran18/Self-Driven-Car",
  contactEmail: process.env.NEXT_PUBLIC_CONTACT_EMAIL || null,
  apiUrl: (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, ""),
  legalUpdated: "September 25, 2026",
  keywords: [
    "self-driving car simulation",
    "neuroevolution",
    "NEAT algorithm",
    "genetic algorithm",
    "neural network visualization",
    "machine learning for beginners",
    "reinforcement learning demo",
    "interactive AI lab",
  ],
}

export const legalLinks = [
  { href: "/privacy", label: "Privacy Policy" },
  { href: "/terms", label: "Terms of Service" },
  { href: "/disclaimer", label: "Disclaimer" },
]

export const marketingLinks = [
  { href: "/#how-it-learns", label: "How It Learns" },
  { href: "/how-it-works", label: "Deep Dive" },
  { href: "/#faq", label: "FAQ" },
]
