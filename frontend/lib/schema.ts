import { site } from "@/lib/site"

export function websiteSchema() {
  return {
    "@context": "https://schema.org",
    "@type": "WebSite",
    name: site.name,
    url: site.url,
    description: site.description,
    inLanguage: "en",
  }
}

export function webAppSchema() {
  return {
    "@context": "https://schema.org",
    "@type": "WebApplication",
    name: site.name,
    url: `${site.url}/simulator`,
    applicationCategory: "EducationalApplication",
    operatingSystem: "Any modern web browser",
    description: site.description,
    isAccessibleForFree: true,
    offers: { "@type": "Offer", price: "0", priceCurrency: "USD" },
    featureList: [
      "Live neuroevolution simulation with NEAT",
      "Real-time neural network visualization",
      "Step-by-step view of the real Python source code",
      "Generation statistics and species tracking",
      "Saved training history for signed-in users",
    ],
    sameAs: [site.githubUrl],
  }
}

export function breadcrumbSchema(items: { name: string; path: string }[]) {
  return {
    "@context": "https://schema.org",
    "@type": "BreadcrumbList",
    itemListElement: items.map((item, index) => ({
      "@type": "ListItem",
      position: index + 1,
      name: item.name,
      item: `${site.url}${item.path}`,
    })),
  }
}

export function faqSchema(items: { question: string; answer: string }[]) {
  return {
    "@context": "https://schema.org",
    "@type": "FAQPage",
    mainEntity: items.map((item) => ({
      "@type": "Question",
      name: item.question,
      acceptedAnswer: { "@type": "Answer", text: item.answer },
    })),
  }
}

export function articleSchema({
  title,
  description,
  path,
}: {
  title: string
  description: string
  path: string
}) {
  return {
    "@context": "https://schema.org",
    "@type": "TechArticle",
    headline: title,
    description,
    url: `${site.url}${path}`,
    inLanguage: "en",
    isAccessibleForFree: true,
    about: ["Neuroevolution", "NEAT", "Neural networks", "Genetic algorithms"],
    publisher: { "@type": "Organization", name: site.name, url: site.url },
  }
}
