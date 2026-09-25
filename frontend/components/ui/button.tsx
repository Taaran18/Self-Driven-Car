import { LoaderCircle } from "lucide-react"
import Link from "next/link"
import { forwardRef } from "react"
import { cn } from "@/lib/cn"

type Variant = "primary" | "secondary" | "outline" | "ghost" | "danger" | "danger-outline"
type Size = "sm" | "md" | "lg" | "icon" | "icon-sm"

const base =
  "inline-flex shrink-0 items-center justify-center gap-2 rounded-xl font-semibold whitespace-nowrap transition-[background-color,color,border-color,box-shadow,transform] duration-150 select-none active:scale-[0.98] disabled:pointer-events-none disabled:opacity-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-ring"

const variants: Record<Variant, string> = {
  primary: "bg-primary text-primary-fg shadow-sm hover:bg-primary-hover",
  secondary: "bg-surface-3 text-fg hover:bg-border",
  outline:
    "border border-border-strong bg-surface text-fg hover:border-fg-subtle hover:bg-surface-2",
  ghost: "text-fg-muted hover:bg-surface-3 hover:text-fg",
  danger: "bg-danger-solid text-white shadow-sm hover:bg-danger-solid-hover",
  "danger-outline": "border border-danger/40 text-danger hover:bg-danger-soft",
}

const sizes: Record<Size, string> = {
  sm: "h-9 px-3 text-sm",
  md: "h-11 px-4 text-sm",
  lg: "h-13 px-6 text-base",
  icon: "size-11",
  "icon-sm": "size-9",
}

export function buttonClasses(variant: Variant = "primary", size: Size = "md", className?: string) {
  return cn(base, variants[variant], sizes[size], className)
}

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant
  size?: Size
  loading?: boolean
  loadingText?: string
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  {
    variant = "primary",
    size = "md",
    loading,
    loadingText,
    className,
    children,
    disabled,
    type = "button",
    ...props
  },
  ref,
) {
  return (
    <button
      ref={ref}
      type={type}
      className={buttonClasses(variant, size, className)}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      {...props}
    >
      {loading ? <LoaderCircle aria-hidden className="size-4 animate-spin" /> : null}
      {loading && loadingText ? loadingText : children}
    </button>
  )
})

interface ButtonLinkProps extends React.ComponentProps<typeof Link> {
  variant?: Variant
  size?: Size
}

export function ButtonLink({
  variant = "primary",
  size = "md",
  className,
  ...props
}: ButtonLinkProps) {
  return <Link className={buttonClasses(variant, size, className)} {...props} />
}
