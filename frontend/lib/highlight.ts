export type TokenKind =
  "keyword" | "string" | "number" | "function" | "builtin" | "comment" | "plain"

export interface Token {
  kind: TokenKind
  text: string
}

const KEYWORDS = new Set([
  "def",
  "return",
  "for",
  "in",
  "if",
  "elif",
  "else",
  "while",
  "not",
  "and",
  "or",
  "is",
  "None",
  "True",
  "False",
  "class",
  "yield",
  "with",
  "as",
  "import",
  "from",
  "lambda",
  "continue",
  "break",
  "pass",
  "try",
  "except",
  "finally",
  "raise",
  "self",
])

const BUILTINS = new Set([
  "len",
  "range",
  "int",
  "float",
  "min",
  "max",
  "sum",
  "list",
  "dict",
  "zip",
  "enumerate",
  "sorted",
  "any",
  "all",
  "round",
  "np",
  "neat",
  "statistics",
  "random",
])

const PATTERN =
  /(#.*$)|("(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*')|(\b\d+(?:\.\d+)?(?:e[+-]?\d+)?\b)|([A-Za-z_][A-Za-z0-9_]*)|(\s+)|([^\sA-Za-z0-9_#"']+)/g

export function tokenizeLine(line: string): Token[] {
  const tokens: Token[] = []
  PATTERN.lastIndex = 0
  let match: RegExpExecArray | null
  while ((match = PATTERN.exec(line))) {
    const [text, comment, str, num, ident] = match
    if (comment) tokens.push({ kind: "comment", text })
    else if (str) tokens.push({ kind: "string", text })
    else if (num) tokens.push({ kind: "number", text })
    else if (ident) {
      const rest = line.slice(PATTERN.lastIndex)
      if (KEYWORDS.has(ident)) tokens.push({ kind: "keyword", text })
      else if (BUILTINS.has(ident)) tokens.push({ kind: "builtin", text })
      else if (/^\s*\(/.test(rest)) tokens.push({ kind: "function", text })
      else if (/^[A-Z][A-Z0-9_]+$/.test(ident)) tokens.push({ kind: "number", text })
      else tokens.push({ kind: "plain", text })
    } else tokens.push({ kind: "plain", text })
  }
  return tokens
}
