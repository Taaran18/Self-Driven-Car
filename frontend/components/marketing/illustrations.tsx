const RAY_ANGLES = [0, 45, 90, 135, 180, 225, 270, 315]

export function SensorIllustration({ className }: { className?: string }) {
  const lengths = [150, 118, 64, 90, 150, 96, 70, 104]
  return (
    <svg viewBox="0 0 320 320" className={className} aria-hidden>
      <path
        d="M96 0 C 80 110, 140 200, 100 320 L 236 320 C 272 210, 208 110, 228 0 Z"
        fill="var(--track-road)"
      />
      <path
        d="M96 0 C 80 110, 140 200, 100 320"
        fill="none"
        stroke="var(--track-edge)"
        strokeWidth="3"
      />
      <path
        d="M228 0 C 208 110, 272 210, 236 320"
        fill="none"
        stroke="var(--track-edge)"
        strokeWidth="3"
      />
      {RAY_ANGLES.map((angle, i) => {
        const rad = (angle * Math.PI) / 180
        const x = 166 + Math.sin(rad) * lengths[i]
        const y = 176 - Math.cos(rad) * lengths[i]
        const close = lengths[i] < 80
        const color = close
          ? "var(--danger)"
          : lengths[i] < 110
            ? "var(--warning)"
            : "var(--success)"
        return (
          <g key={angle}>
            <line
              x1="166"
              y1="176"
              x2={x}
              y2={y}
              stroke={color}
              strokeWidth="2"
              strokeDasharray="4 4"
            />
            {lengths[i] < 150 ? <circle cx={x} cy={y} r="5" fill={color} /> : null}
          </g>
        )
      })}
      <rect x="148" y="146" width="36" height="62" rx="12" fill="var(--primary)" />
      <rect x="154" y="158" width="24" height="14" rx="4" fill="var(--primary-fg)" opacity="0.8" />
    </svg>
  )
}

export function NetworkIllustration({ className }: { className?: string }) {
  const inputs = Array.from({ length: 9 }, (_, i) => 30 + i * 32)
  const hidden = [110, 190]
  const outputs = [80, 140, 200, 260].map((y) => y + 8)
  return (
    <svg viewBox="0 0 320 320" className={className} aria-hidden>
      {inputs.map((y1, i) =>
        outputs.map((y2, j) => (
          <line
            key={`${i}-${j}`}
            x1="50"
            y1={y1}
            x2="270"
            y2={y2}
            stroke={(i + j) % 3 === 0 ? "var(--danger)" : "var(--chart-1)"}
            strokeOpacity={0.18 + ((i * 7 + j * 3) % 5) * 0.08}
            strokeWidth={1 + ((i + j) % 3)}
          />
        )),
      )}
      {hidden.map((y) => (
        <g key={y}>
          <line
            x1="50"
            y1={inputs[2]}
            x2="160"
            y2={y}
            stroke="var(--violet)"
            strokeOpacity="0.6"
            strokeWidth="2"
          />
          <line
            x1="160"
            y1={y}
            x2="270"
            y2={outputs[1]}
            stroke="var(--violet)"
            strokeOpacity="0.6"
            strokeWidth="2"
          />
          <circle
            cx="160"
            cy={y}
            r="11"
            fill="var(--surface)"
            stroke="var(--violet)"
            strokeWidth="3"
          />
        </g>
      ))}
      {inputs.map((y, i) => (
        <circle
          key={y}
          cx="50"
          cy={y}
          r="9"
          fill={i % 3 === 0 ? "var(--success)" : "var(--surface)"}
          stroke="var(--success)"
          strokeWidth="3"
        />
      ))}
      {outputs.map((y, i) => (
        <circle
          key={y}
          cx="270"
          cy={y}
          r="12"
          fill={i === 0 || i === 3 ? "var(--warning)" : "var(--surface)"}
          stroke="var(--warning)"
          strokeWidth="3"
        />
      ))}
    </svg>
  )
}

export function DecideIllustration({ className }: { className?: string }) {
  const bars = [
    { label: "Accelerate", value: 0.86, on: true },
    { label: "Brake", value: 0.12, on: false },
    { label: "Turn Left", value: 0.31, on: false },
    { label: "Turn Right", value: 0.74, on: true },
  ]
  return (
    <svg viewBox="0 0 320 320" className={className} aria-hidden>
      {bars.map((bar, i) => {
        const y = 40 + i * 66
        return (
          <g key={bar.label}>
            <text x="24" y={y} className="fill-fg-muted text-[14px] font-semibold">
              {bar.label}
            </text>
            <rect x="24" y={y + 12} width="272" height="16" rx="8" fill="var(--surface-3)" />
            <rect
              x="24"
              y={y + 12}
              width={272 * bar.value}
              height="16"
              rx="8"
              fill={bar.on ? "var(--success)" : "var(--border-strong)"}
            />
            <line
              x1={24 + 272 * 0.75}
              x2={24 + 272 * 0.75}
              y1={y + 6}
              y2={y + 34}
              stroke="var(--fg)"
              strokeWidth="2"
            />
          </g>
        )
      })}
      <text x={24 + 272 * 0.75} y="310" textAnchor="middle" className="fill-fg-subtle text-[12px]">
        threshold 0.5
      </text>
    </svg>
  )
}

export function MoveIllustration({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 320 320" className={className} aria-hidden>
      <path
        d="M60 300 C 90 200, 200 180, 240 40"
        fill="none"
        stroke="var(--border-strong)"
        strokeWidth="3"
        strokeDasharray="6 8"
      />
      {[0.2, 0.45, 0.7].map((t, i) => {
        const x = 60 + (240 - 60) * t + Math.sin(t * Math.PI) * 30
        const y = 300 - (300 - 40) * t
        return (
          <rect
            key={t}
            x={x - 14}
            y={y - 22}
            width="28"
            height="46"
            rx="9"
            fill="var(--primary)"
            opacity={0.3 + i * 0.3}
            transform={`rotate(${20 + i * 8} ${x} ${y})`}
          />
        )
      })}
      <path
        d="M232 70 L 248 30 L 262 72"
        fill="none"
        stroke="var(--primary)"
        strokeWidth="4"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <text x="40" y="60" className="fill-fg-muted text-[14px] font-semibold">
        heading +2° per turn
      </text>
      <text x="40" y="84" className="fill-fg-subtle text-[13px]">
        speed ≤ 10 per tick
      </text>
    </svg>
  )
}

export function ScoreIllustration({ className }: { className?: string }) {
  return (
    <svg viewBox="0 0 320 320" className={className} aria-hidden>
      <rect x="40" y="40" width="240" height="18" rx="9" fill="var(--surface-3)" />
      <rect x="40" y="40" width="180" height="18" rx="9" fill="var(--success)" />
      <text x="40" y="30" className="fill-fg-muted text-[14px] font-semibold">
        fitness grows with distance
      </text>
      {[
        { y: 110, label: "Hit a wall", out: true },
        { y: 160, label: "Fell behind", out: false },
        { y: 210, label: "Drove backward", out: false },
        { y: 260, label: "Stalled", out: false },
      ].map((row) => (
        <g key={row.label}>
          <rect
            x="40"
            y={row.y - 22}
            width="240"
            height="38"
            rx="10"
            fill={row.out ? "var(--danger-soft)" : "var(--surface-2)"}
          />
          <text
            x="56"
            y={row.y + 2}
            className={
              row.out ? "fill-danger text-[14px] font-semibold" : "fill-fg-muted text-[14px]"
            }
          >
            {row.label}
          </text>
          <text
            x="264"
            y={row.y + 2}
            textAnchor="end"
            className={
              row.out ? "fill-danger text-[13px] font-bold" : "fill-fg-subtle text-[13px] font-bold"
            }
          >
            {row.out ? "OUT" : "OK"}
          </text>
        </g>
      ))}
    </svg>
  )
}
