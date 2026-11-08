type Tone = 'false' | 'partial' | 'true' | 'unknown' | 'absent'

const MAX_LABEL = 40

const TONE_CLASS: Record<Tone, string> = {
  false: 'border-stamp bg-stamp text-panel',
  partial: 'border-mark bg-mark text-panel',
  true: 'border-affirm bg-affirm text-panel',
  unknown: 'border-quiet/50 bg-transparent text-quiet',
  absent: 'border-quiet/50 bg-transparent text-quiet',
}

interface VerdictStampProps {
  label: string
  tone?: string
}

export default function VerdictStamp({ label, tone }: VerdictStampProps) {
  const t: Tone = tone && tone in TONE_CLASS ? (tone as Tone) : 'unknown'
  const text = label.length > MAX_LABEL ? `${label.slice(0, MAX_LABEL).trimEnd()}…` : label
  return (
    <span
      title={text === label ? undefined : label}
      className={`inline-flex max-w-full items-center break-words border px-3 py-1.5 font-mono text-sm font-semibold uppercase leading-none tracking-[0.12em] ${TONE_CLASS[t]}`}
    >
      {text}
    </span>
  )
}
