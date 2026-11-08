import { useT } from '../i18n'
import type { AnalyseResponse } from '../types'
import Gauge from './Gauge'

export type BandKey = 'low' | 'inconclusive' | 'high'

export function band(score: number): { key: BandKey; tone: string } {
  if (score < 0.4) return { key: 'low', tone: 'bg-ink/5 text-muted' }
  if (score <= 0.6) return { key: 'inconclusive', tone: 'bg-meter-mid/20 text-meter-high' }
  return { key: 'high', tone: 'bg-meter-high/10 text-meter-high' }
}

interface ScorePanelProps {
  result: AnalyseResponse
}

export default function ScorePanel({ result }: ScorePanelProps) {
  const t = useT()
  const pct = Math.round(result.score * 100)
  const b = band(result.score)
  const copy = t.score.bands[b.key]

  return (
    <section aria-label={t.score.region} className="rounded-2xl border border-line bg-panel p-6 sm:p-8">
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">{t.score.eyebrow}</p>

      <div className="mt-4 flex flex-wrap items-baseline gap-x-4 gap-y-1">
        <span className="font-mono text-6xl font-semibold tabular-nums leading-none">{pct}%</span>
        <span
          className={`inline-flex items-center rounded-full px-3 py-1 text-sm font-medium ${b.tone}`}
        >
          {copy.chip}
        </span>
      </div>

      <p className="mt-4 max-w-prose font-display text-xl leading-snug text-ink">{copy.sentence(pct)}</p>

      <div className="mt-7">
        <Gauge value={result.score} />
      </div>

      <p className="mt-7 border-t border-line pt-4 text-sm text-muted">{t.score.uncertainty}</p>
    </section>
  )
}
