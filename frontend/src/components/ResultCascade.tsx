import { useT } from '../i18n'
import { PUBLIC_BUILD } from '../config'
import type { AnalyseResponse } from '../types'

export type Tier = 'evidence' | 'indicators' | 'classifier' | 'none'

export function tierOf(result: AnalyseResponse): Tier {
  if (result.evidence.fact_checks.length > 0) return 'evidence'
  if (result.indicators.length > 0) return 'indicators'
  return PUBLIC_BUILD ? 'none' : 'classifier'
}

interface ResultCascadeProps {
  result: AnalyseResponse
}

export default function ResultCascade({ result }: ResultCascadeProps) {
  const t = useT()
  const tier = tierOf(result)
  const copy = t.cascade.tiers[tier]

  return (
    <section
      aria-label={t.cascade.region}
      className="rounded-2xl border border-line bg-panel p-6 sm:p-8"
    >
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">{t.cascade.eyebrow}</p>
      <p className="mt-3 font-display text-xl leading-snug text-ink">{copy.headline}</p>
      <p className="mt-2 max-w-prose text-sm leading-relaxed text-quiet">{copy.detail}</p>

      {'nextSteps' in copy && (
        <div className="mt-4 border-t border-line pt-4">
          <p className="text-xs font-medium uppercase tracking-wide text-muted">
            {t.cascade.nextStepsLabel}
          </p>
          <ul className="mt-2 flex list-disc flex-col gap-1 pl-5 text-sm text-quiet">
            {copy.nextSteps.map((step: string, i: number) => (
              <li key={i}>{step}</li>
            ))}
          </ul>
        </div>
      )}
    </section>
  )
}
