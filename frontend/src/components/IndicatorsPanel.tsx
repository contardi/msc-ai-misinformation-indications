import { useT } from '../i18n'
import type { Dictionary } from '../i18n/pt-BR'
import type { Indicator, IndicatorCoverage } from '../types'

function copyFor(t: Dictionary, ind: Indicator): { label: string; detail: string } {
  const cues = t.indicators.cues
  switch (ind.kind) {
    case 'exclamations':
      return { label: cues.exclamations.label, detail: cues.exclamations.detail(ind.count ?? 0) }
    case 'sensational_terms':
      return {
        label: cues.sensational_terms.label,
        detail: cues.sensational_terms.detail(ind.terms ?? []),
      }
    default:
      return { label: ind.label, detail: ind.detail }
  }
}

function highlight(excerpt: string, match: string) {
  if (!match || !excerpt.includes(match)) return excerpt
  return excerpt.split(match).flatMap((part, i) =>
    i === 0 ? [part] : [<strong key={i} className="font-semibold text-ink">{match}</strong>, part],
  )
}

interface IndicatorsPanelProps {
  indicators: Indicator[]
  coverage?: IndicatorCoverage | null
}

export default function IndicatorsPanel({ indicators, coverage }: IndicatorsPanelProps) {
  const t = useT()
  const found = indicators.length > 0

  return (
    <section
      aria-label={t.indicators.region}
      className="flex h-full flex-col rounded-2xl border border-dashed border-line bg-paper p-6"
    >
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">{t.indicators.eyebrow}</p>

      {found ? (
        <>
          <ul className="mt-3 flex flex-col gap-4">
            {indicators.map((ind, i) => {
              const copy = copyFor(t, ind)
              return (
                <li key={`${ind.kind || ind.label}-${i}`} className="text-sm">
                  <p>
                    <span className="font-medium text-ink">{copy.label}</span>
                    <span className="text-muted"> - {copy.detail}</span>
                  </p>

                  {(ind.hits ?? []).length > 0 && (
                    <ul className="mt-1.5 flex flex-col gap-1">
                      {(ind.hits ?? []).map((hit, h) => (
                        <li
                          key={h}
                          className="border-l-2 border-line pl-3 font-mono text-xs leading-relaxed text-quiet"
                        >
                          {highlight(hit.excerpt, hit.match)}
                        </li>
                      ))}
                    </ul>
                  )}

                  {ind.evidence && (
                    <p className="mt-1.5 text-xs text-quiet">
                      {t.indicators.evidence(
                        ind.evidence.fake_rate,
                        ind.evidence.real_rate,
                        ind.evidence.corpus,
                      )}
                    </p>
                  )}
                </li>
              )
            })}
          </ul>

          <p className="mt-4 text-xs italic text-muted">{t.indicators.caveat}</p>
        </>
      ) : (
        <>
          <p className="mt-3 text-sm font-medium text-ink">{t.indicators.none.title}</p>
          {coverage && (
            <p className="mt-2 text-sm text-muted">{t.indicators.none.limit(coverage.one_in)}</p>
          )}
          <p className="mt-3 text-xs italic text-muted">{t.indicators.none.next}</p>
        </>
      )}
    </section>
  )
}
