import { useT } from '../i18n'
import type { Explanation } from '../types'

interface ExplanationPanelProps {
  explanation: Explanation
  score: number
}

export default function ExplanationPanel({ explanation, score }: ExplanationPanelProps) {
  const t = useT()
  const pct = Math.round(score * 100)
  const terms = explanation.top_tokens.slice(0, 5).map((tok) => tok.token).filter(Boolean)
  const maxWeight = explanation.top_tokens.reduce((max, t) => Math.max(max, Math.abs(t.weight)), 0)

  return (
    <section
      aria-label={t.explanationPanel.region}
      className="flex h-full flex-col rounded-2xl border border-line bg-panel p-6"
    >
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">{t.explanationPanel.eyebrow}</p>

      <p className="mt-3 text-ink">{t.explanationPanel.body(pct, terms)}</p>

      {explanation.top_tokens.length > 0 && (
        <div className="mt-5">
          <h3 className="font-mono text-[11px] uppercase tracking-wider text-muted">
            {t.explanationPanel.topTokens}
          </h3>
          <ul className="mt-3 flex flex-col gap-2">
            {explanation.top_tokens.map((t, i) => {
              const width = maxWeight > 0 ? (Math.abs(t.weight) / maxWeight) * 100 : 0
              const towardMisinfo = t.weight >= 0
              return (
                <li key={`${t.token}-${i}`} className="flex items-center gap-3">
                  <span className="w-28 shrink-0 truncate font-mono text-sm text-ink" title={t.token}>
                    {t.token}
                  </span>
                  <span className="h-2 flex-1 overflow-hidden rounded-full bg-ink/5">
                    <span
                      className={`block h-full rounded-full ${towardMisinfo ? 'bg-meter-high' : 'bg-meter-low'}`}
                      style={{ width: `${width}%` }}
                    />
                  </span>
                  <span className="w-12 shrink-0 text-right font-mono text-xs tabular-nums text-muted">
                    {t.weight.toFixed(2)}
                  </span>
                </li>
              )
            })}
          </ul>
          <p className="mt-3 text-xs italic text-muted">
            {t.explanationPanel.shapNote}
          </p>
        </div>
      )}
    </section>
  )
}
