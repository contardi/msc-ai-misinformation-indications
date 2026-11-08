import { useT } from '../i18n'
import type { Evidence } from '../types'
import VerdictStamp from './VerdictStamp'

interface EvidencePanelProps {
  evidence: Evidence
}

export default function EvidencePanel({ evidence }: EvidencePanelProps) {
  const t = useT()
  const checks = evidence.fact_checks

  if (checks.length === 0) {
    return (
      <section aria-label={t.evidence.eyebrow} className="border border-line bg-panel p-6 sm:p-8">
        <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-quiet">
          {t.evidence.eyebrow}
        </p>
        <div className="mt-5 flex flex-col gap-4 sm:flex-row sm:items-center">
          <VerdictStamp label={t.evidence.none} tone="absent" />
          <p className="max-w-prose font-display text-lg leading-snug text-ink">
            {t.evidence.noneTitle}{' '}
            <strong className="font-semibold">{t.evidence.noneEmphasis}</strong>
          </p>
        </div>
        <p className="mt-4 max-w-prose text-sm text-quiet">
          {t.evidence.noneDetail}
        </p>
      </section>
    )
  }

  return (
    <section aria-label={t.evidence.eyebrow} className="border border-line bg-panel p-6 sm:p-8">
      <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-quiet">
        {t.evidence.eyebrow}
        <span className="ml-2 text-ink">{checks.length}</span>
      </p>
      <p className="mt-3 max-w-prose font-display text-lg leading-snug text-ink">
        {t.evidence.foundTitle}
      </p>

      <ul className="mt-6 flex flex-col">
        {checks.map((fc, i) => (
          <li
            key={`${fc.url}-${i}`}
            className="flex flex-col gap-3 border-t border-line py-5 sm:flex-row sm:gap-5"
          >
            <div className="min-w-0 sm:w-40 sm:shrink-0">
              <VerdictStamp
                label={fc.rating || t.evidence.unrated}
                tone={fc.rating ? fc.verdict : 'unknown'}
              />
              <p className="mt-2 font-mono text-[11px] uppercase tracking-wider text-quiet">
                {fc.publisher}
              </p>
            </div>
            <div className="min-w-0">
              <p className="text-ink">{fc.claim}</p>
              <a
                href={fc.url}
                target="_blank"
                rel="noopener noreferrer"
                className="mt-2 inline-block font-mono text-xs uppercase tracking-wider text-stamp underline underline-offset-4 hover:text-ink"
              >
                {t.evidence.read}
              </a>
            </div>
          </li>
        ))}
      </ul>

      <p className="mt-5 border-t border-line pt-4 text-sm text-quiet">
        {t.evidence.attribution}
      </p>
    </section>
  )
}
