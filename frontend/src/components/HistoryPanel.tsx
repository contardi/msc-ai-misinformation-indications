import { PUBLIC_BUILD } from '../config'
import { useI18n } from '../i18n'
import type { LocaleCode } from '../i18n'
import type { HistoryEntry } from '../hooks/useHistory'

interface HistoryPanelProps {
  entries: HistoryEntry[]
  onSelect: (entry: HistoryEntry) => void
  onClear: () => void
}

function formatTime(at: number, locale: LocaleCode): string {
  try {
    return new Date(at).toLocaleString(locale, { dateStyle: 'short', timeStyle: 'short' })
  } catch {
    return ''
  }
}

export default function HistoryPanel({ entries, onSelect, onClear }: HistoryPanelProps) {
  const { t, locale } = useI18n()
  if (entries.length === 0) return null

  return (
    <section aria-label={t.history.region} className="mt-12">
      <div className="flex items-center justify-between gap-4">
        <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">
          {t.history.eyebrow}
        </p>
        <button
          type="button"
          onClick={onClear}
          className="text-xs font-medium text-muted underline underline-offset-2 hover:text-meter-high"
        >
          {t.history.clear}
        </button>
      </div>

      <ul className="mt-4 flex flex-col divide-y divide-line overflow-hidden rounded-2xl border border-line bg-panel">
        {entries.map((entry) => {
          const pct = Math.round(entry.result.score * 100)
          const found = entry.result.evidence?.fact_checks?.length ?? 0
          return (
            <li key={entry.id}>
              <button
                type="button"
                onClick={() => onSelect(entry)}
                aria-label={`${t.history.replay}: ${entry.preview}`}
                className="flex w-full items-center gap-4 p-4 text-left transition hover:bg-paper"
              >
                <span className="w-12 shrink-0 text-center font-mono text-lg font-semibold tabular-nums text-ink">
                  {PUBLIC_BUILD ? found : `${pct}%`}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-ink">{entry.preview}</span>
                  <span className="font-mono text-[11px] text-muted">{formatTime(entry.at, locale)}</span>
                </span>
                <span aria-hidden="true" className="shrink-0 text-muted">
                  ↺
                </span>
              </button>
            </li>
          )
        })}
      </ul>

      <p className="mt-3 text-xs text-muted">
        {t.history.note}
      </p>
    </section>
  )
}
