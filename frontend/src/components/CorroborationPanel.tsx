import { useT } from '../i18n'
import type { Corroboration } from '../types'

interface CorroborationPanelProps {
  corroboration: Corroboration
}

export default function CorroborationPanel({ corroboration }: CorroborationPanelProps) {
  const t = useT()
  const { checked } = corroboration
  const reported = checked.filter((c) => c.found)
  const notFound = checked.filter((c) => !c.found)

  return (
    <section
      aria-label={t.corroboration.region}
      className="flex h-full flex-col rounded-2xl border border-dashed border-line bg-paper p-6"
    >
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">{t.corroboration.eyebrow}</p>

      {checked.length === 0 ? (
        <p className="mt-3 text-sm text-muted">
          {t.corroboration.disabled}
        </p>
      ) : (
        <div className="mt-3 flex flex-col gap-3 text-sm">
          {reported.length > 0 && (
            <div>
              <p className="text-ink">{t.corroboration.foundIn}</p>
              <ul className="mt-1 flex flex-col gap-1">
                {reported.map((c) => (
                  <li key={c.name} className="flex items-center gap-2">
                    <span aria-hidden="true" className="inline-block h-2 w-2 rounded-full bg-meter-high" />
                    <a
                      href={c.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="font-medium text-meter-low underline underline-offset-2 hover:text-ink"
                    >
                      {c.name} ↗
                    </a>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {notFound.length > 0 && (
            <div>
              <p className="text-ink">
                {t.corroboration.notFoundIn}
              </p>
              <p className="mt-1 text-muted">{notFound.map((c) => c.name).join(', ')}.</p>
            </div>
          )}
        </div>
      )}

      <p className="mt-3 text-xs italic text-muted">
        {t.corroboration.caveat}
      </p>
    </section>
  )
}
