import { useT } from '../i18n'

export function ArtefactNotice() {
  const t = useT()
  return (
    <aside
      aria-label={t.artefact.eyebrow}
      className="mb-8 border-l-2 border-stamp bg-panel p-4"
    >
      <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-quiet">
        {t.artefact.eyebrow}
      </p>
      <p className="mt-2 max-w-prose text-sm leading-relaxed text-ink">{t.artefact.body}</p>
    </aside>
  )
}
