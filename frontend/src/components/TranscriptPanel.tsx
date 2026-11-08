import { useT } from '../i18n'
import type { SourceInfo } from '../types'

interface TranscriptPanelProps {
  source: SourceInfo
}

export default function TranscriptPanel({ source }: TranscriptPanelProps) {
  const t = useT()
  if (source.kind === 'text' || !source.extracted_text) return null

  const minutes = Math.round(source.duration_seconds / 60)

  return (
    <section aria-label={t.transcript.region} className="border border-line bg-panel p-6 sm:p-8">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <p className="font-mono text-[11px] uppercase tracking-[0.22em] text-quiet">
          {t.transcript.eyebrow}
        </p>
        <p className="font-mono text-[11px] uppercase tracking-wider text-quiet">
          {source.kind === 'video' ? t.transcript.video : t.transcript.audio}
          {minutes > 0 && ` · ${minutes} min`}
        </p>
      </div>

      {source.notes.map((note) => (
        <p key={note} className="mt-3 border-l-2 border-mark pl-3 text-sm text-ink">
          {note}
        </p>
      ))}

      <p
        tabIndex={0}
        className="mt-4 max-h-64 overflow-y-auto whitespace-pre-wrap font-mono text-sm leading-relaxed text-ink focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-stamp"
      >
        {source.extracted_text}
      </p>

      <p className="mt-4 border-t border-line pt-4 text-sm text-quiet">
        {t.transcript.caveat}
      </p>
    </section>
  )
}
