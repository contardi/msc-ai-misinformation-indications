import { useState, type DragEvent, type FormEvent } from 'react'
import { ACCEPTED_MEDIA, MAX_UPLOAD_BYTES } from '../config'
import { useT } from '../i18n'

interface InputPanelProps {
  onText: (text: string) => void
  onFile: (file: File) => void
  loading: boolean
  stage: 'idle' | 'text' | 'media'
}

export default function InputPanel({ onText, onFile, loading, stage }: InputPanelProps) {
  const [text, setText] = useState('')
  const [file, setFile] = useState<File | null>(null)
  const [localError, setLocalError] = useState<string | null>(null)
  const [dragging, setDragging] = useState(false)
  const t = useT()

  const ready = file !== null || text.trim().length > 0

  function accept(f: File) {
    setLocalError(null)
    if (f.size > MAX_UPLOAD_BYTES) {
      setLocalError(
        t.input.tooLarge((f.size / 1e6).toFixed(1), Math.round(MAX_UPLOAD_BYTES / 1e6)),
      )
      return
    }
    setFile(f)
    setText('')
  }

  function handleDrop(e: DragEvent) {
    e.preventDefault()
    setDragging(false)
    const f = e.dataTransfer.files?.[0]
    if (f) accept(f)
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!ready || loading) return
    if (file) onFile(file)
    else onText(text.trim())
  }

  return (
    <form onSubmit={handleSubmit} className="border border-line bg-panel">
      <div
        onDragOver={(e) => {
          e.preventDefault()
          setDragging(true)
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        className={`p-6 transition-colors sm:p-8 ${dragging ? 'bg-mark/5' : ''}`}
      >
        <label htmlFor="content" className="font-display text-xl leading-snug text-ink">
          {t.input.label}
        </label>

        {file ? (
          <div className="mt-4 flex items-center justify-between gap-4 border border-line bg-paper p-4">
            <div className="min-w-0">
              <p className="truncate font-mono text-sm text-ink">{file.name}</p>
              <p className="mt-0.5 font-mono text-[11px] uppercase tracking-wider text-quiet">
                {(file.size / 1e6).toFixed(1)} MB
              </p>
            </div>
            <button
              type="button"
              onClick={() => setFile(null)}
              className="shrink-0 font-mono text-xs uppercase tracking-wider text-quiet underline underline-offset-4 hover:text-ink"
            >
              {t.input.remove}
            </button>
          </div>
        ) : (
          <textarea
            id="content"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={5}
            placeholder={t.input.placeholder}
            className="mt-4 w-full resize-y border border-line bg-paper p-4 text-ink placeholder:text-quiet/70 focus:border-stamp focus:outline-none"
          />
        )}

        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-2">
          <input
            id="media"
            type="file"
            accept={ACCEPTED_MEDIA}
            className="peer sr-only"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) accept(f)
              e.target.value = ''
            }}
          />
          <label
            htmlFor="media"
            className="cursor-pointer font-mono text-xs uppercase tracking-wider text-stamp underline underline-offset-4 hover:text-ink peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-4 peer-focus-visible:outline-stamp"
          >
            {t.input.chooseFile}
          </label>
          <p className="font-mono text-[11px] uppercase tracking-wider text-quiet">
            mp3 · m4a · ogg · wav · mp4 · webm · mov
          </p>
        </div>

        {localError && (
          <p role="alert" className="mt-3 text-sm text-stamp">
            {localError}
          </p>
        )}
      </div>

      <div className="flex flex-col gap-3 border-t border-line px-6 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-8">
        <p className="max-w-prose text-xs text-quiet">
          {t.input.privacy}
        </p>
        <button
          type="submit"
          disabled={!ready || loading}
          className="shrink-0 bg-ink px-6 py-3 font-mono text-xs uppercase tracking-[0.16em] text-paper transition-opacity disabled:opacity-40"
        >
          {loading ? t.input.submitting : t.input.submit}
        </button>
      </div>

      {loading && (
        <div
          role="status"
          aria-live="polite"
          className="border-t border-line bg-paper px-6 py-4 sm:px-8"
        >
          <p className="font-mono text-xs uppercase tracking-wider text-ink">
            {stage === 'media' ? t.input.workingMedia : t.input.workingText}
          </p>
          {stage === 'media' && (
            <>
              <p className="mt-1 text-sm text-quiet">
                {t.input.mediaSlow}
              </p>
              <div
                aria-hidden="true"
                className="mt-3 h-px w-full overflow-hidden bg-line"
              >
                <div className="h-full w-1/3 animate-[slide_1.6s_ease-in-out_infinite] bg-stamp" />
              </div>
            </>
          )}
        </div>
      )}
    </form>
  )
}
