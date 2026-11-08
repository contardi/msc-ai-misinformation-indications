import { useState } from 'react'
import { I18nContext, useLocaleState } from './i18n'
import LanguageSwitcher from './components/LanguageSwitcher'
import { analyse, analyseMedia, ApiError } from './api'
import { PUBLIC_BUILD } from './config'
import { ArtefactNotice } from './components/ArtefactNotice'
import type { Dictionary } from './i18n/pt-BR'
import type { AnalyseResponse } from './types'
import { useHistory, type HistoryEntry } from './hooks/useHistory'
import InputPanel from './components/InputPanel'
import TranscriptPanel from './components/TranscriptPanel'
import EvidencePanel from './components/EvidencePanel'
import FeedbackPanel from './components/FeedbackPanel'
import ScorePanel from './components/ScorePanel'
import ExplanationPanel from './components/ExplanationPanel'
import CorroborationPanel from './components/CorroborationPanel'
import IndicatorsPanel from './components/IndicatorsPanel'
import ResultCascade from './components/ResultCascade'
import HistoryPanel from './components/HistoryPanel'

function messageFor(err: unknown, t: Dictionary): string {
  if (err instanceof ApiError) {
    if (err.serverDetail) return err.serverDetail
    if (err.key === 'failed') return t.errors.failed(err.status ?? 0)
    return t.errors[err.key]
  }
  return t.errors.unexpected
}

export default function App() {
  const [result, setResult] = useState<AnalyseResponse | null>(null)
  const [stage, setStage] = useState<'idle' | 'text' | 'media'>('idle')
  const [error, setError] = useState<string | null>(null)
  const { entries, add, clear } = useHistory()
  const { locale, setLocale, t } = useLocaleState()
  const loading = stage !== 'idle'

  async function run(kind: 'text' | 'media', fn: () => Promise<AnalyseResponse>, label: string) {
    setStage(kind)
    setError(null)
    try {
      const data = await fn()
      setResult(data)
      add(label, data)
    } catch (err) {
      setError(messageFor(err, t))
      setResult(null)
    } finally {
      setStage('idle')
    }
  }

  function showFromHistory(entry: HistoryEntry) {
    setError(null)
    setResult(entry.result)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <I18nContext.Provider value={{ locale, setLocale, t }}>
    <div className="min-h-screen">
      <div className="mx-auto max-w-3xl px-4 py-10 sm:py-16">
        <header className="mb-10">
          <div className="flex items-start justify-between gap-4">
            <p className="font-mono text-[11px] uppercase tracking-[0.25em] text-quiet">
              {t.header.eyebrow}
            </p>
            <LanguageSwitcher />
          </div>
          <h1 className="mt-4 font-display text-5xl leading-[1.05] text-ink sm:text-6xl">
            {t.header.title}
          </h1>
          <p className="mt-5 max-w-prose text-lg leading-relaxed text-quiet">{t.header.lede}</p>
        </header>

        <ArtefactNotice />

        <InputPanel
          onText={(text) => run('text', () => analyse(text), text)}
          onFile={(file) => run('media', () => analyseMedia(file), file.name)}
          loading={loading}
          stage={stage}
        />

        {error && (
          <p
            role="alert"
            className="mt-6 border-l-2 border-stamp bg-panel p-4 text-sm text-ink"
          >
            {error}
          </p>
        )}

        {result && (
          <main className="mt-8 flex flex-col gap-4">
            <TranscriptPanel source={result.source} />

            <ResultCascade result={result} />

            <EvidencePanel evidence={result.evidence} />

            <CorroborationPanel corroboration={result.corroboration} />

            <IndicatorsPanel
              indicators={result.indicators}
              coverage={result.indicator_coverage}
            />

            {!PUBLIC_BUILD && (
              <>
                <ScorePanel result={result} />
                <ExplanationPanel explanation={result.explanation} score={result.score} />
              </>
            )}

            <FeedbackPanel key={result.explanation.text} result={result} />
          </main>
        )}

        <HistoryPanel entries={entries} onSelect={showFromHistory} onClear={clear} />

        <footer className="mt-16 border-t border-line pt-6 text-xs leading-relaxed text-quiet">
          <p>{t.footer.purpose}</p>
          <p className="mt-2">{t.footer.academic}</p>
        </footer>
      </div>
    </div>
    </I18nContext.Provider>
  )
}
