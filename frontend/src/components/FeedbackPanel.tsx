import { useState } from 'react'
import { sendFeedback } from '../api'
import { PUBLIC_BUILD } from '../config'
import { useI18n, useT } from '../i18n'
import type { AnalyseResponse, FeedbackReason } from '../types'
import { band } from './ScorePanel'

const SCORE_REASONS: FeedbackReason[] = ['score_too_high', 'score_too_low', 'explanation_unclear']
const EVIDENCE_REASONS: FeedbackReason[] = ['evidence_irrelevant', 'evidence_missing']

interface FeedbackPanelProps {
  result: AnalyseResponse
}

export default function FeedbackPanel({ result }: FeedbackPanelProps) {
  const t = useT()
  const { locale } = useI18n()
  const [choice, setChoice] = useState<FeedbackReason | null>(null)
  const [state, setState] = useState<'idle' | 'sending' | 'sent' | 'failed'>('idle')

  const reasons: FeedbackReason[] = [
    ...EVIDENCE_REASONS,
    ...(PUBLIC_BUILD ? [] : SCORE_REASONS),
    'other',
  ]

  async function submit() {
    if (!choice) return
    setState('sending')
    try {
      await sendFeedback({
        reason: choice,
        band: PUBLIC_BUILD ? null : band(result.score).key,
        had_evidence: result.evidence.fact_checks.length > 0,
        locale,
      })
      setState('sent')
    } catch {
      setState('failed')
    }
  }

  return (
    <section
      aria-label={t.feedback.region}
      className="rounded-2xl border border-dashed border-line bg-paper p-6"
    >
      <p className="font-mono text-xs uppercase tracking-[0.2em] text-muted">{t.feedback.eyebrow}</p>

      {state === 'sent' ? (
        <p role="status" className="mt-3 text-sm text-ink">
          {t.feedback.thanks}
        </p>
      ) : (
        <>
          <fieldset className="mt-3 border-0 p-0">
            <legend className="font-display text-lg leading-snug text-ink">
              {t.feedback.question}
            </legend>

            <div className="mt-3 flex flex-col gap-2">
              {reasons.map((reason) => (
                <label key={reason} className="flex cursor-pointer items-start gap-2 text-sm text-ink">
                  <input
                    type="radio"
                    name="feedback-reason"
                    value={reason}
                    checked={choice === reason}
                    onChange={() => setChoice(reason)}
                    disabled={state === 'sending'}
                    className="mt-0.5 accent-meter-low"
                  />
                  <span>{t.feedback.reasons[reason]}</span>
                </label>
              ))}
            </div>
          </fieldset>

          <button
            type="button"
            onClick={submit}
            disabled={!choice || state === 'sending'}
            className="mt-4 border border-ink px-4 py-2 text-sm font-medium text-ink transition-colors hover:bg-ink hover:text-panel disabled:cursor-not-allowed disabled:border-line disabled:text-quiet disabled:hover:bg-transparent disabled:hover:text-quiet"
          >
            {t.feedback.submit}
          </button>

          {state === 'failed' && (
            <p role="alert" className="mt-3 text-sm text-stamp">
              {t.feedback.failed}
            </p>
          )}
        </>
      )}

      <p className="mt-4 text-xs italic text-muted">{t.feedback.privacy}</p>
    </section>
  )
}
