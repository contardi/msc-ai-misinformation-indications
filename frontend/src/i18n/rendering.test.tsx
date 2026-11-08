import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { I18nContext, LOCALES } from './index'
import type { LocaleCode } from './index'
import ExplanationPanel from '../components/ExplanationPanel'
import IndicatorsPanel from '../components/IndicatorsPanel'
import ScorePanel from '../components/ScorePanel'
import { makeResponse } from '../test/fixtures'
import type { AnalyseResponse } from '../types'

function withLocale(locale: LocaleCode, ui: React.ReactNode) {
  return render(
    <I18nContext.Provider value={{ locale, t: LOCALES[locale], setLocale: () => {} }}>
      {ui}
    </I18nContext.Provider>,
  )
}

function serverPortuguese(): AnalyseResponse {
  return makeResponse({
    score: 0.74,
    uncertainty_note: 'AVISO-DO-SERVIDOR-EM-PORTUGUES',
    explanation: {
      text: 'TEXTO-DO-SERVIDOR-EM-PORTUGUES',
      top_tokens: [
        { token: 'urgente', weight: 0.42 },
        { token: 'segredo', weight: 0.31 },
      ],
    },
    indicators: [
      { kind: 'exclamations', label: 'ROTULO-SERVIDOR', detail: 'DETALHE-SERVIDOR', count: 7, terms: [] },
      {
        kind: 'sensational_terms',
        label: 'ROTULO-SERVIDOR-2',
        detail: 'DETALHE-SERVIDOR-2',
        count: null,
        terms: ['atenção', 'urgente'],
      },
    ],
  })
}

describe('en-GB renders English, not the server’s Portuguese', () => {
  it('composes the explanation itself rather than echoing explanation.text', () => {
    const result = serverPortuguese()
    const { container } = withLocale('en-GB', (
      <ExplanationPanel explanation={result.explanation} score={result.score} />
    ))
    const text = container.textContent ?? ''

    expect(text).not.toContain('TEXTO-DO-SERVIDOR-EM-PORTUGUES')
    expect(text).toMatch(/74% likelihood that this text contains misinformation/i)
    expect(text).toContain('urgente')
  })

  it('composes each style cue from its kind and its variable', () => {
    const { container } = withLocale('en-GB', (
      <IndicatorsPanel indicators={serverPortuguese().indicators} />
    ))
    const text = container.textContent ?? ''

    expect(text).not.toMatch(/ROTULO-SERVIDOR|DETALHE-SERVIDOR/)
    expect(text).toContain('Sensationalist punctuation')
    expect(text).toContain('7 exclamation marks.')
    expect(text).toContain('Urgency wording')
    expect(text).toContain('atenção')
  })

  it('shows the uncertainty note in English rather than the server field', () => {
    const { container } = withLocale('en-GB', <ScorePanel result={serverPortuguese()} />)
    const text = container.textContent ?? ''

    expect(text).not.toContain('AVISO-DO-SERVIDOR-EM-PORTUGUES')
    expect(text).toContain('not a verdict')
  })
})

describe('pt-BR still renders Brazilian Portuguese', () => {
  it('says the same things in the reference language', () => {
    const result = serverPortuguese()
    const { container } = withLocale('pt-BR', (
      <ExplanationPanel explanation={result.explanation} score={result.score} />
    ))
    expect(container.textContent ?? '').toMatch(
      /O modelo estima 74% de probabilidade de o texto conter desinformação/,
    )
  })

  it('keeps the responsible-AI hedge at every locale', () => {
    for (const code of Object.keys(LOCALES) as LocaleCode[]) {
      const { container, unmount } = withLocale(code, <ScorePanel result={serverPortuguese()} />)
      expect(container.textContent ?? '', `${code} lost the uncertainty note`).toMatch(
        /não um veredito|not a verdict/i,
      )
      unmount()
    }
  })
})

describe('an unknown cue degrades rather than disappears', () => {
  it('falls back to the server’s own wording', () => {
    const { container } = withLocale('en-GB', (
      <IndicatorsPanel
        indicators={[
          { kind: 'a_cue_from_a_newer_backend', label: 'Rótulo novo', detail: 'Detalhe novo.' },
        ]}
      />
    ))
    expect(container.textContent ?? '').toContain('Rótulo novo')
    expect(container.textContent ?? '').toContain('Detalhe novo.')
  })
})
