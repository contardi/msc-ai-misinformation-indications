import { render } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { I18nContext, LOCALES } from '../i18n'
import EvidencePanel from './EvidencePanel'
import VerdictStamp from './VerdictStamp'
import type { FactCheck } from '../types'

function check(over: Partial<FactCheck> = {}): FactCheck {
  return {
    claim: 'Os gastos militares globais cresceram pelo nono ano consecutivo',
    rating: 'Verdadeiro',
    publisher: 'Agência Lupa',
    url: 'https://lupa.uol.com.br/exemplo',
    verdict: 'true',
    ...over,
  }
}

function panel(checks: FactCheck[]) {
  return render(
    <I18nContext.Provider value={{ locale: 'pt-BR', t: LOCALES['pt-BR'], setLocale: () => {} }}>
      <EvidencePanel evidence={{ fact_checks: checks, note: '' }} />
    </I18nContext.Provider>,
  )
}

const stampOf = (c: HTMLElement) => c.querySelector('span.inline-flex') as HTMLElement

describe('a true verdict is not styled as a debunk', () => {
  it('gives verdadeiro the affirm tone, not the stamp tone', () => {
    const { container } = panel([check()])
    const cls = stampOf(container).className
    expect(cls).toContain('bg-affirm')
    expect(cls).not.toContain('bg-stamp')
  })

  it('still stamps a real debunk in the stamp tone', () => {
    const { container } = panel([check({ rating: 'Falso', verdict: 'false' })])
    expect(stampOf(container).className).toContain('bg-stamp')
  })

  it('treats "verdadeiro, mas..." as partial rather than true', () => {
    const { container } = panel([check({ rating: 'Verdadeiro, mas...', verdict: 'partial' })])
    const cls = stampOf(container).className
    expect(cls).toContain('bg-mark')
    expect(cls).not.toContain('bg-affirm')
  })
})

describe('an unreadable rating is neutral, never a debunk', () => {
  it('renders a blank rating as unrated without the debunk colour', () => {
    const { container, getByText } = panel([check({ rating: '', verdict: 'unknown' })])
    const cls = stampOf(container).className
    expect(cls).not.toContain('bg-stamp')
    expect(cls).toContain('bg-transparent')
    getByText('sem classificação')
  })

  it('falls back to neutral when the server sends a verdict it does not know', () => {
    const { container } = panel([check({ rating: 'Algo novo', verdict: 'a_new_verdict' })])
    expect(stampOf(container).className).not.toContain('bg-stamp')
  })
})

describe('an over-long rating cannot break the layout', () => {
  const long =
    'É falso o vídeo que circula em grupos de whatsapp, que mostra o governador dançando ao som ' +
    'de uma música de religião de matriz afrobrasileira, pois no conteúdo original ele dança com ' +
    'moradores da comunidade quilombola Kalunga'

  it('truncates the label and keeps the full text reachable', () => {
    const { container } = panel([check({ rating: long, verdict: 'false' })])
    const stamp = stampOf(container)
    expect(stamp.textContent!.length).toBeLessThanOrEqual(41)
    expect(stamp.getAttribute('title')).toBe(long)
  })

  it('does not pin the stamp against shrinking', () => {
    const { container } = panel([check({ rating: long, verdict: 'false' })])
    expect(stampOf(container).className).not.toContain('shrink-0')
  })
})

describe('the page still may not assert truth in its own voice', () => {
  it('shows an affirmative verdict only as an attributed quotation', () => {
    const { container, getByText } = panel([check()])
    expect(container.textContent).not.toMatch(/\bé verdadeiro\b/i)
    getByText('Agência Lupa')
  })

  it('renders the empty state without any verdict colour', () => {
    const { container } = render(
      <I18nContext.Provider value={{ locale: 'pt-BR', t: LOCALES['pt-BR'], setLocale: () => {} }}>
        <VerdictStamp label="nada encontrado" tone="absent" />
      </I18nContext.Provider>,
    )
    const cls = stampOf(container).className
    expect(cls).toContain('bg-transparent')
    expect(cls).not.toMatch(/bg-(stamp|mark|affirm)/)
  })
})
