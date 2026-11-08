import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import { I18nContext } from '../i18n'
import { ptBR } from '../i18n/pt-BR'
import { enGB } from '../i18n/en-GB'
import { ArtefactNotice } from './ArtefactNotice'

function renderIn(dict: typeof ptBR) {
  return render(
    <I18nContext.Provider value={{ locale: dict.meta.code as 'pt-BR', t: dict, setLocale: () => {} }}>
      <ArtefactNotice />
    </I18nContext.Provider>,
  )
}

describe('ArtefactNotice', () => {
  it('states it is an academic artefact and not a service, in pt-BR', () => {
    renderIn(ptBR)
    const text = screen.getByRole('complementary').textContent ?? ''
    expect(text).toMatch(/dissertação de mestrado/i)
    expect(text).toMatch(/não é um serviço|não um serviço/i)
    expect(text).toMatch(/um terço/i)
  })

  it('states the same in en-GB', () => {
    renderIn(enGB as unknown as typeof ptBR)
    const text = screen.getByRole('complementary').textContent ?? ''
    expect(text).toMatch(/MSc dissertation artefact/i)
    expect(text).toMatch(/not a public fact-checking service/i)
    expect(text).toMatch(/a third/i)
  })

  it('carries no percentage, in either locale', () => {
    for (const dict of [ptBR, enGB as unknown as typeof ptBR]) {
      const { container, unmount } = renderIn(dict)
      expect(
        container.textContent ?? '',
        'the notice must state its error rate in words: see publicBuild.test.tsx',
      ).not.toMatch(/\d+[.,]?\d*\s*%/)
      unmount()
    }
  })
})
