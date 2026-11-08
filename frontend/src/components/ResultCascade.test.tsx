import { render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import ResultCascade, { tierOf } from './ResultCascade'
import { makeResponse } from '../test/fixtures'
import type { AnalyseResponse, FactCheck, Indicator } from '../types'

const CHECK: FactCheck = {
  claim: 'Vacina causa autismo',
  rating: 'Falso',
  publisher: 'Aos Fatos',
  url: 'https://example.org/check',
  verdict: 'false',
}

const CUE: Indicator = {
  kind: 'exclamations',
  label: 'Pontuação sensacionalista',
  detail: '3 pontos de exclamação.',
  count: 3,
}

function withEvidence(checks: FactCheck[], indicators: Indicator[] = []): AnalyseResponse {
  const base = makeResponse()
  return { ...base, evidence: { ...base.evidence, fact_checks: checks }, indicators }
}

describe('which tier answers', () => {
  it('a published fact-check outranks everything else', () => {
    expect(tierOf(withEvidence([CHECK], [CUE]))).toBe('evidence')
  })

  it('falls to indicators when no check was found', () => {
    expect(tierOf(withEvidence([], [CUE]))).toBe('indicators')
  })

  it('falls to the classifier only when neither is present', () => {
    expect(tierOf(withEvidence([], []))).toBe('classifier')
  })
})

describe('when nothing was found', () => {
  it('says so, rather than rendering an empty space', () => {
    render(<ResultCascade result={withEvidence([], [])} />)
    expect(screen.getByText(/Nenhuma checagem publicada e nenhum indício/)).toBeInTheDocument()
  })

  it('tells the reader the score is the weakest evidence, not a verdict', () => {
    render(<ResultCascade result={withEvidence([], [])} />)
    expect(screen.getByText(/evidência mais fraca desta página/)).toBeInTheDocument()
    expect(screen.getByText(/não trate esse número como um veredito/i)).toBeInTheDocument()
  })

  it('offers something the reader can actually do', () => {
    render(<ResultCascade result={withEvidence([], [])} />)
    expect(screen.getByText('O que você pode fazer')).toBeInTheDocument()
    expect(screen.getAllByRole('listitem').length).toBeGreaterThan(2)
  })
})

describe('when a check was found', () => {
  it('names it as the strongest answer available', () => {
    render(<ResultCascade result={withEvidence([CHECK])} />)
    expect(screen.getByText(/Encontramos checagens já publicadas/)).toBeInTheDocument()
  })

  it('offers no next steps, because the question was answered', () => {
    render(<ResultCascade result={withEvidence([CHECK])} />)
    expect(screen.queryByText('O que você pode fazer')).not.toBeInTheDocument()
  })
})

describe('when only indicators fired', () => {
  it('separates how a text is written from whether it is true', () => {
    render(<ResultCascade result={withEvidence([], [CUE])} />)
    expect(screen.getByText(/não se ele é verdadeiro/)).toBeInTheDocument()
  })

  it('is labelled for assistive technology', () => {
    render(<ResultCascade result={withEvidence([], [CUE])} />)
    expect(screen.getByRole('region', { name: 'O que encontramos' })).toBeInTheDocument()
  })
})

describe('in the public build', () => {
  async function loadPublic() {
    vi.stubEnv('VITE_PUBLIC_BUILD', '1')
    vi.resetModules()
    return import('./ResultCascade')
  }

  afterEach(() => {
    vi.unstubAllEnvs()
  })

  it('has no classifier tier to fall back to, because there is no score', async () => {
    const { tierOf: publicTierOf } = await loadPublic()
    expect(publicTierOf(withEvidence([], []))).toBe('none')
  })

  it('still reports a published check as the strongest answer', async () => {
    const { tierOf: publicTierOf } = await loadPublic()
    expect(publicTierOf(withEvidence([CHECK]))).toBe('evidence')
    expect(publicTierOf(withEvidence([], [CUE]))).toBe('indicators')
  })

  it('never promises a number it is not going to show', async () => {
    const { default: PublicCascade } = await loadPublic()
    render(<PublicCascade result={withEvidence([], [])} />)

    expect(screen.getByText(/Nenhuma checagem publicada foi encontrada sobre este conteúdo/))
      .toBeInTheDocument()
    expect(screen.queryByText(/evidência mais fraca desta página/)).toBeNull()
    expect(screen.queryByText(/não trate esse número como um veredito/i)).toBeNull()
    expect(screen.getByText('O que você pode fazer')).toBeInTheDocument()
  })
})
