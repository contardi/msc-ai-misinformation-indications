import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import IndicatorsPanel from './IndicatorsPanel'
import type { Indicator, IndicatorCoverage } from '../types'

const COVERAGE: IndicatorCoverage = {
  detection_rate: 14.12,
  false_alarm_rate: 2.39,
  one_in: 7,
  corpus: 'FakeRecogna 2.0',
}

const EXCLAMATIONS: Indicator = {
  kind: 'exclamations',
  label: 'Pontuação sensacionalista',
  detail: '3 pontos de exclamação.',
  count: 3,
  evidence: { fake_rate: 10.45, real_rate: 0.33, precision: 96.94, n_fired: 2846, corpus: 'FakeRecogna 2.0' },
  hits: [{ start: 7, end: 8, match: '!', excerpt: 'URGENTE!! O governo escondeu isso...' }],
}

const TERMS: Indicator = {
  kind: 'sensational_terms',
  label: 'Termos de apelo',
  detail: 'Termos como: compartilhe, urgente.',
  terms: ['compartilhe', 'urgente'],
  evidence: { fake_rate: 6.44, real_rate: 2.07, precision: 75.68, n_fired: 2245, corpus: 'FakeRecogna 2.0' },
  hits: [
    {
      start: 47,
      end: 58,
      match: 'Compartilhe',
      excerpt: 'URGENTE!! O governo escondeu isso. Compartilhe agora, antes que apaguem tudo!',
    },
  ],
}

describe('when no cue fires', () => {
  it('still renders the panel rather than disappearing', () => {
    render(<IndicatorsPanel indicators={[]} coverage={COVERAGE} />)
    expect(screen.getByRole('region', { name: 'Pistas de estilo' })).toBeInTheDocument()
  })

  it('says plainly that nothing was found', () => {
    render(<IndicatorsPanel indicators={[]} coverage={COVERAGE} />)
    expect(screen.getByText(/Nenhum indício de linguagem sensacionalista/)).toBeInTheDocument()
  })

  it('states the detection limit, so silence cannot read as a pass', () => {
    render(<IndicatorsPanel indicators={[]} coverage={COVERAGE} />)
    expect(screen.getByText(/1 a cada 7/)).toBeInTheDocument()
    expect(screen.getByText(/não significa que o conteúdo seja confiável/)).toBeInTheDocument()
  })

  it('degrades without the coverage field instead of breaking', () => {
    render(<IndicatorsPanel indicators={[]} coverage={null} />)
    expect(screen.getByText(/Nenhum indício/)).toBeInTheDocument()
    expect(screen.queryByText(/1 a cada/)).not.toBeInTheDocument()
  })
})

describe('when a cue fires', () => {
  it('shows the measured rates rather than asserting the cue matters', () => {
    render(<IndicatorsPanel indicators={[EXCLAMATIONS]} coverage={COVERAGE} />)
    expect(screen.getByText(/10,4% dos textos falsos e 0,3% dos verdadeiros/)).toBeInTheDocument()
  })

  it('shows where the cue fired, in context, with the match emphasised', () => {
    const { container } = render(<IndicatorsPanel indicators={[TERMS]} coverage={COVERAGE} />)
    expect(container.textContent).toContain('Compartilhe agora, antes que apaguem tudo')
    expect(screen.getByText('Compartilhe', { selector: 'strong' })).toBeInTheDocument()
  })

  it('falls back to plain text when the match is not inside the excerpt', () => {
    const odd: Indicator = {
      ...TERMS,
      hits: [{ start: 0, end: 5, match: 'ausente', excerpt: 'Um trecho qualquer do texto.' }],
    }
    const { container } = render(<IndicatorsPanel indicators={[odd]} coverage={COVERAGE} />)
    expect(container.textContent).toContain('Um trecho qualquer do texto.')
  })

  it('renders a cue that arrives without its measurement', () => {
    const bare: Indicator = { kind: 'exclamations', label: 'X', detail: 'Y', count: 2 }
    render(<IndicatorsPanel indicators={[bare]} coverage={COVERAGE} />)
    expect(screen.getByText(/2 pontos de exclamação/)).toBeInTheDocument()
  })

  it('does not show the nothing-found message', () => {
    render(<IndicatorsPanel indicators={[EXCLAMATIONS]} coverage={COVERAGE} />)
    expect(screen.queryByText(/Nenhum indício/)).not.toBeInTheDocument()
  })
})
