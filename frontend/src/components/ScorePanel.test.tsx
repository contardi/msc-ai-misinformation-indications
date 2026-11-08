import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ScorePanel, { band } from './ScorePanel'
import { makeResponse } from '../test/fixtures'

describe('band thresholds', () => {
  it('treats scores below 0.4 as few signs', () => {
    expect(band(0.0).key).toBe('low')
    expect(band(0.399).key).toBe('low')
  })

  it('treats 0.4 to 0.6 inclusive as inconclusive', () => {
    expect(band(0.4).key).toBe('inconclusive')
    expect(band(0.5).key).toBe('inconclusive')
    expect(band(0.6).key).toBe('inconclusive')
  })

  it('flags only above 0.6', () => {
    expect(band(0.601).key).toBe('high')
    expect(band(1.0).key).toBe('high')
  })
})

describe('ScorePanel rendering', () => {
  it('shows the classifier probability as a rounded percentage', () => {
    render(<ScorePanel result={makeResponse({ score: 0.678 })} />)
    expect(screen.getByText('68%')).toBeInTheDocument()
  })

  it('always renders the uncertainty note from the API', () => {
    const result = makeResponse({ score: 0.9 })
    render(<ScorePanel result={result} />)
    expect(screen.getByText(result.uncertainty_note)).toBeInTheDocument()
  })

  it('is labelled for assistive technology', () => {
    render(<ScorePanel result={makeResponse()} />)
    expect(screen.getByRole('region', { name: 'Avaliação' })).toBeInTheDocument()
  })

  it('shows only the score, never evidence or corroboration', () => {
    const result = makeResponse({ score: 0.9 })
    render(<ScorePanel result={result} />)
    expect(screen.queryByText(/Aos Fatos/)).not.toBeInTheDocument()
    expect(screen.queryByText(result.corroboration.note)).not.toBeInTheDocument()
  })
})
