import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ScorePanel from './components/ScorePanel'
import { makeResponse } from './test/fixtures'

const VERDICT_PHRASES = [
  /\bé falso\b/i,
  /\bé verdadeiro\b/i,
  /\bé mentira\b/i,
  /\bcomprovadamente\b/i,
  /\bconfirmado como\b/i,
  /\bdefinitivamente\b/i,
  /\bcertamente\b/i,
]

const SCORES = [0, 0.05, 0.25, 0.39, 0.4, 0.5, 0.6, 0.61, 0.85, 0.99, 1]

describe('the UI never states a verdict', () => {
  it.each(SCORES)('uses hedged language at score %s', (score) => {
    const { container } = render(<ScorePanel result={makeResponse({ score })} />)
    const text = container.textContent ?? ''
    for (const phrase of VERDICT_PHRASES) {
      expect(text, `score ${score} produced verdict-like wording matching ${phrase}`).not.toMatch(
        phrase,
      )
    }
  })

  it.each(SCORES)('always shows a percentage, so the claim stays probabilistic at %s', (score) => {
    const { container } = render(<ScorePanel result={makeResponse({ score })} />)
    expect(container.textContent ?? '').toMatch(/\d+%/)
  })

  it('still hedges at the maximum score', () => {
    render(<ScorePanel result={makeResponse({ score: 1 })} />)
    expect(
      screen.getByText(/Estimamos 100% de probabilidade de este texto conter desinformação/i),
    ).toBeInTheDocument()
  })

  it('carries the uncertainty note at every score', () => {
    for (const score of SCORES) {
      const { container, unmount } = render(<ScorePanel result={makeResponse({ score })} />)
      expect(container.textContent ?? '').toContain('não um veredito')
      unmount()
    }
  })
})
