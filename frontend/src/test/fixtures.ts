import type { AnalyseResponse } from '../types'

export function makeResponse(overrides: Partial<AnalyseResponse> = {}): AnalyseResponse {
  return {
    score: 0.5,
    label: 'likely_misinformation',
    source: {
      kind: 'text',
      extracted_text: '',
      duration_seconds: 0,
      model: '',
      notes: [],
    },
    uncertainty_note: 'Avaliação probabilística, não um veredito.',
    explanation: {
      text: 'O modelo estima 50% de probabilidade de o texto conter desinformação.',
      top_tokens: [
        { token: 'urgente', weight: 0.42 },
        { token: 'segredo', weight: 0.31 },
      ],
    },
    evidence: {
      fact_checks: [
        {
          claim: 'Urnas eletrônicas foram fraudadas',
          rating: 'Falso',
          publisher: 'Aos Fatos',
          url: 'https://example.org/checagem',
          verdict: 'false',
        },
      ],
      note: 'Checagens relacionadas encontradas.',
    },
    corroboration: {
      found_in_trusted_sources: true,
      note: 'Há checagem relacionada em: Aos Fatos.',
      checked: [
        { name: 'Aos Fatos', found: true, url: 'https://example.org/a' },
        { name: 'Agência Lupa', found: false, url: 'https://example.org/b' },
      ],
    },
    indicators: [
      {
        kind: 'caps',
        label: 'Uso de maiúsculas',
        detail: '3 palavra(s) em CAIXA ALTA.',
        count: 3,
        terms: [],
      },
    ],
    ...overrides,
  }
}
