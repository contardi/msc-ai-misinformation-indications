export interface TokenWeight {
  token: string
  weight: number
}

export interface Explanation {
  text: string
  top_tokens: TokenWeight[]
}

export interface FactCheck {
  claim: string
  rating: string
  publisher: string
  url: string
  verdict: string
}

export interface Evidence {
  fact_checks: FactCheck[]
  note: string
}

export interface CheckerStatus {
  name: string
  found: boolean
  url: string
}

export interface Corroboration {
  found_in_trusted_sources: boolean
  note: string
  checked: CheckerStatus[]
}

export interface Indicator {
  kind: string
  label: string // pre-rendered pt-BR; the fallback when `kind` is unknown, never the primary path
  detail: string
  count?: number | null
  terms?: string[]
  evidence?: CueEvidence | null
  hits?: CueHit[]
}

export interface CueEvidence {
  fake_rate: number
  real_rate: number
  precision: number
  n_fired: number
  corpus: string
}

export interface IndicatorCoverage {
  detection_rate: number
  false_alarm_rate: number
  one_in: number
  corpus: string
}

export interface CueHit {
  start: number
  end: number
  match: string
  excerpt: string
}

export interface SourceInfo {
  kind: string
  extracted_text: string
  duration_seconds: number
  model: string
  notes: string[]
}

export interface AnalyseResponse {
  score: number // 0..1, the only number
  label: string
  uncertainty_note: string
  explanation: Explanation
  evidence: Evidence
  corroboration: Corroboration
  indicators: Indicator[]
  indicator_coverage?: IndicatorCoverage | null
  source: SourceInfo
}

export type FeedbackReason =
  | 'evidence_irrelevant'
  | 'evidence_missing'
  | 'score_too_high'
  | 'score_too_low'
  | 'explanation_unclear'
  | 'other'

export interface FeedbackRequest {
  reason: FeedbackReason
  band: 'low' | 'inconclusive' | 'high' | null
  had_evidence: boolean
  locale: 'pt-BR' | 'en-GB'
}
