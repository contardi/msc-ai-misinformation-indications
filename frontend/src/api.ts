import { API_PATHS } from './apiPaths'
import type { AnalyseResponse, FeedbackRequest } from './types'

const API_BASE = import.meta.env.VITE_API_BASE ?? ''

export type ErrorKey =
  | 'network'
  | 'mediaNetwork'
  | 'modelMissing'
  | 'failed'
  | 'rateLimited'
  | 'mediaFailed'
  | 'feedbackFailed'

export class ApiError extends Error {
  constructor(
    readonly key: ErrorKey,
    readonly status?: number,
    readonly serverDetail?: string,
  ) {
    super(key)
    this.name = 'ApiError'
  }
}

export async function analyse(text: string): Promise<AnalyseResponse> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${API_PATHS.analyse}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    })
  } catch {
    throw new ApiError('network')
  }

  if (res.status === 503) throw new ApiError('modelMissing', 503)
  if (res.status === 429) throw new ApiError('rateLimited', 429)
  if (!res.ok) throw new ApiError('failed', res.status)
  return (await res.json()) as AnalyseResponse
}

export async function analyseMedia(file: File): Promise<AnalyseResponse> {
  const body = new FormData()
  body.append('file', file)

  let res: Response
  try {
    res = await fetch(`${API_BASE}${API_PATHS.media}`, { method: 'POST', body })
  } catch {
    throw new ApiError('mediaNetwork')
  }

  if (res.status === 422 || res.status === 503) {
    const detail = await res.json().catch(() => null)
    const message = typeof detail?.detail === 'string' ? detail.detail : undefined
    throw new ApiError('mediaFailed', res.status, message)
  }
  if (res.status === 429) throw new ApiError('rateLimited', 429)
  if (!res.ok) throw new ApiError('failed', res.status)
  return (await res.json()) as AnalyseResponse
}

export async function sendFeedback(payload: FeedbackRequest): Promise<void> {
  let res: Response
  try {
    res = await fetch(`${API_BASE}${API_PATHS.feedback}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })
  } catch {
    throw new ApiError('network')
  }
  if (res.status === 429) throw new ApiError('rateLimited', 429)
  if (!res.ok) throw new ApiError('feedbackFailed', res.status)
}
