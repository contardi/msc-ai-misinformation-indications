import { afterEach, describe, expect, it, vi } from 'vitest'
import { analyse, analyseMedia, ApiError } from './api'
import { makeResponse } from './test/fixtures'

function mockFetch(impl: () => Promise<Response> | Response) {
  const spy = vi.fn(impl)
  vi.stubGlobal('fetch', spy)
  return spy
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.restoreAllMocks()
})

describe('analyse()', () => {
  it('posts the text as JSON to /analyse', async () => {
    const payload = makeResponse()
    const spy = mockFetch(() => new Response(JSON.stringify(payload), { status: 200 }))

    await analyse('um texto qualquer')

    expect(spy).toHaveBeenCalledOnce()
    const [url, init] = spy.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toContain('/analyse')
    expect(init.method).toBe('POST')
    expect(JSON.parse(init.body as string)).toEqual({ text: 'um texto qualquer' })
  })

  it('returns the parsed contract intact', async () => {
    const payload = makeResponse({ score: 0.42 })
    mockFetch(() => new Response(JSON.stringify(payload), { status: 200 }))

    const result = await analyse('texto')

    expect(result.score).toBe(0.42)
    expect(result.evidence.fact_checks).toHaveLength(1)
    expect(result.corroboration.checked).toHaveLength(2)
    expect(result.explanation.top_tokens[0].token).toBe('urgente')
  })

  it('flags the 503 that appears before a model is trained', async () => {
    mockFetch(() => new Response('', { status: 503 }))
    await expect(analyse('texto')).rejects.toMatchObject({ key: 'modelMissing', status: 503 })
  })

  it('carries the status code for other HTTP failures', async () => {
    mockFetch(() => new Response('', { status: 500 }))
    await expect(analyse('texto')).rejects.toMatchObject({ key: 'failed', status: 500 })
  })

  it('reports a network failure without leaking the raw error', async () => {
    mockFetch(() => Promise.reject(new TypeError('Failed to fetch')))
    const err = await analyse('texto').catch((e) => e)
    expect(err).toBeInstanceOf(ApiError)
    expect(err.key).toBe('network')
    expect(err.serverDetail).toBeUndefined()
  })
})

describe('rate limiting is reported in the reader’s language', () => {
  it('maps 429 on /analyse to the translated key, not a generic failure', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 429 }))
    await expect(analyse('texto')).rejects.toMatchObject({ key: 'rateLimited', status: 429 })
  })
})

describe('a failed upload says something a reader can act on', () => {
  it('does not tell the reader to check a connection that is fine', async () => {
    mockFetch(() => Promise.reject(new TypeError('Failed to fetch')))
    const file = new File([new Uint8Array(16)], 'nota.m4a', { type: 'audio/x-m4a' })

    const err = await analyseMedia(file).catch((e) => e)

    expect(err).toBeInstanceOf(ApiError)
    expect(err.key).toBe('mediaNetwork')
    expect(err.key).not.toBe('network')
    expect(err.serverDetail).toBeUndefined()
  })

  it('still reports a rate limit as a rate limit, not as an upload problem', async () => {
    mockFetch(() => new Response('', { status: 429 }))
    const file = new File([new Uint8Array(16)], 'nota.m4a', { type: 'audio/x-m4a' })

    await expect(analyseMedia(file)).rejects.toMatchObject({ key: 'rateLimited', status: 429 })
  })
})
