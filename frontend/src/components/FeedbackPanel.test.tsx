import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import FeedbackPanel from './FeedbackPanel'
import { makeResponse } from '../test/fixtures'

function fetchOk() {
  return vi.fn().mockResolvedValue({ ok: true, status: 204 })
}

function sentBody(fetchMock: ReturnType<typeof vi.fn>) {
  return JSON.parse(fetchMock.mock.calls[0][1].body as string)
}

async function choose(label: RegExp) {
  fireEvent.click(await screen.findByLabelText(label))
  fireEvent.click(screen.getByRole('button', { name: /enviar/i }))
}

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
})

describe('what the panel sends', () => {
  it('sends the band, never the score', async () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    render(<FeedbackPanel result={makeResponse({ score: 0.72 })} />)

    await choose(/estimativa parece alta/i)

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const body = sentBody(fetchMock)
    expect(body.band).toBe('high')
    expect(JSON.stringify(body)).not.toContain('0.72')
  })

  it('sends nothing but the four declared fields', async () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    render(<FeedbackPanel result={makeResponse({ score: 0.72 })} />)

    await choose(/estimativa parece alta/i)

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const body = sentBody(fetchMock)
    expect(Object.keys(body).sort()).toEqual(['band', 'had_evidence', 'locale', 'reason'])
    expect(JSON.stringify(body)).not.toMatch(/urgente|segredo|urnas/i)
  })

  it('reports the chosen reason', async () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    render(<FeedbackPanel result={makeResponse()} />)

    await choose(/checagens não têm relação/i)

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(sentBody(fetchMock).reason).toBe('evidence_irrelevant')
    expect(sentBody(fetchMock).had_evidence).toBe(true)
  })

  it('reports an empty evidence panel', async () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    const empty = makeResponse()
    empty.evidence.fact_checks = []
    render(<FeedbackPanel result={empty} />)

    await choose(/faltou uma checagem/i)

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(sentBody(fetchMock).had_evidence).toBe(false)
  })
})

describe('the control itself', () => {
  it('offers no free-text field, in any state', () => {
    vi.stubGlobal('fetch', fetchOk())
    const { container } = render(<FeedbackPanel result={makeResponse()} />)
    expect(container.querySelector('textarea')).toBeNull()
    expect(container.querySelector('input[type="text"]')).toBeNull()
    expect(container.querySelectorAll('input').length).toBe(
      container.querySelectorAll('input[type="radio"]').length,
    )
  })

  it('will not submit before a reason is chosen', () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    render(<FeedbackPanel result={makeResponse()} />)

    const button = screen.getByRole('button', { name: /enviar/i })
    expect(button).toBeDisabled()
    fireEvent.click(button)
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('acknowledges once and does not invite a second submission', async () => {
    vi.stubGlobal('fetch', fetchOk())
    render(<FeedbackPanel result={makeResponse()} />)

    await choose(/checagens não têm relação/i)

    expect(await screen.findByText(/sua resposta foi registrada/i)).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /enviar/i })).toBeNull()
  })

  it('says so when the store is unavailable, rather than thanking the reader', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: false, status: 503 }))
    render(<FeedbackPanel result={makeResponse()} />)

    await choose(/checagens não têm relação/i)

    expect(await screen.findByRole('alert')).toHaveTextContent(/não foi possível registrar/i)
    expect(screen.queryByText(/sua resposta foi registrada/i)).toBeNull()
  })

  it('states what leaves the browser', () => {
    vi.stubGlobal('fetch', fetchOk())
    const { container } = render(<FeedbackPanel result={makeResponse()} />)
    expect(container.textContent).toMatch(/não é enviado nem armazenado/i)
  })

  it('renders no percentage, so the public build stays clean', () => {
    vi.stubGlobal('fetch', fetchOk())
    const { container } = render(<FeedbackPanel result={makeResponse({ score: 0.72 })} />)
    expect(container.textContent ?? '').not.toMatch(/\d+%/)
  })
})

describe('in the public build', () => {
  async function renderPublic(result = makeResponse({ score: 0.72 })) {
    vi.stubEnv('VITE_PUBLIC_BUILD', '1')
    vi.resetModules()
    const { default: Panel } = await import('./FeedbackPanel')
    return render(<Panel result={result} />)
  }

  it('offers no reasons about a score the reader never saw', async () => {
    vi.stubGlobal('fetch', fetchOk())
    await renderPublic()

    expect(screen.getByLabelText(/checagens não têm relação/i)).toBeInTheDocument()
    expect(screen.queryByLabelText(/estimativa parece alta/i)).toBeNull()
    expect(screen.queryByLabelText(/estimativa parece baixa/i)).toBeNull()
    expect(screen.queryByLabelText(/não entendi a explicação/i)).toBeNull()
  })

  it('sends a null band, because there was no number on screen', async () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    await renderPublic()

    await choose(/checagens não têm relação/i)

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    expect(sentBody(fetchMock).band).toBeNull()
  })

  it('still sends nothing but the four declared fields', async () => {
    const fetchMock = fetchOk()
    vi.stubGlobal('fetch', fetchMock)
    await renderPublic()

    await choose(/checagens não têm relação/i)

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())
    const body = sentBody(fetchMock)
    expect(Object.keys(body).sort()).toEqual(['band', 'had_evidence', 'locale', 'reason'])
    expect(JSON.stringify(body)).not.toMatch(/urgente|segredo|urnas/i)
  })
})
