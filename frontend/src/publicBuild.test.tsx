import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { makeResponse } from './test/fixtures'

async function renderApp(publicBuild: boolean) {
  vi.stubEnv('VITE_PUBLIC_BUILD', publicBuild ? '1' : '0')
  vi.resetModules()
  const { default: App } = await import('./App')
  return render(<App />)
}

async function submit(container: HTMLElement) {
  fireEvent.change(screen.getByLabelText(/cole a mensagem|paste the message/i), {
    target: { value: 'Urnas eletrônicas foram fraudadas na eleição' },
  })
  fireEvent.click(screen.getByRole('button', { name: /verificar|check it/i }))
  await waitFor(() => expect(container.textContent).toMatch(/Aos Fatos/))
}

beforeEach(() => {
  localStorage.clear()
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(makeResponse({ score: 0.67 })), { status: 200 })),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
  vi.unstubAllEnvs()
  localStorage.clear()
})

describe('VITE_PUBLIC_BUILD=1', () => {
  it('renders no percentage anywhere on the page, results and history included', async () => {
    const { container } = await renderApp(true)
    await submit(container)

    const text = container.textContent ?? ''
    expect(text, 'a percentage reached the public build').not.toMatch(/\d+\s*%/)
    expect(text).not.toMatch(/probabilidade de (este|esse) texto conter/i)
  })

  it('still shows the published fact-check, which is what it exists to report', async () => {
    const { container } = await renderApp(true)
    await submit(container)

    expect(screen.getAllByText(/Aos Fatos/).length).toBeGreaterThan(0)
    expect(container.textContent).toMatch(/Falso/i)
  })
})

describe('the artefact build (control)', () => {
  it('does render the percentage, proving the assertion above is meaningful', async () => {
    const { container } = await renderApp(false)
    await submit(container)

    expect(container.textContent ?? '').toMatch(/67\s*%/)
  })
})
