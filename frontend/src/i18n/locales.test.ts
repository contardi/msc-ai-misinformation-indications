import { describe, expect, it } from 'vitest'
import { LOCALES, type LocaleCode } from './index'

type Node = Record<string, unknown>

function leaves(node: Node, prefix = ''): Array<[string, string]> {
  const out: Array<[string, string]> = []
  for (const [key, value] of Object.entries(node)) {
    const path = prefix ? `${prefix}.${key}` : key
    if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
      out.push(...leaves(value as Node, path))
    } else {
      out.push([path, typeof value])
    }
  }
  return out
}

const CODES = Object.keys(LOCALES) as LocaleCode[]
const REFERENCE: LocaleCode = 'pt-BR'

describe('every locale', () => {
  const reference = leaves(LOCALES[REFERENCE] as unknown as Node)

  it.each(CODES)('%s has exactly the reference keys, with the same shapes', (code) => {
    const actual = leaves(LOCALES[code] as unknown as Node)
    expect(actual.map(([p, t]) => `${p}:${t}`).sort()).toEqual(
      reference.map(([p, t]) => `${p}:${t}`).sort(),
    )
  })

  it.each(CODES)('%s has no empty or placeholder strings', (code) => {
    for (const [path, kind] of leaves(LOCALES[code] as unknown as Node)) {
      if (kind !== 'string') continue
      const value = path
        .split('.')
        .reduce<unknown>((acc, key) => (acc as Node)[key], LOCALES[code]) as string
      expect(value.trim(), `${code} → ${path} is blank`).not.toBe('')
      expect(value, `${code} → ${path} looks like an untranslated marker`).not.toMatch(/^TODO/i)
    }
  })

  it.each(CODES)('%s declares a meta.code matching its registry key', (code) => {
    expect(LOCALES[code].meta.code).toBe(code)
    expect(LOCALES[code].meta.flag.length).toBeGreaterThan(0)
  })
})

describe('pt-BR is Brazilian, not European', () => {
  const EUROPEAN = [
    /\bficheiro/i,
    /\bguardad[oa]\b/i,
    /\bcontact(ar|e|o)\b/i,
    /\butilizador/i,
    /\becrã/i,
    /\btelemóvel/i,
    /\bdemasiad/i,
    /\bestá a \p{L}+r\b/iu, // the pt-PT progressive: "está a transcrever"
    /\bestamos a \p{L}+r\b/iu,
  ]

  it.each(EUROPEAN)('contains no %s', (pattern) => {
    for (const [path, kind] of leaves(LOCALES[REFERENCE] as unknown as Node)) {
      if (kind !== 'string') continue
      const value = path
        .split('.')
        .reduce<unknown>((acc, key) => (acc as Node)[key], LOCALES[REFERENCE]) as string
      expect(value, `pt-BR → ${path}`).not.toMatch(pattern)
    }
  })
})

describe('the disclaimer survives translation', () => {
  it.each(CODES)('%s still denies that absence implies truth', (code) => {
    const emphasis = LOCALES[code].evidence.noneEmphasis.toLowerCase()
    expect(emphasis).toMatch(/\b(não|not|nao)\b/)
    expect(emphasis.length).toBeGreaterThan(10)
  })
})
