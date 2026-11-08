import { createContext, useContext, useEffect, useState } from 'react'
import { ptBR, type Dictionary } from './pt-BR'
import { enGB } from './en-GB'

export const LOCALES = { 'pt-BR': ptBR, 'en-GB': enGB } as const

export type LocaleCode = keyof typeof LOCALES

const STORAGE_KEY = 'lang'

export function detectLocale(): LocaleCode {
  try {
    const saved = localStorage.getItem(STORAGE_KEY)
    if (saved && saved in LOCALES) return saved as LocaleCode
  } catch {
  }
  const nav = typeof navigator !== 'undefined' ? navigator.language : ''
  if (nav && !nav.toLowerCase().startsWith('pt')) return 'en-GB'
  return 'pt-BR'
}

interface I18nValue {
  locale: LocaleCode
  t: Dictionary
  setLocale: (code: LocaleCode) => void
}

export const I18nContext = createContext<I18nValue>({
  locale: 'pt-BR',
  t: ptBR,
  setLocale: () => {},
})

export function useI18n(): I18nValue {
  return useContext(I18nContext)
}

export function useT(): Dictionary {
  return useContext(I18nContext).t
}

export function useLocaleState() {
  const [locale, setLocale] = useState<LocaleCode>(detectLocale)

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, locale)
    } catch {
    }
    document.documentElement.lang = LOCALES[locale].meta.htmlLang
    document.title = LOCALES[locale].meta.documentTitle
  }, [locale])

  return { locale, setLocale, t: LOCALES[locale] as Dictionary }
}
