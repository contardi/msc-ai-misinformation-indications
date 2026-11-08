import { LOCALES, useI18n, type LocaleCode } from '../i18n'

export default function LanguageSwitcher() {
  const { locale, setLocale, t } = useI18n()
  const codes = Object.keys(LOCALES) as LocaleCode[]

  return (
    <div className="flex shrink-0 items-center gap-1" role="group" aria-label={t.language.current}>
      {codes.map((code) => {
        const meta = LOCALES[code].meta
        const active = code === locale
        return (
          <button
            key={code}
            type="button"
            onClick={() => setLocale(code)}
            aria-pressed={active}
            title={`${t.language.switchTo}: ${meta.label}`}
            className={`flex items-center gap-1.5 whitespace-nowrap border px-2 py-1 font-mono text-[11px] uppercase tracking-wider transition-colors ${
              active
                ? 'border-ink bg-ink text-paper'
                : 'border-line text-quiet hover:border-quiet hover:text-ink'
            }`}
          >
            <span aria-hidden="true">{meta.flag}</span>
            {code}
          </button>
        )
      })}
    </div>
  )
}
