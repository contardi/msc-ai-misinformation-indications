import { chromium } from 'playwright'
import { AxeBuilder } from '@axe-core/playwright'
import { writeFileSync } from 'node:fs'

const BASE = process.env.AUDIT_URL ?? 'http://frontend:5173'
const OUT = process.env.AUDIT_OUT ?? '/audit/out/a11y-results.json'

const TAGS = ['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']

const SAMPLE =
  'URGENTE!!! Compartilhe agora antes que apaguem: as urnas eletronicas foram fraudadas na eleicao'

async function newPage(browser, locale) {
  const context = await browser.newContext()
  await context.addInitScript((lang) => window.localStorage.setItem('lang', lang), locale)
  const page = await context.newPage()
  return { page, context }
}

async function analyse(page) {
  await page.fill('textarea', SAMPLE)
  await page.click('button[type="submit"], button:has-text("Verificar"), button:has-text("Check it")')
  await page.waitForSelector('section[aria-label]', { timeout: 90000 })
  await page.waitForLoadState('networkidle')
}

async function audit(page, name) {
  const results = await new AxeBuilder({ page }).withTags(TAGS).analyze()
  const violations = results.violations.map((v) => ({
    id: v.id,
    impact: v.impact,
    help: v.help,
    helpUrl: v.helpUrl,
    nodes: v.nodes.map((n) => ({ target: n.target, failureSummary: n.failureSummary })),
  }))
  const counts = violations.reduce((acc, v) => {
    acc[v.impact ?? 'unknown'] = (acc[v.impact ?? 'unknown'] ?? 0) + v.nodes.length
    return acc
  }, {})
  console.log(
    `${name.padEnd(28)} violations: ${violations.length.toString().padStart(2)}  ` +
      `nodes: ${JSON.stringify(counts)}`,
  )
  return { state: name, passes: results.passes.length, violations }
}

async function assertAppLoaded(page, where) {
  const textareas = await page.locator('textarea').count()
  const buttons = await page.locator('button').count()
  if (textareas === 0 || buttons === 0) {
    const body = (await page.locator('body').innerText()).slice(0, 300)
    throw new Error(
      `Not the application at ${where}: found ${textareas} textarea(s) and ${buttons} button(s). ` +
        `Page begins: ${JSON.stringify(body)}`,
    )
  }
}

const browser = await chromium.launch()
const report = []

for (const locale of ['pt-BR', 'en-GB']) {
  const { page, context } = await newPage(browser, locale)
  await page.goto(BASE)
  await page.waitForLoadState('networkidle')
  await assertAppLoaded(page, BASE)

  report.push(await audit(page, `landing (${locale})`))
  await analyse(page)
  report.push(await audit(page, `result (${locale})`))

  await context.close()
}

await browser.close()

const total = report.reduce((n, r) => n + r.violations.length, 0)
const nodes = report.reduce((n, r) => n + r.violations.reduce((m, v) => m + v.nodes.length, 0), 0)

writeFileSync(OUT, JSON.stringify({ tags: TAGS, total, nodes, report }, null, 2))
console.log(`\ntotal: ${total} distinct violations across ${nodes} nodes -> ${OUT}`)

process.exit(total === 0 ? 0 : 1)
