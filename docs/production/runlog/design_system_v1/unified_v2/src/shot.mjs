import { createRequire } from 'node:module'
const req = createRequire('/Users/idantal/ShermanQC/FullProject/frontend/react/package.json')
const { chromium } = req('playwright')
const dir = new URL('.', import.meta.url).pathname
const [out, ...specs] = process.argv.slice(2)
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1920, height: 1080 } })
p.on('pageerror', e => console.log('ERR', e.message))
p.on('console', m => m.type() === 'error' && console.log('CONSOLE', m.text()))
for (const s of specs) {
  const i = s.indexOf('='), name = s.slice(0, i), url = s.slice(i + 1)
  await p.goto(`file://${dir}${url}`)
  await p.evaluate(() => document.fonts.ready)
  await p.waitForTimeout(250)
  await p.screenshot({ path: `${out}/${name}.png` })
  console.log('shot', name)
}
await b.close()
