import { createRequire } from 'node:module'
const req = createRequire('/Users/idantal/ShermanQC/FullProject/frontend/react/package.json')
const { chromium } = req('playwright')
const dir = new URL('.', import.meta.url).pathname
const out = process.argv[2]
const names = process.argv.slice(3)
const b = await chromium.launch()
const p = await b.newPage({ viewport: { width: 1920, height: 1080 } })
p.on('pageerror', e => console.log('ERR', e.message))
for (const n of names) {
  await p.goto(`file://${dir}${n}.html`)
  await p.evaluate(() => document.fonts.ready)
  await p.waitForTimeout(200)
  await p.screenshot({ path: `${out}/${n}.png` })
  console.log('shot', n)
}
await b.close()
