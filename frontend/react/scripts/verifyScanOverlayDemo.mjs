import fs from 'node:fs/promises'
import path from 'node:path'
import { chromium } from 'playwright'

function parseArgs(argv) {
  const options = {
    baseUrl: 'http://127.0.0.1:4174',
    outDir: '/tmp/scan-overlay-demo-smoke',
  }
  for (let index = 0; index < argv.length; index += 1) {
    const token = argv[index]
    if (token === '--base-url' && argv[index + 1]) {
      options.baseUrl = argv[index + 1]
      index += 1
      continue
    }
    if (token === '--out-dir' && argv[index + 1]) {
      options.outDir = argv[index + 1]
      index += 1
    }
  }
  return options
}

async function selectedCardText(page) {
  const raw = await page.locator('text=Selected bend card').locator('..').textContent()
  return (raw || '').replace(/\s+/g, ' ').trim()
}

async function pickChipLabel(page) {
  const labels = await page.locator('button').evaluateAll((nodes) =>
    nodes
      .map((node) => (node.textContent || '').replace(/\s+/g, ' ').trim())
      .filter((text) => /^S\d+$/i.test(text)),
  )
  if (!labels.length) throw new Error('No bend chip buttons were found on the demo page')
  return labels.includes('S8') ? 'S8' : labels[0]
}

async function main() {
  const { baseUrl, outDir } = parseArgs(process.argv.slice(2))
  await fs.mkdir(outDir, { recursive: true })

  const browser = await chromium.launch({ headless: true })
  const page = await browser.newPage({ viewport: { width: 1600, height: 1200 } })
  const demoUrl = `${baseUrl.replace(/\/$/, '')}/bend-inspection/demo/scan-overlay`

  try {
    await page.goto(demoUrl, { waitUntil: 'networkidle' })

    const heading = await page.locator('h1').textContent()
    if (!(heading || '').includes('Clickable bend marks and bend cards')) {
      throw new Error(`Unexpected demo heading: ${heading || '<missing>'}`)
    }

    const overviewPath = path.join(outDir, 'overview.png')
    await page.screenshot({ path: overviewPath, fullPage: true })

    const chipLabel = await pickChipLabel(page)
    await page.getByRole('button', { name: chipLabel, exact: true }).click()
    await page.waitForTimeout(250)

    const cardAfterChip = await selectedCardText(page)
    if (!cardAfterChip.includes(chipLabel)) {
      throw new Error(`Selected bend card did not update after chip click. Expected ${chipLabel}, got: ${cardAfterChip}`)
    }

    const chipPath = path.join(outDir, `selected-${chipLabel.toLowerCase()}-chip.png`)
    await page.screenshot({ path: chipPath, fullPage: true })

    const callout = page.locator('[role="button"]').first()
    const calloutText = ((await callout.textContent()) || '').replace(/\s+/g, ' ').trim()
    if (!calloutText) {
      throw new Error('Expected an in-view bend callout button after chip selection')
    }
    await callout.click()
    await page.waitForTimeout(200)

    const cardAfterCallout = await selectedCardText(page)
    if (!cardAfterCallout.includes(chipLabel)) {
      throw new Error(`Selected bend card lost focus after callout click. Expected ${chipLabel}, got: ${cardAfterCallout}`)
    }

    const calloutPath = path.join(outDir, `selected-${chipLabel.toLowerCase()}-callout.png`)
    await page.screenshot({ path: calloutPath, fullPage: true })

    const summary = {
      ok: true,
      demoUrl,
      chipLabel,
      overviewPath,
      chipPath,
      calloutPath,
      cardAfterChip,
      cardAfterCallout,
      calloutText,
    }
    const summaryPath = path.join(outDir, 'summary.json')
    await fs.writeFile(summaryPath, `${JSON.stringify(summary, null, 2)}\n`, 'utf8')
    console.log(JSON.stringify(summary, null, 2))
  } finally {
    await browser.close()
  }
}

main().catch((error) => {
  console.error(error instanceof Error ? error.stack || error.message : String(error))
  process.exitCode = 1
})
