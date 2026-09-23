import { chromium, devices } from 'playwright';
const browser = await chromium.launch();

// desktop bar
const d = await browser.newPage({ viewport: { width: 1020, height: 860 }, deviceScaleFactor: 3 });
await d.goto('http://127.0.0.1:5174/', { waitUntil: 'networkidle' });
await d.waitForTimeout(3000);
await d.locator('.composer-wrap').screenshot({ path: process.argv[2] });

// phone
const ctx = await browser.newContext({ ...devices['iPhone 13'] });
const p = await ctx.newPage();
const errs = [];
p.on('pageerror', e => errs.push('PAGEERROR: ' + e.message));
await p.goto('http://127.0.0.1:5174/', { waitUntil: 'networkidle' });
await p.waitForTimeout(3200);
await p.screenshot({ path: process.argv[3] });
console.log('phone errors:', errs.length ? errs.slice(0,3).join(' | ') : 'none');
const beam = await p.evaluate(() => {
  const el = document.querySelector('[class*="voice"], canvas');
  return { canvases: document.querySelectorAll('canvas').length };
});
console.log('phone canvases:', JSON.stringify(beam));
await browser.close();
