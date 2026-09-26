// Печатает HTML в PDF через Chromium (Playwright).
// usage: NODE_PATH=$(npm root -g) node render_pdf.cjs input.html output.pdf
const path = require('path');
const { chromium } = require('playwright');

const FOOTER = `
<div style="font-family:'DejaVu Sans',sans-serif;font-size:7px;color:#7a8594;width:100%;
            padding:0 15mm;display:flex;justify-content:space-between;">
  <span>WestCV · план решения · WIUT Hackathon 2026, CV Track</span>
  <span><span class="pageNumber"></span> / <span class="totalPages"></span></span>
</div>`;

(async () => {
  const [input, output] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + path.resolve(input), { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.pdf({
    path: output,
    format: 'A4',
    preferCSSPageSize: true,
    printBackground: true,
    displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: FOOTER,
    margin: { top: '14mm', bottom: '16mm', left: '15mm', right: '15mm' },
    outline: true,
    tagged: true,
  });
  await browser.close();
})();
