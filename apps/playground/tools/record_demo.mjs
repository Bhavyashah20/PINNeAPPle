// Records the Playground demo (about 25 s): reshape an airfoil, watch L/D and the trust verdict change, leave the
// validity range, come back, open "Why trust this?".
//
//   python3 build.py --serve --port 8765 &        (from apps/playground)
//   node tools/record_demo.mjs http://localhost:8765/ /tmp/demo
//   ffmpeg -i /tmp/demo/*.webm ...                 (see docs/guides/playground.md)
import { chromium } from "@playwright/test";

const base = process.argv[2] || "http://localhost:8765/", out = process.argv[3] || "demo-video";
const W = 1280, H = 800;
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: W, height: H }, recordVideo: { dir: out, size: { width: W, height: H } }, colorScheme: "light" });
const page = await ctx.newPage();
await page.goto(`${base}?nosw#/airfoil?preset=NACA%200012&a=4`);
await page.waitForSelector("[data-testid=verdict]");
// a visible pointer (videos do not show the system cursor)
await page.addStyleTag({ content: "#demo-cursor{position:fixed;z-index:9999;width:22px;height:22px;margin:-3px 0 0 -3px;pointer-events:none;transition:transform .08s} #demo-cursor.down{transform:scale(.85)}" });
await page.evaluate(() => {
  const c = document.createElement("div"); c.id = "demo-cursor";
  c.innerHTML = '<svg viewBox="0 0 24 24" width="22" height="22"><path d="M3 2l7 19 2.5-7.5L20 11z" fill="#111" stroke="#fff" stroke-width="1.5" stroke-linejoin="round"/></svg>';
  document.body.append(c);
  addEventListener("mousemove", (e) => { c.style.left = `${e.clientX}px`; c.style.top = `${e.clientY}px`; }, true);
  addEventListener("mousedown", () => c.classList.add("down"), true); addEventListener("mouseup", () => c.classList.remove("down"), true);
});
const wait = (ms) => page.waitForTimeout(ms);
let mx = W / 2, my = H / 2;
async function glide(x, y, ms = 600) {
  const n = Math.max(2, Math.round(ms / 16));
  for (let i = 1; i <= n; i++) { const t = i / n, e = t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2; await page.mouse.move(mx + (x - mx) * e, my + (y - my) * e); await wait(16); }
  mx = x; my = y;
}
async function center(sel) { const b = await page.locator(sel).boundingBox(); return [b.x + b.width / 2, b.y + b.height / 2]; }
async function drag(sel, dy, ms = 1500) {
  const [x, y] = await center(sel);
  await glide(x, y, 500); await page.mouse.down(); await wait(120);
  await glide(x, y + dy, ms); await wait(120); await page.mouse.up();
}

await page.mouse.move(mx, my);
await wait(1200);
await drag('[data-handle="upper1"]', -5, 1100);          // camber the mid chord: lift and L/D rise
await wait(500);
await drag('[data-handle="lower1"]', -5, 1000);
await wait(500);
await drag('[data-handle="lower2"]', -3, 800);           // and the aft part
await wait(600);
const [ax, ay] = await center("[data-testid=alpha]");
const box = await page.locator("[data-testid=alpha]").boundingBox(), at = (a) => box.x + 9 + ((a + 6) / 24) * (box.width - 18);
await glide(at(4), ay, 400); await page.mouse.down(); await glide(at(8), ay, 1000); await glide(at(5), ay, 700); await page.mouse.up();
await wait(500);
await drag('[data-handle="upper0"]', -14, 1300);         // a fat nose: T0 leaves the trained range -> red verdict
await wait(1600);
await drag('[data-handle="upper0"]', 10, 900);           // back inside
await wait(1200);
const [px, py] = await center('[data-preset="NACA 2412"]');
await glide(px, py, 700); await page.mouse.down(); await wait(90); await page.mouse.up();
await wait(1600);
await page.locator("#polar-chart, [data-testid=polar-chart]").first().scrollIntoViewIfNeeded().catch(() => {});
const [tx, ty] = await center("#trust-open");
await glide(tx, ty, 700); await page.mouse.down(); await wait(90); await page.mouse.up();
await wait(2200);
await page.mouse.wheel(0, 500); await wait(1200);
await ctx.close();
await browser.close();
console.log(`video written to ${out}/`);
