// Loads each mode and checks the numbers on the page against the Python zoo model for fixed inputs
// (tests/fixtures/smoke.json is written by tools/build_assets.py from pp.zoo).
import { expect, test } from "@playwright/test";
import { readFileSync } from "node:fs";

const smoke = JSON.parse(readFileSync(new URL("../fixtures/smoke.json", import.meta.url)));
const pred = (page) => page.evaluate(() => window.__playground && window.__playground.pred);
const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-12);

async function settle(page) {
  await expect(page.getByTestId("verdict")).toBeVisible();
  await page.waitForFunction(() => window.__playground && window.__playground.pred);
  await page.waitForTimeout(150);
}

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (e) => { throw e; });
});

test("airfoil: Cl, Cd, Cm and the ensemble spread match the zoo model", async ({ page }) => {
  for (const c of smoke.airfoil) {
    await page.goto(`./?nosw#/airfoil?s=${c.shape.join(",")}&a=${c.alpha}`);
    await settle(page);
    const p = await pred(page);
    expect(Math.abs(p.Cl - c.Cl)).toBeLessThan(1e-5);
    expect(rel(p.Cd, c.Cd)).toBeLessThan(1e-5);
    expect(Math.abs(p.Cm - c.Cm)).toBeLessThan(1e-5);
    expect(Math.abs(p.Cl_std - c.Cl_std)).toBeLessThan(1e-5);
    // what the page shows is the same number
    expect(Number(await page.getByTestId("Cl").getAttribute("data-raw"))).toBeCloseTo(p.Cl, 12);
    // the range verdict agrees with ZooModel.in_range (NACA 4415 sits just outside the CST box: M2 0.123 > 0.12)
    await expect(page.locator('main [data-check="cst-range"]')).toHaveClass(c.in_range ? /pass/ : /fail/);
    await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", c.in_range ? "pass" : "fail");
  }
  await expect(page.getByTestId("field-note")).toContainText("not available yet");
});

test("airfoil: leaving the validity range turns the verdict red", async ({ page }) => {
  await page.goto("./?nosw#/airfoil");
  await settle(page);
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "pass");
  await page.getByRole("button", { name: "NACA 0025" }).click();
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "fail");
  await expect(page.locator('main [data-check="cst-range"]')).toContainText("T0");
  await page.getByRole("button", { name: "NACA 2412" }).click();
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "pass");
  await page.getByTestId("alpha").fill("16");
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "fail");
  await expect(page.locator('main [data-check="alpha-range"]')).toHaveClass(/fail/);
});

test("airfoil: dragging a handle reshapes the airfoil and updates L/D and the URL", async ({ page }) => {
  await page.goto("./?nosw#/airfoil");
  await settle(page);
  const before = await pred(page), hash0 = await page.evaluate(() => location.hash);
  const h = page.locator('[data-handle="upper1"]'), box = await h.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2, box.y - 25, { steps: 8 });
  await page.mouse.up();
  await page.waitForTimeout(300);
  const after = await pred(page);
  expect(after.Cl).not.toBeCloseTo(before.Cl, 4);
  expect(await page.evaluate(() => location.hash)).not.toBe(hash0);
  // the shared link reproduces the state exactly
  const url = page.url(), p2 = await page.context().newPage();
  await p2.goto(url);
  await expect(p2.getByTestId("verdict")).toBeVisible();
  await p2.waitForFunction(() => window.__playground && window.__playground.pred);
  await p2.waitForTimeout(150);
  expect((await pred(p2)).LD).toBeCloseTo(after.LD, 2);
});

test("heat sink: R, the conformal bound and the pressure drop match the zoo model", async ({ page }) => {
  const c = smoke.heatsink[0], S = { base_width_mm: "W", base_depth_mm: "D", base_thickness_mm: "tb", n_fins: "n", fin_thickness_mm: "t", fin_height_mm: "H", k_w_mk: "k", source_width_mm: "sw", source_depth_mm: "sd", t_ambient_c: "Ta", air_velocity_m_s: "V", power_w: "Q" };
  const q = Object.entries(c.inputs).map(([k, v]) => `${S[k]}=${v}`).join("&");
  await page.goto(`./?nosw#/heatsink?${q}`);
  await settle(page);
  const p = await pred(page);
  expect(rel(p.R, c.r_base_k_w)).toBeLessThan(1e-5);
  expect(rel(p.R95, c.r_base_upper95_k_w)).toBeLessThan(1e-5);
  expect(rel(p.dP, c.pressure_drop_pa)).toBeLessThan(1e-5);
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "pass");
  await page.getByTestId("t").fill("4.5");
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "fail");
});

test("rover: drawbar pull, load and torque match the zoo model", async ({ page }) => {
  for (const c of smoke.rover) {
    await page.goto(`./?nosw#/rover?s=${c.slip}&z=${c.sinkage_m * 1000}`);
    await settle(page);
    const p = await pred(page);
    expect(Math.abs(p.Fx - c.Fx_N)).toBeLessThan(0.02);
    expect(Math.abs(p.Fz - c.Fz_N)).toBeLessThan(0.02);
    expect(Math.abs(p.My - c.My_Nm)).toBeLessThan(0.01);
    await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "pass");
  }
  await page.getByTestId("sinkage").fill("75");
  await expect(page.getByTestId("verdict")).toHaveAttribute("data-verdict", "fail");
});

test("gallery: the index, a 3-D view and the info card", async ({ page }) => {
  await page.goto("./?nosw#/gallery?g=wigley_hull");
  await expect(page.getByTestId("geometry-info")).toContainText("Wigley");
  await expect(page.locator(".gcard")).toHaveCount(17);
  await expect(page.getByTestId("viewer")).toHaveAttribute("data-loaded", "wigley_hull", { timeout: 20_000 });
  await page.locator('.gcard[data-name="pin_fin_heatsink"]').click();
  await expect(page.getByTestId("geometry-info")).toContainText("heat sink");
  await expect(page).toHaveURL(/g=pin_fin_heatsink/);
  await expect(page.getByTestId("geometry-info")).toContainText("pp.geometries.build");
});

test("why trust this: the model card, the evidence and the checks that ran", async ({ page }) => {
  await page.goto("./?nosw#/airfoil");
  await settle(page);
  await page.getByRole("button", { name: "Why trust this?" }).click();
  const d = page.locator("#trust");
  await expect(d).toHaveClass(/open/);
  await expect(d).toContainText("airfoil_cst_rans_mlp v1.0.0");
  await expect(d).toContainText("APPROVED");
  await expect(d).toContainText("Validity range");
  await expect(d).toContainText("lift slope vs thin-airfoil");
  await expect(d).toContainText("Data provenance");
});

test("offline: after one visit the playground works without a network", async ({ page, context }) => {
  await page.goto("./#/airfoil");
  await settle(page);
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await page.waitForFunction(async () => (await caches.keys()).some((k) => k.startsWith("playground-")) && !!navigator.serviceWorker.controller, null, { timeout: 20_000 }).catch(() => {});
  await page.reload();
  await page.evaluate(async () => { await navigator.serviceWorker.ready; });
  await context.setOffline(true);
  await page.goto("./#/rover?s=0.3&z=25");
  await settle(page);
  expect((await pred(page)).Fx).toBeGreaterThan(0);
  await context.setOffline(false);
});

test("mobile layout: no horizontal scroll, the shape comes first @mobile", async ({ page }) => {
  await page.goto("./?nosw#/airfoil");
  await settle(page);
  const [sw, cw] = await page.evaluate(() => [document.documentElement.scrollWidth, document.documentElement.clientWidth]);
  expect(sw).toBeLessThanOrEqual(cw);
  const stage = await page.getByTestId("airfoil-svg").boundingBox(), ld = await page.getByTestId("LD").boundingBox();
  expect(stage.y).toBeLessThan(ld.y);
});

test("engine=ort: onnxruntime-web gives the same numbers", async ({ page }) => {
  // the page loads onnxruntime-web from the CDN; serve the same pinned version from node_modules instead
  const dist = new URL("../../node_modules/onnxruntime-web/dist/", import.meta.url);
  await page.route("https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/**", (route) => {
    const file = route.request().url().split("/dist/")[1].split("?")[0];
    route.fulfill({ body: readFileSync(new URL(file, dist)), contentType: file.endsWith(".wasm") ? "application/wasm" : "text/javascript", headers: { "access-control-allow-origin": "*" } });
  });
  for (const c of smoke.airfoil) {
    await page.goto(`./?nosw#/airfoil?s=${c.shape.join(",")}&a=${c.alpha}&engine=ort`);
    await settle(page);
    const p = await pred(page);
    expect(Math.abs(p.Cl - c.Cl)).toBeLessThan(1e-5);
    expect(rel(p.Cd, c.Cd)).toBeLessThan(1e-5);
    expect(Math.abs(p.Cm - c.Cm)).toBeLessThan(1e-5);
  }
  await expect(page.locator("#engine-note")).toContainText("onnxruntime-web");
});
