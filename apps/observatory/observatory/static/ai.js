// PINNeAPPle models in the observatory: orbit decay PINN, small bodies and the learned propagator, space weather,
// observing conditions, stellar ML, the black-hole panel. Each block shows a trust card: verdict, checks against an
// independent reference, and "how this was computed, how much to trust it".
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "—" : (+v).toFixed(d));
const fmtv = (v) => (v === null || v === undefined ? "—" : Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? (+v).toExponential(2) : (+v).toPrecision(3));
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}

export function trustCard({ title, verdict, checks = [], how, trust, extra = "" }) {
  const v = verdict?.status || verdict || "unknown";
  return `<div class="card trust"><div class="row"><b>${esc(title)}</b><span class="badge ${esc(v)}">${esc(String(v).replace(/_/g, " "))}</span></div>
    ${extra}
    ${checks.length ? `<table><tr><th>Check</th><th class="r">value</th><th class="r">limit</th><th></th></tr>${checks.map((c) =>
      `<tr><td>${esc(c.name)}</td><td class="r">${fmtv(c.value)}${c.unit && c.unit !== "-" ? " " + esc(c.unit) : ""}</td><td class="r">${c.higher_is_better ? "≥ " : "≤ "}${fmtv(c.threshold)}</td><td><span class="badge ${esc(c.status)}">${esc(c.status)}</span></td></tr>`).join("")}</table>` : ""}
    <details><summary>How this was computed, how much to trust it</summary><p class="small">${how}</p>${trust ? `<p class="small"><b>Trust:</b> ${trust}</p>` : ""}</details></div>`;
}

// ---------------------------------------------------------------------------------------- orbit decay
export async function renderDecay(el, ctx) {
  el.innerHTML = `<h3>Orbit decay (physics-informed)</h3>
    <div class="row small"><label>days <input id="dDays" type="number" value="60" min="5" max="365" style="width:64px"></label>
    <label>F10.7 <input id="dF" type="number" value="150" min="60" max="300" style="width:64px"></label>
    <label>Ap <input id="dAp" type="number" value="15" min="0" max="200" style="width:56px"></label>
    <label>B = C<sub>D</sub>A/m <input id="dB" type="number" value="0.005" step="0.001" style="width:72px"></label>
    <button class="primary" id="dRun">Predict ISS decay</button></div><div id="dOut"></div>`;
  const run = async () => {
    const out = el.querySelector("#dOut");
    out.innerHTML = `<p class="muted">Training the PINN and the Monte-Carlo envelope on the server (~20 s)…</p>`;
    try {
      const q = `days=${el.querySelector("#dDays").value}&f107=${el.querySelector("#dF").value}&ap=${el.querySelector("#dAp").value}&B=${el.querySelector("#dB").value}&t=${ctx.iso()}`;
      const r = await api(`/api/sat/decay?norad=25544&${q}`);
      const x = r.t_days.map((t) => +t.toFixed(1));
      const drop = r.h_pinn_km[0] - r.h_pinn_km.at(-1);
      out.innerHTML = trustCard({ title: `${r.satellite ? esc(r.satellite.name) : "ISS-like orbit"}: ${num(r.h0_km, 1)} km → ${num(r.h_pinn_km.at(-1), 1)} km in ${r.days} days`,
        verdict: r.verdict, checks: r.checks,
        extra: `<p class="small">Decay now ${num(r.decay_rate_m_per_day_t0, 0)} m/day · loss over the window ${num(drop, 2)} km (5–95 %: ${num(r.h0_km - r.env_p95_km.at(-1), 2)}–${num(r.h0_km - r.env_p05_km.at(-1), 2)} km) · density: ${esc(r.density_model)}</p><div id="dChart"></div>
          ${r.sgp4 ? `<p class="small">SGP4 from the loaded TLE (its own drag model): ${num(r.sgp4.rate_sgp4_m_day, 1)} m/day; this model with the same date and solar activity: ${num(r.sgp4.rate_model_m_day, 1)} m/day (ratio ${num(r.sgp4.ratio, 2)}).</p>` : `<p class="small muted">No current ISS TLE loaded, so no SGP4 comparison (the offline check is in the test suite: ratio 0.94 for a 2018 TLE).</p>`}`,
        how: esc(r.method) + `. Altitude from the semi-major axis of a near-circular orbit. The ballistic coefficient B is an input (ISS ≈ 0.005 m²/kg: C<sub>D</sub> 2.2, ~950 m² mean cross-section, 420 t); SGP4's B* is not a physical B.`,
        trust: "The PINN reproduces the integrator to metres; the forecast itself is limited by the inputs: solar flux and geomagnetic storms weeks ahead are uncertain (shown by the envelope), and the attitude-dependent drag area of a real spacecraft varies." });
      PChart.mount(out.querySelector("#dChart"), { x, height: 200, yLabel: "km", series: [
        { name: "PINN", y: r.h_pinn_km, color: "#7fb4ff", width: 2 },
        { name: "RK integration", y: r.h_ref_km, color: "#ffcf6e", dash: "4 3" },
        { name: "5–95 % envelope", y: r.env_median_km, lo: r.env_p05_km, hi: r.env_p95_km, color: "#3ccf8e", dash: "2 2" }] });
    } catch (e) { out.textContent = e.message; }
  };
  el.querySelector("#dRun").onclick = run;
}

// ------------------------------------------------------------------------------------- space weather
export async function renderSpaceWeather(el, ctx) {
  el.innerHTML = `<p class="muted">Loading Kp and training the forecaster (first time ~1 min)…</p>`;
  try {
    const r = await api(`/api/spaceweather?${ctx.qs()}`);
    const f = (h) => r.forecasts.find((x) => x.horizon_h === h);
    const row = (x) => `<tr><td>+${x.horizon_h} h</td><td class="r">${num(x.median, 1)}</td><td class="r">${num(x.lo, 1)}–${num(x.hi, 1)}</td><td class="r">${num(x.aurora.p_overhead * 100, 0)} %</td><td class="r">${num(x.aurora.p_horizon * 100, 0)} %</td><td><span class="badge ${x.verdict.status}">${x.verdict.status.replace(/_/g, " ")}</span></td></tr>`;
    el.innerHTML = `<div class="kv"><span>Kp now</span><span>${num(r.kp_now, 1)}</span><span>Your geomagnetic latitude</span><span>${num(r.maglat, 1)}°</span>
      <span>Aurora reaches (now)</span><span>${num(r.boundary_now, 1)}° geomag.</span><span>Kp needed overhead / on the horizon</span><span>${r.forecasts[0] ? num(r.forecasts[0].aurora.kp_needed_overhead, 1) + " / " + num(r.forecasts[0].aurora.kp_needed_horizon, 1) : "—"}</span></div>
      <table><tr><th>Lead</th><th class="r">Kp</th><th class="r">80 %</th><th class="r">aurora overhead</th><th class="r">on horizon</th><th></th></tr>${r.forecasts.map(row).join("")}</table>
      <p class="small muted">Source: ${esc(r.source)}</p>` +
      trustCard({ title: "Kp forecaster", verdict: f(24)?.verdict, checks: [...(f(3)?.checks || []), ...(f(24)?.checks || [])],
        how: esc(r.method) + `. At 3 h the network's median lost to persistence on the calibration block, so persistence is used as the median with the calibrated network interval around it.`,
        trust: "Trained and scored on one year (2000, solar maximum): it beats persistence at 24 h but misses most storm onsets (they are driven by solar-wind structures it does not see). Treat it as a calibrated range, not a storm warning. Aurora visibility also needs a dark, clear sky." });
  } catch (e) { el.textContent = e.message; }
}

// --------------------------------------------------------------------------------------- conditions
export async function renderConditions(el, ctx) {
  el.innerHTML = `<p class="muted">Fetching the ensemble forecast…</p>`;
  const r = await api(`/api/conditions?${ctx.qs()}`).catch((e) => ({ error: e.message }));
  if (r.error) {
    el.innerHTML = trustCard({ title: "Observing conditions (Open-Meteo ensemble)", verdict: "not_available",
      extra: `<p class="small">${esc(r.error)}</p>`,
      how: "Per hour: P(clear sky) = fraction of ensemble members with total cloud &lt; 30 %; cloud cover median and 10–90 %; transparency from cloud and humidity; seeing ranked from the 250 hPa jet-stream wind.",
      trust: "The ensemble spread is not yet recalibrated against observations at the site (that needs a forecast/observation history; <code>conditions.calibrate</code> does it). The seeing proxy is a ranking, not arc-seconds." });
    return;
  }
  const x = r.time.map((t) => t.replace("T", " ").slice(5, 16));
  el.innerHTML = `<div id="cChart"></div><p class="small">${r.n_members} members · source ${esc(r.source)}</p>` +
    trustCard({ title: "Observing conditions", verdict: "use_with_caution",
      how: "P(clear) = fraction of members with total cloud &lt; 30 %; transparency from cloud and humidity; seeing ranked from the 250 hPa wind.",
      trust: "Probabilities are the raw ensemble spread (uncalibrated); typical NWP cloud forecasts are over-confident beyond 24 h." });
  PChart.mount(el.querySelector("#cChart"), { x, height: 180, yLabel: "%", yMin: 0, yMax: 100, series: [
    { name: "P(clear) %", y: r.p_clear.map((p) => (p === null ? null : p * 100)), color: "#3ccf8e", width: 2 },
    { name: "cloud cover (median, 10–90 %)", y: r.cloud_q50, lo: r.cloud_q10, hi: r.cloud_q90, color: "#8d9bb3" }] });
}

// ------------------------------------------------------------------------------------------ small bodies
export async function renderSmallBodies(el, ctx) {
  const list = await api("/api/smallbodies");
  el.innerHTML = `<div class="row"><select id="sbBody">${list.bodies.map((b) => `<option value="${b.key}">${esc(b.name)}</option>`).join("")}</select>
    <select id="sbYears"><option value="5">5 years</option><option value="10" selected>10 years</option><option value="23">to 2049</option></select>
    <button class="primary" id="sbRun">Close approaches</button><button class="ghost" id="sbShow">Show orbit</button></div><div id="sbOut"></div><div id="sbVal"><p class="muted">Loading the validation…</p></div>`;
  const out = el.querySelector("#sbOut");
  el.querySelector("#sbRun").onclick = async () => {
    out.innerHTML = `<p class="muted">Integrating (Sun, planets, Moon, relativity)…</p>`;
    try {
      const b = el.querySelector("#sbBody").value;
      const r = await api(`/api/smallbodies/approaches?body=${b}&start=${ctx.iso()}&years=${el.querySelector("#sbYears").value}&max_au=0.2`);
      out.innerHTML = r.approaches.length ? `<table><tr><th>Date (UTC)</th><th class="r">distance</th><th class="r">v<sub>rel</sub></th><th class="r">JPL</th></tr>${r.approaches.map((c) =>
        `<tr${c.dist_km < 1e6 ? ' style="color:#ffcf6e"' : ""}><td>${c.utc.slice(0, 16).replace("T", " ")}</td><td class="r">${c.dist_km < 2e6 ? num(c.dist_km / 1000, 1) + " Mm (" + num(c.lunar_distances, 2) + " LD)" : num(c.dist_au, 4) + " AU"}</td><td class="r">${num(c.v_rel_km_s, 2)} km/s</td><td class="r">${c.jpl ? num(c.jpl.dist_au, 6) + " AU" : "—"}</td></tr>`).join("")}</table><p class="small muted">${esc(r.method)}</p>`
        : `<p class="muted">No approach within 0.2 AU in this window.</p>`;
    } catch (e) { out.textContent = e.message; }
  };
  el.querySelector("#sbShow").onclick = () => ctx.showSmallBody(el.querySelector("#sbBody").value);
  const v = await api("/api/smallbodies/validation").catch((e) => ({ error: e.message }));
  if (v.error) { el.querySelector("#sbVal").textContent = v.error; return; }
  const a = v.apophis_2029;
  let html = trustCard({ title: "Small-body propagator", verdict: v.verdict, checks: v.checks,
    extra: `<p class="small">Apophis on ${esc(a.jpl_cd)}: ${num(a.model_dist_km, 0)} km from the Earth's centre (JPL: ${num(a.jpl_dist_km, 0)} ± ${num(a.jpl_3sigma_km, 0)} km), ${num(a.model_v_km_s, 2)} km/s (JPL ${num(a.jpl_v_km_s, 2)}).</p>`,
    how: "Test-particle integration (DOP853, rtol 1e-11) in the DE421 field of the Sun, the eight planets and the Moon, plus the Sun's 1PN relativistic term, from the JPL SBDB osculating elements.",
    trust: "No Yarkovsky thermal force and no asteroid perturbers: fine for years, not for impact-probability work decades ahead (Apophis's 2068 geometry depends on Yarkovsky)." });
  html += `<div id="sbPinn"><button class="ghost" id="sbPinnGo">Train a PINN for the selected body (1 year, ~1 min)</button></div>`;
  const nr = v.neural;
  if (nr) html += trustCard({ title: "Learned propagator for many NEOs (one network)", verdict: nr.verdict, checks: nr.checks,
    extra: `<p class="small">Trained on ${nr.info.n_train} synthetic NEO orbits, tested on ${nr.info.n_test} unseen ones. ${esc(nr.note)}</p>`,
    how: "An MLP maps the osculating elements at the epoch and the time to the deviation from the Kepler orbit (radial / along-track / normal), trained on the full integrator; early stopping on held-out bodies.",
    trust: "Not used for any result in the app: the gate rejects it. Kept to show what the trust checks catch." });
  el.querySelector("#sbVal").innerHTML = html;
  el.querySelector("#sbPinnGo").onclick = async () => {
    const box = el.querySelector("#sbPinn"), b = el.querySelector("#sbBody").value;
    box.innerHTML = `<p class="muted">Training the PINN for ${esc(b)} on the server…</p>`;
    try {
      const r = await api(`/api/smallbodies/pinn?body=${b}&t=${ctx.iso()}&days=365`);
      box.innerHTML = trustCard({ title: `PINN orbit surrogate: ${esc(r.body)}`, verdict: r.verdict, checks: r.checks,
        extra: `<p class="small">max error (held-out times) ${num(r.summary.max_error_km_test, 0)} km = ${num(r.summary.max_angle_arcsec_test, 3)}″ from the Sun; Kepler alone ${num(r.summary.kepler_max_error_km, 0)} km; trained in ${num(r.train_seconds, 0)} s.</p><div id="sbPinnChart"></div>`,
        how: "r(t) = r<sub>Kepler</sub>(t) + τ² g<sub>θ</sub>(τ): the network learns the planetary perturbation only, from the residual of the full N-body equation of motion (planets from DE421) and 8 anchor states; split-conformal band on held-out integrator states.",
        trust: "Valid for this body over this year; it is a compact, differentiable surrogate of one trajectory, not a general propagator." });
      const x = r.t_days.map((t) => Math.round(t));
      PChart.mount(box.querySelector("#sbPinnChart"), { x, height: 160, yLog: true, yLabel: "km", series: [
        { name: "PINN − integrator", y: r.error_km.map((e) => Math.max(e, 1)), color: "#7fb4ff", width: 2 },
        { name: "Kepler − integrator", y: r.kepler_error_km.map((e) => Math.max(e, 1)), color: "#ff6b6b", dash: "4 3" },
        { name: "conformal band", y: r.band_km.map((e) => Math.max(e, 1)), color: "#3ccf8e", dash: "2 2" }] });
    } catch (e) { box.textContent = e.message; }
  };
}

// ---------------------------------------------------------------------------------------------- stellar ML
export async function renderStarsML(el, ctx) {
  el.innerHTML = `<p class="muted">Clustering and cross-validating…</p>`;
  const r = await api("/api/stars/ml");
  const cols = ["#7fb4ff", "#ff9a6a", "#ffcf6e"];
  const W = el.clientWidth || 360, H = 260, L = 34, B = 24;
  const X = (bv) => L + (bv + 0.4) / 2.4 * (W - L - 6), Y = (m) => 6 + (m + 8) / 24 * (H - 6 - B);
  const up = ctx.aboveHorizon();
  let dots = "";
  for (let k = 0; k < r.hr.length; k++) {
    const isUp = up.has(r.hr[k]);
    dots += `<circle cx="${X(r.bv[k]).toFixed(1)}" cy="${Y(r.Mv[k]).toFixed(1)}" r="${isUp ? 2.2 : 1.2}" fill="${cols[r.cluster[k]]}" opacity="${isUp ? 1 : 0.45}"/>`;
  }
  const legend = r.cluster_names.map((n, j) => `<span><i style="background:${cols[j]}"></i>${esc(n)}</span>`).join("");
  const lumRec = r.lum_cv.recall.map((x, i) => `${esc(r.lum_groups[i])}: ${x === null ? "—" : Math.round(x * 100) + "%"}`).join(" · ");
  el.innerHTML = `<svg width="${W}" height="${H}" style="font:10px system-ui;fill:#8d9bb3">${dots}
    ${[-0.4, 0, 0.5, 1, 1.5, 2].map((v) => `<text x="${X(v)}" y="${H - 6}" text-anchor="middle">${v}</text>`).join("")}${[-6, -3, 0, 3, 6, 9, 12, 15].map((m) => `<text x="${L - 4}" y="${Y(m) + 3}" text-anchor="end">${m}</text>`).join("")}
    <text x="${W / 2}" y="${H}" text-anchor="middle">B − V</text></svg><div class="pchart-legend">${legend}</div><p class="small muted">Bright dots: above your horizon now.</p>` +
    trustCard({ title: "Stellar ML on the HR diagram", verdict: r.verdict, checks: r.checks,
      extra: `<p class="small">${r.n_stars_hr} stars with a trigonometric parallax ≥ 5 mas (${r.n_labelled} with an MK luminosity class). Luminosity class: ${Math.round(r.lum_cv.accuracy * 100)} % (baseline ${Math.round(r.lum_cv.baseline * 100)} %); recall ${lumRec}. Spectral class from B−V: ${Math.round(r.spec_cv.accuracy * 100)} % (baseline ${Math.round(r.spec_cv.baseline * 100)} %).</p>`,
      how: esc(r.method) + ". The clusters are found without labels and only named afterwards by their majority MK class.",
      trust: "Subgiants and bright giants are hard (few examples, they sit between the branches): their recall is low, as the table says. Colours are not dereddened; BSC5 parallaxes are pre-Hipparcos." });
}

// ------------------------------------------------------------------------------------------- black hole
export async function renderBlackHole(el, ctx) {
  el.innerHTML = `<div class="row"><label class="small"><input type="checkbox" id="bhDop" checked> Doppler beaming (physical)</label>
    <label class="small">inclination <input id="bhInc" type="range" min="60" max="89" value="86"></label>
    <button class="primary" id="bhRun">Render where I look</button></div><div id="bhOut"></div><div id="bhZoo"></div>`;
  el.querySelector("#bhRun").onclick = () => {
    const [ra, dec] = ctx.viewRaDec();
    const out = el.querySelector("#bhOut");
    out.innerHTML = `<p class="muted">Tracing null geodesics through the disc (first render ~20 s, then about a second for any direction)…</p>`;
    const img = new Image(); img.className = "lens";
    img.src = `/api/analysis/accretion.png?ra=${ra.toFixed(3)}&dec=${dec.toFixed(3)}&doppler=${el.querySelector("#bhDop").checked}&incl=${el.querySelector("#bhInc").value}`;
    img.onload = () => { out.innerHTML = ""; out.appendChild(img); out.insertAdjacentHTML("beforeend", `<p class="small muted">A Schwarzschild black hole with a thin accretion disc (exact null geodesics; inner edge at the ISCO, 6 GM/c²; Keplerian gas; Doppler beaming and gravitational redshift, g⁴ intensity) in front of the real stars and Milky Way around RA ${ra.toFixed(1)}°, Dec ${dec.toFixed(1)}°. The disc's far side appears above and below the shadow because its light is bent around the hole.</p>`); };
    img.onerror = () => { out.textContent = "Render failed (server busy?)."; };
  };
  const z = await api("/api/blackhole/status").catch(() => ({ available: false, note: "status unavailable" }));
  el.querySelector("#bhZoo").innerHTML = z.available
    ? `<div class="card"><b>Black-hole weather forecast</b> from the model zoo: <code>${esc(z.model)}</code> (gate passed).</div>`
    : `<div class="card"><b>Black-hole weather forecast</b> <span class="badge neutral">placeholder</span><p class="small">${esc(z.note)}</p></div>`;
}
