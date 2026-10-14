import * as THREE from "three";
import { SkyView } from "./sky.js";
import { SystemView } from "./system.js";
import { D2R, R2D, altazVec, compass, dateToJd, fmtDec, fmtRA, jdToDate, vecAltaz, mul3, radecVec } from "./astro.js";
import { renderBlackHole, renderConditions, renderDecay, renderSmallBodies, renderSpaceWeather, renderStarsML } from "./ai.js";

// ------------------------------------------------------------------------------------------------- setup
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const num = (v, d = 2) => (v === null || v === undefined || !isFinite(v) ? "—" : (+v).toFixed(d));
const params = new URLSearchParams(location.search);
const CITIES = [
  ["São Paulo", -23.5505, -46.6333, 760, "America/Sao_Paulo"], ["Rio de Janeiro", -22.9068, -43.1729, 10, "America/Sao_Paulo"],
  ["Brasília", -15.7939, -47.8828, 1170, "America/Sao_Paulo"], ["Lisbon", 38.7223, -9.1393, 50, "Europe/Lisbon"],
  ["London", 51.5074, -0.1278, 20, "Europe/London"], ["New York", 40.7128, -74.006, 10, "America/New_York"],
  ["San Francisco", 37.7749, -122.4194, 20, "America/Los_Angeles"], ["Tokyo", 35.6762, 139.6503, 40, "Asia/Tokyo"],
  ["Sydney", -33.8688, 151.2093, 50, "Australia/Sydney"], ["Atacama (ALMA)", -23.0293, -67.7548, 5050, "America/Santiago"],
  ["Mauna Kea", 19.8207, -155.4681, 4205, "Pacific/Honolulu"], ["Reykjavík", 64.1466, -21.9426, 30, "Atlantic/Reykjavik"],
  ["Cape Town", -33.9249, 18.4241, 20, "Africa/Johannesburg"], ["Equator (0°, 0°)", 0, 0, 0, "UTC"],
];
const canvas = $("#gl");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance", preserveDrawingBuffer: params.has("capture") });
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.outputColorSpace = THREE.SRGBColorSpace;

class Labels {
  constructor(el) { this.el = el; this.pool = []; this.n = 0; }
  begin() { this.n = 0; }
  add(x, y, text, cls, id) {
    let d = this.pool[this.n];
    if (!d) { d = document.createElement("div"); this.el.appendChild(d); this.pool.push(d); }
    if (d._t !== text) { d.textContent = text; d._t = text; }
    const c = "lbl " + cls; if (d.className !== c) d.className = c;
    d.style.left = x.toFixed(1) + "px"; d.style.top = y.toFixed(1) + "px"; d.style.display = "";
    this.n++;
  }
  end() { for (let i = this.n; i < this.pool.length; i++) this.pool[i].style.display = "none"; }
}
const labels = new Labels($("#labels"));
const sky = new SkyView(renderer, labels);
const sys = new SystemView(renderer, labels);

const S = {
  obs: { name: "São Paulo", lat: -23.5505, lon: -46.6333, elev: 760, tz: "America/Sao_Paulo" },
  simJD: dateToJd(new Date()), speed: 1, playing: true, view: "sky", sliderBase: null,
  skyData: null, skyLoading: false, sysData: null, sysLoading: false, satData: null, satLoading: false,
  selected: null, lastNowRefresh: 0,
};
try { const saved = JSON.parse(localStorage.getItem("obs.loc") || "null"); if (saved && saved.lat !== undefined) S.obs = saved; } catch (e) { /* storage blocked */ }
if (params.has("lat")) S.obs = { name: params.get("name") || "Custom", lat: +params.get("lat"), lon: +params.get("lon"), elev: +(params.get("elev") || 0), tz: params.get("tz") || Intl.DateTimeFormat().resolvedOptions().timeZone };
if (params.has("t")) S.simJD = dateToJd(new Date(params.get("t")));
if (params.has("speed")) S.speed = +params.get("speed");
if (params.has("fov")) sky.fov = +params.get("fov");
sky.lookAt(params.has("alt") ? +params.get("alt") : 30, params.has("az") ? +params.get("az") : (S.obs.lat < 0 ? 180 : 0));
if (params.get("layers")) for (const kv of params.get("layers").split(",")) { const [k, v] = kv.split(":"); sky.layers[k] = v !== "0"; }

// --------------------------------------------------------------------------------------------- data I/O
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error((await r.json().catch(() => ({}))).detail || r.statusText);
  return r.json();
}
const isoOf = (jd) => jdToDate(jd).toISOString();
const qs = () => `lat=${S.obs.lat}&lon=${S.obs.lon}&elev=${S.obs.elev || 0}`;
function viewRaDec() {                                   // J2000 RA/Dec of the centre of the sky view
  const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(sky.camera.quaternion);
  const M = sky.Mj2000; const v = [0, 1, 2].map((c) => M[0][c] * dir.x + M[1][c] * dir.y + M[2][c] * dir.z);
  return [((Math.atan2(v[1], v[0]) * R2D) + 360) % 360, Math.asin(v[2]) * R2D];
}
function aboveHorizon() {
  const out = new Set(), M = sky.Mj2000, cat = sky.cat;
  if (!M || !cat) return out;
  for (let i = 0; i < cat.hr.length; i++) { const s = radecVec(cat.ra[i], cat.dec[i]); if (M[1][0] * s[0] + M[1][1] * s[1] + M[1][2] * s[2] > 0) out.add(cat.hr[i]); }
  return out;
}
const ctx = { iso: () => isoOf(S.simJD), qs, viewRaDec, aboveHorizon,
  showSmallBody: async (key) => { const tr = await api(`/api/smallbodies/track?body=${key}&t=${isoOf(S.simJD)}&days=600`); sys.setSmallBody(tr); if (S.view !== "system") switchView("system"); } };

async function loadSky(force) {
  if (S.skyLoading) return;
  const d = S.skyData;
  const sp = Math.abs(S.speed);
  const span = sp > 600 ? 48 : sp > 30 ? 12 : 4;
  if (!force && d && d.observer.lat === S.obs.lat && d.observer.lon === ((S.obs.lon + 540) % 360) - 180) {
    const j = d.samples.jd, margin = (j[j.length - 1] - j[0]) * 0.2;
    if (S.simJD > j[0] + margin && S.simJD < j[j.length - 1] - margin) return;
  }
  S.skyLoading = true;
  try {
    const data = await api(`/api/sky?${qs()}&t=${isoOf(S.simJD)}&span_h=${span}&step_min=${(span * 60 / 72).toFixed(2)}`);
    S.skyData = data; sky.setSky(data); renderTonight(); if (S.selected) renderObject(S.selected);
  } catch (e) { toast("Sky: " + e.message); }
  S.skyLoading = false;
}
async function loadSystem(force) {
  if (S.sysLoading) return;
  const d = S.sysData;
  if (!force && d && S.simJD + 0.0008 > d.jd[0] + 3 && S.simJD < d.jd[d.jd.length - 1] - 3) return;
  S.sysLoading = true;
  try { const data = await api(`/api/system?t=${isoOf(S.simJD)}&days=60`); S.sysData = data; sys.setData(data); }
  catch (e) { toast("Solar system: " + e.message); }
  S.sysLoading = false;
}
async function loadSats(force) {
  if (S.satLoading || !sky.layers.satellites || Math.abs(S.speed) > 120) return;
  const d = S.satData;
  if (!force && d && S.simJD > d.jd[0] + 0.001 && S.simJD < d.jd[d.jd.length - 1] - 0.001) return;
  if (!force && d && d.satellites.length === 0 && performance.now() - (S.satTried || 0) < 60000) return;
  S.satLoading = true; S.satTried = performance.now();
  try { const data = await api(`/api/satellites?${qs()}&t=${isoOf(S.simJD)}&minutes=20&step_s=10&limit=60`); S.satData = data; sky.setSats(data); }
  catch (e) { /* satellites are optional */ }
  S.satLoading = false;
}

// ------------------------------------------------------------------------------------------------ the loop
let last = performance.now();
function resize() {
  const w = window.innerWidth, h = window.innerHeight;
  renderer.setSize(w, h, false);
  for (const c of [sky.camera, sys.camera]) { c.aspect = w / h; c.updateProjectionMatrix(); }
}
window.addEventListener("resize", resize); resize();

function frame() {
  const now = performance.now();               // (a capture script may drive this clock)
  const dt = Math.min(0.25, (now - last) / 1000); last = now;
  if (S.playing && !params.has("freeze")) S.simJD += dt * S.speed / 86400;
  const w = window.innerWidth, h = window.innerHeight;
  if (S.view === "sky") {
    loadSky(); loadSats();
    sky.update(S.simJD, now / 1000);
    sky.camera.updateProjectionMatrix(); sky.camera.updateMatrixWorld();
    sky.render(); sky.drawLabels(w, h);
  } else {
    loadSystem();
    sys.observer = S.obs;
    sys.update(S.simJD);
    sys.render(); sys.drawLabels(w, h);
  }
  if (S.view === "sky" && S.skyData && now - S.lastNowRefresh > 20000 && Math.abs(S.speed) <= 60) {
    S.lastNowRefresh = now; if (Math.abs(S.simJD - S.skyData.jd_utc) * 86400 > 60) loadSky(true);
  }
  updateClock();
  requestAnimationFrame(frame);
}

// ---------------------------------------------------------------------------------------------- the clock
function updateClock() {
  const d = jdToDate(S.simJD);
  const tz = S.obs.tz || undefined;
  let local;
  try { local = d.toLocaleString([], { timeZone: tz, weekday: "short", day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", second: "2-digit" }); }
  catch (e) { local = d.toLocaleString(); }
  $("#clock").textContent = local;
  const sun = sky.bodies?.sun?.alt;
  const lst = sky.lst / 15;
  $("#clockSub").textContent = `${d.toISOString().slice(0, 19).replace("T", " ")} UTC · LST ${Math.floor(lst)}h${String(Math.floor((lst % 1) * 60)).padStart(2, "0")} · ${S.obs.name || ""} ${num(S.obs.lat, 3)}°, ${num(S.obs.lon, 3)}°` + (sun !== undefined && S.view === "sky" ? ` · Sun ${num(sun, 1)}°` : "");
  const sp = S.speed;
  $("#speed").textContent = !S.playing ? "paused" : sp === 1 ? "×1" : `×${Math.abs(sp) >= 3600 ? (sp / 3600) + "h/s" : Math.abs(sp) >= 60 ? (sp / 60) + "m/s" : sp}`;
  $("#tPlay").textContent = S.playing ? "❚❚" : "▶";
}

// ------------------------------------------------------------------------------------------ the controls
const SPEEDS = [-86400, -3600, -600, -60, 1, 60, 600, 3600, 86400];
$("#tPlay").onclick = () => { S.playing = !S.playing; };
$("#tNow").onclick = () => { S.simJD = dateToJd(new Date()); S.speed = 1; S.playing = true; $("#tSlider").value = 0; S.sliderBase = null; loadSky(true); };
$("#tFwd").onclick = () => { const i = SPEEDS.indexOf(S.speed); S.speed = SPEEDS[Math.min(SPEEDS.length - 1, (i < 0 ? 4 : i) + 1)]; S.playing = true; };
$("#tBack").onclick = () => { const i = SPEEDS.indexOf(S.speed); S.speed = SPEEDS[Math.max(0, (i < 0 ? 4 : i) - 1)]; S.playing = true; };
$("#tSlider").oninput = (e) => {
  if (S.sliderBase === null) S.sliderBase = S.simJD;
  S.simJD = S.sliderBase + (+e.target.value) / 1440; S.playing = false;
};
$("#tSlider").onchange = () => { loadSky(); };
$("#tDate").onchange = (e) => { if (e.target.value) { S.simJD = dateToJd(new Date(e.target.value)); S.sliderBase = null; $("#tSlider").value = 0; loadSky(true); loadSystem(true); } };
$("#btnPanel").onclick = () => $("#panel").classList.toggle("open");
$("#tLayers").onclick = () => $("#hud").classList.toggle("showlayers");
if (window.innerWidth < 760 && !params.has("panel")) $("#panel").classList.remove("open");
if (params.get("panel") === "0") $("#panel").classList.remove("open");

const SIZE_K = params.has("size") ? +params.get("size") : 12;     // planet size exaggeration in the system view
sys.setSizeScale(SIZE_K);
const LAYERS = [["constellations", "Constellations"], ["names", "Names"], ["altaz", "Alt-az grid"], ["equatorial", "Equatorial grid + ecliptic"],
  ["milkyway", "Milky Way"], ["atmosphere", "Atmosphere"], ["ground", "Ground"], ["satellites", "Satellites"], ["planetsBig", "Planets ×60"]];
$("#layers").innerHTML = LAYERS.map(([k, t]) => `<label><input type="checkbox" data-k="${k}" ${sky.layers[k] ? "checked" : ""}>${t}</label>`).join("") +
  `<label class="sysonly"><input type="checkbox" id="logScale" checked>Log distances</label>` +
  `<label class="sysonly">Planet size ×<input type="range" id="sizeK" min="1" max="24" value="${SIZE_K}" step="0.5" style="width:90px"><span id="sizeKv">${SIZE_K}</span></label>`;
for (const cb of document.querySelectorAll("#layers input[data-k]")) cb.onchange = (e) => { sky.layers[e.target.dataset.k] = e.target.checked; if (e.target.dataset.k === "satellites") loadSats(true); };
$("#logScale").onchange = (e) => { sys.logScale = e.target.checked; sys._buildOrbits(); };
$("#sizeK").oninput = (e) => { sys.setSizeScale(+e.target.value); $("#sizeKv").textContent = e.target.value; };

// location
const sel = $("#city");
sel.innerHTML = CITIES.map((c, i) => `<option value="${i}">${c[0]}</option>`).join("") + `<option value="here">My location</option>`;
const ci = CITIES.findIndex((c) => Math.abs(c[1] - S.obs.lat) < 1e-3 && Math.abs(c[2] - S.obs.lon) < 1e-3);
sel.value = ci >= 0 ? ci : "here";
sel.onchange = () => {
  if (sel.value === "here") return locate();
  const c = CITIES[+sel.value];
  setObserver({ name: c[0], lat: c[1], lon: c[2], elev: c[3], tz: c[4] });
};
$("#btnLocate").onclick = locate;
function locate() {
  if (!navigator.geolocation) return toast("Geolocation not available");
  navigator.geolocation.getCurrentPosition((p) => {
    setObserver({ name: "My location", lat: +p.coords.latitude.toFixed(4), lon: +p.coords.longitude.toFixed(4), elev: Math.max(0, Math.round(p.coords.altitude || 0)), tz: Intl.DateTimeFormat().resolvedOptions().timeZone });
    sel.value = "here";
  }, (e) => toast("Location: " + e.message), { enableHighAccuracy: false, timeout: 10000 });
}
function setObserver(o) {
  S.obs = o;
  try { localStorage.setItem("obs.loc", JSON.stringify(o)); } catch (e) { /* ignore */ }
  $("#locText").textContent = `${num(o.lat, 3)}°, ${num(o.lon, 3)}°`;
  S.satData = null; loadSky(true); loadSats(true); renderSatsPanel();
}
$("#locText").textContent = `${num(S.obs.lat, 3)}°, ${num(S.obs.lon, 3)}°`;

function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), 4000); }

// --------------------------------------------------------------------------------------- pointer / touch
let drag = null; const pointers = new Map();
canvas.addEventListener("pointerdown", (e) => {
  pointers.set(e.pointerId, [e.clientX, e.clientY]);
  drag = { x: e.clientX, y: e.clientY, moved: 0, fov: sky.fov, dist: pinchDist() };
  canvas.setPointerCapture(e.pointerId);
});
canvas.addEventListener("pointermove", (e) => {
  if (!pointers.has(e.pointerId) || !drag) return;
  const prev = pointers.get(e.pointerId); pointers.set(e.pointerId, [e.clientX, e.clientY]);
  drag.moved += Math.abs(e.clientX - prev[0]) + Math.abs(e.clientY - prev[1]);
  if (S.view !== "sky" || sky.gyro) return;
  if (pointers.size === 2) { const d = pinchDist(); if (drag.dist) sky.fov = clampFov(drag.fov * drag.dist / d); return; }
  const k = sky.fov / window.innerHeight;
  sky.yaw = (sky.yaw - (e.clientX - prev[0]) * k + 360) % 360;
  sky.pitch = Math.max(-30, Math.min(89.5, sky.pitch + (e.clientY - prev[1]) * k));
});
canvas.addEventListener("pointerup", (e) => {
  pointers.delete(e.pointerId);
  if (drag && drag.moved < 6) onClick(e);
  if (!pointers.size) drag = null;
});
canvas.addEventListener("wheel", (e) => { if (S.view === "sky") { e.preventDefault(); sky.fov = clampFov(sky.fov * Math.exp(e.deltaY * 0.0012)); } }, { passive: false });
function pinchDist() { const p = [...pointers.values()]; return p.length === 2 ? Math.hypot(p[0][0] - p[1][0], p[0][1] - p[1][1]) : 0; }
function clampFov(f) { return Math.max(0.5, Math.min(110, f)); }
function onClick(e) {
  const x = (e.clientX / window.innerWidth) * 2 - 1, y = -(e.clientY / window.innerHeight) * 2 + 1;
  const hit = S.view === "sky" ? sky.pick(x, y) : sys.pick(x, y);
  if (!hit) return;
  S.selected = hit; renderObject(hit); openTab("object");
  if (S.view === "system" && hit.kind === "body") focusSystem(hit.id);
}
window.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.code === "Space") { S.playing = !S.playing; e.preventDefault(); }
  if (e.key === "+") sky.fov = clampFov(sky.fov / 1.25);
  if (e.key === "-") sky.fov = clampFov(sky.fov * 1.25);
});

// ------------------------------------------------------------------------------------------------- gyro
const hasOrientation = "DeviceOrientationEvent" in window && /Mobi|Android|iPhone|iPad/i.test(navigator.userAgent);
if (hasOrientation) $("#btnGyro").hidden = false;
$("#btnGyro").onclick = async () => {
  if (sky.gyro) { sky.gyro = false; window.removeEventListener("deviceorientation", onOrient); $("#btnGyro").textContent = "📱 Point"; $("#reticle").hidden = true; return; }
  try { if (typeof DeviceOrientationEvent.requestPermission === "function") { const r = await DeviceOrientationEvent.requestPermission(); if (r !== "granted") return toast("Motion permission denied"); } }
  catch (e) { return toast("Motion sensors unavailable"); }
  sky.gyro = true; $("#btnGyro").textContent = "✋ Drag"; $("#reticle").hidden = false;
  window.addEventListener("deviceorientationabsolute", onOrient); window.addEventListener("deviceorientation", onOrient);
  toast("Point your phone at the sky. Heading uses the compass: calibrate it if the sky looks rotated.");
};
const zee = new THREE.Vector3(0, 0, 1), q0 = new THREE.Quaternion(), q1 = new THREE.Quaternion(-Math.sqrt(0.5), 0, 0, Math.sqrt(0.5));
function onOrient(e) {
  if (!sky.gyro || e.beta === null) return;
  if (e.type === "deviceorientation" && e.absolute === false && e.webkitCompassHeading === undefined && window.ondeviceorientationabsolute !== undefined) return;
  // alpha: rotation about the vertical, 0 = north for absolute events; iOS reports the compass heading separately
  const alpha = (e.webkitCompassHeading !== undefined ? (360 - e.webkitCompassHeading) : e.alpha) * D2R;
  const beta = e.beta * D2R, gamma = e.gamma * D2R, orient = (screen.orientation?.angle || window.orientation || 0) * D2R;
  const euler = new THREE.Euler(beta, alpha, -gamma, "YXZ");
  const q = new THREE.Quaternion().setFromEuler(euler).multiply(q1).multiply(q0.setFromAxisAngle(zee, -orient));
  sky.camera.quaternion.copy(q);
}

// ------------------------------------------------------------------------------------ view transitions
$("#vSky").onclick = () => switchView("sky");
$("#vSys").onclick = () => switchView("system");
function setViewButtons() { $("#vSky").classList.toggle("on", S.view === "sky"); $("#vSys").classList.toggle("on", S.view === "system"); document.body.dataset.view = S.view; }
async function switchView(v, opts = {}) {
  if (v === S.view) return;
  const fade = $("#fade");
  if (v === "system") {
    await loadSystem(true);
    // widen the field of view in the sky, fade, then start from just above the observer and fly out
    const f0 = sky.fov; const t0 = performance.now();
    await new Promise((res) => { const step = () => { const k = Math.min(1, (performance.now() - t0) / 700); sky.fov = f0 + (110 - f0) * k * k; if (k < 1) requestAnimationFrame(step); else res(); }; step(); });
    fade.style.opacity = 1; await sleep(450);
    S.view = "system"; setViewButtons(); sys.controls.enabled = true; sys.focus = null;
    sys.observer = S.obs; sys.update(S.simJD);
    const a = sys.observerShot(); sys.camera.position.copy(a.pos); sys.controls.target.copy(a.target);
    sky.fov = f0;
    fade.style.opacity = 0;
    const mid = sys.planets.earth.grp.position.clone();
    sys.flyTo(mid.clone().add(a.pos.clone().sub(mid).normalize().multiplyScalar(2.2)).add(new THREE.Vector3(0, 1.2, 0)), mid, opts.ms1 || 2600, () => {
      const o = sys.overviewShot(); sys.flyTo(o.pos, o.target, opts.ms2 || 3200);
    });
  } else {
    sys.focus = null;
    const a = sys.observerShot();
    await new Promise((res) => sys.flyTo(a.pos, a.target, 2200, res));
    fade.style.opacity = 1; await sleep(450);
    S.view = "sky"; setViewButtons(); sys.controls.enabled = false; loadSky(true);
    fade.style.opacity = 0;
  }
}
function focusSystem(id) {
  sys.focus = id;
  const t = sys.focusPos();
  const f = id === "sun" ? 1 : (sys.sizeK || 3) / 3;          // planets and Moon are drawn f times enlarged
  const r = f * (id === "sun" ? 8 : id === "moon" ? 0.4 : (sys.planets[id]?.r ?? 0.3) * 7);
  const dir = sys.camera.position.clone().sub(t).normalize();
  sys.flyTo(t.clone().add(dir.multiplyScalar(r)), t, 1800);
}
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
window.__obs = { S, sky, sys, switchView, focusSystem, loadSky, ctx };          // for scripted captures (GIF / screenshots)

// ------------------------------------------------------------------------------------------------- panels
for (const b of document.querySelectorAll(".ptabs button")) b.onclick = () => openTab(b.dataset.tab);
function openTab(t) {
  for (const b of document.querySelectorAll(".ptabs button")) b.classList.toggle("on", b.dataset.tab === t);
  for (const s of document.querySelectorAll(".pbody")) s.classList.toggle("on", s.id === "tab-" + t);
  if (t === "sats") renderSatsPanel();
  if (t === "analyses" && !$("#tab-analyses").dataset.ready) renderAnalyses();
  if (t === "engine" && !$("#tab-engine").dataset.ready) renderEngine();
  $("#panel").classList.add("open");
}
const BNAMES = { sun: "Sun", moon: "Moon", mercury: "Mercury", venus: "Venus", mars: "Mars", jupiter: "Jupiter", saturn: "Saturn", uranus: "Uranus", neptune: "Neptune", earth: "Earth" };
const tfmt = (iso) => { if (!iso) return "—"; try { return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", timeZone: S.obs.tz }); } catch (e) { return iso.slice(11, 16); } };
const dfmt = (iso) => { try { return new Date(iso).toLocaleString([], { day: "2-digit", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit", timeZone: S.obs.tz }); } catch (e) { return iso; } };

function moonIcon(k, waxing, pa) {
  // illuminated fraction k, bright limb on the right when waxing (northern view), rotated by the position angle
  const r = 28, x = (2 * k - 1) * r;
  const sweep = waxing ? 1 : 0;
  const path = `M 32 ${32 - r} A ${r} ${r} 0 0 ${sweep} 32 ${32 + r} A ${Math.abs(x)} ${r} 0 0 ${k > 0.5 ? sweep : 1 - sweep} 32 ${32 - r} Z`;
  return `<svg class="moonicon" viewBox="0 0 64 64"><circle cx="32" cy="32" r="${r}" fill="#1b2130"/><g transform="rotate(${(90 - pa).toFixed(1)} 32 32)"><path d="${path}" fill="#e9e6da" transform="rotate(${waxing ? 0 : 0} 32 32)"/></g></svg>`;
}

function renderTonight() {
  const d = S.skyData; if (!d) return;
  const now = d.now, alm = d.almanac, m = d.moon;
  const tw = alm.twilight;
  const visible = Object.values(now).filter((b) => b.name !== "sun" && b.alt > 0).sort((a, b) => (a.magnitude ?? 9) - (b.magnitude ?? 9));
  let h = `<h3>Sky now</h3><div class="kv"><span>Sky</span><span>${esc(d.sky_state)}</span><span>Sun altitude</span><span>${num(d.sun_alt, 1)}°</span>
    <span>Local sidereal time</span><span>${num(d.lst_hours, 3)} h</span><span>Observer</span><span>${num(d.observer.lat, 3)}°, ${num(d.observer.lon, 3)}°, ${num(d.observer.elevation_m, 0)} m</span></div>`;
  h += `<h3>Moon</h3><div class="card">${moonIcon(m.illuminated_fraction, m.phase_longitude_deg < 180, m.bright_limb_pa_deg)}
    <b>${esc(m.name)}</b><br><span class="muted">${num(m.illuminated_fraction * 100, 1)}% lit · age ${num(m.age_days, 1)} d · ${num(m.distance_km, 0)} km</span>
    <br><span class="small muted">rise ${tfmt(alm.rise_transit_set.moon.rise[0])} · transit ${tfmt(alm.rise_transit_set.moon.transit[0])} · set ${tfmt(alm.rise_transit_set.moon.set[0])}</span><div style="clear:both"></div></div>`;
  h += `<h3>Twilight</h3><table><tr><th></th><th class="r">morning</th><th class="r">evening</th></tr>` +
    [["Sunrise / sunset", "sun"], ["Civil (−6°)", "civil"], ["Nautical (−12°)", "nautical"], ["Astronomical (−18°)", "astronomical"]]
      .map(([t, k]) => `<tr><td>${t}</td><td class="r">${tfmt(tw[k].rising[0])}</td><td class="r">${tfmt(tw[k].setting[0])}</td></tr>`).join("") + "</table>";
  h += `<h3>Visible now</h3>` + (visible.length ? `<table><tr><th>Object</th><th class="r">alt</th><th class="r">az</th><th class="r">mag</th></tr>` +
    visible.map((b) => `<tr data-b="${b.name}" style="cursor:pointer"><td>${BNAMES[b.name]}</td><td class="r">${num(b.alt, 1)}°</td><td class="r">${num(b.az, 0)}° ${compass(b.az)}</td><td class="r">${num(b.magnitude, 1)}</td></tr>`).join("") + "</table>" : `<p class="muted">No planet above the horizon.</p>`);
  h += `<h3>Rise · transit · set</h3><table><tr><th></th><th class="r">rise</th><th class="r">transit</th><th class="r">set</th></tr>` +
    Object.entries(alm.rise_transit_set).map(([b, r]) => `<tr><td>${BNAMES[b]}</td><td class="r">${tfmt(r.rise[0])}</td><td class="r">${tfmt(r.transit[0])}</td><td class="r">${tfmt(r.set[0])}</td></tr>`).join("") + "</table>";
  h += `<p class="small muted">Times in ${esc(S.obs.tz || "local")} for the day starting ${dfmt(alm.window_start)}. Rise/set: centre of the disc 34′ (refraction) + semi-diameter below the horizon, as in the USNO tables. ${esc(d.method)}.</p>`;
  $("#tab-tonight").innerHTML = h;
  for (const tr of $("#tab-tonight").querySelectorAll("tr[data-b]")) tr.onclick = () => { const b = tr.dataset.b; const o = d.now[b]; sky.lookAt(o.alt, o.az); S.selected = { kind: "body", id: b }; renderObject(S.selected); openTab("object"); };
}

async function renderObject(hit) {
  const el = $("#tab-object");
  const d = S.skyData;
  if (hit.kind === "body") {
    const b = hit.id;
    if (b === "earth") { el.innerHTML = `<h3>Earth</h3><p>Home. The pin marks you at ${num(S.obs.lat, 3)}°, ${num(S.obs.lon, 3)}°; the terminator is computed from the Sun's direction.</p>` + elementsHtml(S.sysData?.elements?.earth); return; }
    const o = d?.now?.[b];
    if (!o) { el.innerHTML = `<h3>${BNAMES[b]}</h3>`; return; }
    const r = d.almanac.rise_transit_set[b];
    let h = `<h3>${BNAMES[b]} <span class="badge ${o.alt > 0 ? "pass" : "neutral"}">${o.alt > 0 ? "above horizon" : "below horizon"}</span></h3>
    <div class="kv"><span>Altitude (refracted)</span><span>${num(o.alt_apparent, 3)}°</span><span>Azimuth</span><span>${num(o.az, 3)}° ${compass(o.az)}</span>
    <span>RA / Dec (of date)</span><span>${fmtRA(o.ra)} ${fmtDec(o.dec)}</span><span>RA / Dec (J2000, astrometric)</span><span>${fmtRA(o.ra_j2000)} ${fmtDec(o.dec_j2000)}</span>
    <span>Hour angle</span><span>${num(o.hour_angle / 15, 3)} h</span>
    <span>Distance</span><span>${o.distance_au > 0.1 ? num(o.distance_au, 5) + " AU" : num(o.distance_km, 0) + " km"}</span>
    <span>Light time</span><span>${o.light_time_s > 120 ? num(o.light_time_s / 60, 2) + " min" : num(o.light_time_s, 3) + " s"}</span>
    <span>Range rate (Doppler)</span><span>${num(o.range_rate_km_s, 3)} km/s</span>
    <span>Apparent diameter</span><span>${o.angular_diameter_arcsec > 120 ? num(o.angular_diameter_arcsec / 60, 2) + "′" : num(o.angular_diameter_arcsec, 2) + "″"}</span>
    <span>Magnitude</span><span>${num(o.magnitude, 2)}</span>`;
    if (o.airmass) h += `<span>Airmass · extinction</span><span>${num(o.airmass, 3)} · ${num(o.extinction_mag, 2)} mag</span>`;
    if (o.elongation !== undefined) h += `<span>Elongation from the Sun</span><span>${num(o.elongation, 2)}°</span><span>Phase angle · illuminated</span><span>${num(o.phase_angle, 1)}° · ${num(o.illuminated * 100, 1)}%</span>`;
    h += `<span>Rise · transit · set</span><span>${tfmt(r.rise[0])} · ${tfmt(r.transit[0])} · ${tfmt(r.set[0])}</span></div>`;
    h += `<div class="row"><button class="primary" id="oLook">Look at it</button>${b !== "sun" ? `<button class="ghost" id="oOrbit">Orbit & Doppler</button><button class="ghost" id="oPlan">Altitude tonight</button>` : ""}</div><div id="oMore"></div>`;
    el.innerHTML = h;
    $("#oLook").onclick = () => { if (S.view === "sky") { sky.lookAt(o.alt, o.az); sky.fov = Math.min(sky.fov, b === "moon" || b === "sun" ? 5 : 30); } else focusSystem(b); };
    if ($("#oOrbit")) $("#oOrbit").onclick = () => orbitInto($("#oMore"), b);
    if ($("#oPlan")) $("#oPlan").onclick = () => planInto($("#oMore"), b);
  } else if (hit.kind === "star") {
    el.innerHTML = `<p class="muted">Loading HR ${hit.id}…</p>`;
    try {
      const s = await api(`/api/analysis/star?hr=${hit.id}&t=${isoOf(S.simJD)}&${qs()}`);
      const sw = s.rgb ? `<span class="swatch" style="color:rgb(${s.rgb.map((x) => Math.round(x * 255)).join(",")});background:currentColor"></span>` : "";
      el.innerHTML = `<h3>${sw}${esc(s.name)} <span class="muted">${esc(s.designation)} · HR ${s.hr}</span></h3>
      <div class="kv"><span>Altitude · azimuth</span><span>${num(s.alt, 2)}° · ${num(s.az, 1)}° ${compass(s.az)}</span>
      <span>RA / Dec (of date)</span><span>${fmtRA(s.ra_date)} ${fmtDec(s.dec_date)}</span>
      <span>V magnitude · B−V</span><span>${num(s.v_mag, 2)} · ${num(s.b_v, 2)}</span><span>Spectral type</span><span>${esc(s.spectral_type)}</span>
      <span>Parallax</span><span>${s.parallax_mas ? num(s.parallax_mas, 1) + " mas" + (s.parallax_dynamical ? " (dynamical)" : "") : "—"}</span>
      <span>Distance</span><span>${s.distance_ly ? num(s.distance_ly, 1) + " ly · " + num(s.distance_pc, 2) + " pc" : "—"}</span>
      <span>The light we see left it in</span><span>${s.distance_ly ? Math.round(new Date().getFullYear() - s.distance_ly) : "—"}</span>
      <span>Absolute magnitude M_V</span><span>${num(s.abs_mag_v, 2)}</span><span>Effective temperature</span><span>${s.teff_k ? Math.round(s.teff_k) + " K" : "—"}</span>
      <span>Luminosity</span><span>${s.luminosity_lsun ? num(s.luminosity_lsun, s.luminosity_lsun < 10 ? 2 : 0) + " L☉" : "—"}</span>
      <span>Radius</span><span>${s.radius_rsun ? num(s.radius_rsun, s.radius_rsun < 10 ? 2 : 0) + " R☉" : "—"}</span>
      <span>Radial velocity</span><span>${s.radial_velocity_km_s !== null ? num(s.radial_velocity_km_s, 0) + " km/s" : "—"}</span></div>
      <p class="small muted">${esc(s.method)}. BSC5 parallaxes are pre-Hipparcos: distances beyond ~100 pc are uncertain.</p>
      <h3>Hertzsprung–Russell diagram</h3><div id="oHR"></div><p class="small muted">White: stars above your horizon now · gold: this star.</p>`;
      drawHR($("#oHR"), hit.id);
    } catch (e) { el.innerHTML = `<p>${esc(e.message)}</p>`; }
  } else if (hit.kind === "sat") {
    const o = hit.sat;
    el.innerHTML = `<h3>${esc(o.s.name)} <span class="muted">NORAD ${o.s.norad}</span></h3><div class="kv">
      <span>Altitude · azimuth</span><span>${num(o.alt, 2)}° · ${num(o.az, 1)}°</span><span>Sunlit</span><span>${o.lit ? "yes" : "no (in Earth's shadow)"}</span>
      <span>Magnitude</span><span>${num(o.mag, 1)}</span><span>TLE age</span><span>${num(o.s.tle_age_days, 2)} d</span></div><div id="oPass"></div>`;
    const p = await api(`/api/passes?${qs()}&t=${isoOf(S.simJD)}&days=2&norad=${o.s.norad}`).catch(() => null);
    if (p) $("#oPass").innerHTML = passesTable(p.passes);
  }
}

function elementsHtml(el) {
  if (!el) return "";
  return `<h3>Osculating elements (J2000 ecliptic${el.center === "earth" ? ", geocentric" : ""})</h3><div class="kv">
    <span>Semi-major axis a</span><span>${el.center === "earth" ? num(el.a_km, 0) + " km" : num(el.a_au, 6) + " AU"}</span><span>Eccentricity e</span><span>${num(el.e, 6)}</span>
    <span>Inclination i</span><span>${num(el.i_deg, 4)}°</span><span>Ascending node Ω</span><span>${num(el.Omega_deg, 3)}°</span>
    <span>Argument of perihelion ω</span><span>${num(el.omega_deg, 3)}°</span><span>Mean anomaly M</span><span>${num(el.M_deg, 3)}°</span>
    <span>Period</span><span>${el.period_days > 800 ? num(el.period_years, 3) + " yr" : num(el.period_days, 3) + " d"}</span>
    ${el.center !== "earth" ? `<span>Perihelion · aphelion</span><span>${num(el.perihelion_au, 4)} · ${num(el.aphelion_au, 4)} AU</span>` : ""}</div>`;
}
async function orbitInto(el, b) {
  el.innerHTML = `<p class="muted">Computing…</p>`;
  const o = await api(`/api/analysis/orbit?body=${b}&t=${isoOf(S.simJD)}&${qs()}`);
  el.innerHTML = elementsHtml(o.elements) + `<h3>Light time & Doppler</h3><div class="kv"><span>Distance</span><span>${num(o.distance_au, 6)} AU</span>
    <span>One-way light time</span><span>${num(o.light_time_s / 60, 3)} min</span><span>Range rate</span><span>${num(o.range_rate_km_s, 4)} km/s (${o.range_rate_km_s > 0 ? "receding" : "approaching"})</span>
    <span>Hα 656.281 nm observed at</span><span>${num(o.doppler.observed_nm, 5)} nm</span><span>Shift</span><span>${num(o.doppler.shift_pm, 2)} pm (β = ${o.doppler.beta.toExponential(3)})</span></div><p class="small muted">${esc(o.method)}</p>`;
}
async function planInto(el, name) {
  el.innerHTML = `<p class="muted">Computing…</p>`;
  const p = await api(`/api/analysis/target?name=${encodeURIComponent(name)}&t=${isoOf(S.simJD)}&${qs()}`);
  const w = p.best_window;
  el.innerHTML = `<h3>Altitude, next 24 h</h3><div id="pChart"></div>` + (w ? `<div class="card"><b>Best window</b>: ${dfmt(w.start)} – ${tfmt(w.end)} (${num(w.hours, 1)} h dark, above ${p.min_alt}°)<br>
    highest at ${tfmt(w.best)}: ${num(w.max_alt, 1)}°, airmass ${num(w.airmass, 2)}; Moon ${num(w.moon_separation, 0)}° away at ${num(w.moon_alt, 0)}° altitude</div>`
    : `<p class="muted">No dark-sky window with the target above ${p.min_alt}° in the next 24 h.</p>`);
  const x = p.jd.map((j) => jdToDate(j).toISOString());
  PChart.mount($("#pChart"), { x, height: 200, yLabel: "deg", yMin: -30, yMax: 90,
    series: [{ name: BNAMES[name] || name, y: p.alt, color: "#7fb4ff", width: 2 }, { name: "Sun", y: p.sun_alt, color: "#ffcf6e" }, { name: "Moon", y: p.moon_alt, color: "#c8c8d8", dash: "4 3" }],
    hlines: [{ y: 0, color: "#55607a", label: "horizon" }, { y: -18, color: "#3a4a66", label: "astro. dark", dash: "2 3" }] });
}

let HR = null;
async function drawHR(el, hrSel) {
  HR = HR || await api("/api/analysis/hr");
  const W = el.clientWidth || 360, H = 300, L = 36, B = 26;
  const M = sky.Mj2000, cat = sky.cat;
  const idx = new Map(cat.hr.map((h, i) => [h, i]));
  const X = (bv) => L + (bv + 0.4) / 2.4 * (W - L - 6), Y = (m) => 6 + (m + 8) / 24 * (H - 6 - B);
  let dots = "";
  for (let k = 0; k < HR.hr.length; k++) {
    const i = idx.get(HR.hr[k]); let up = false;
    if (i !== undefined && M) { const s = radecVec(cat.ra[i], cat.dec[i]); up = (M[1][0] * s[0] + M[1][1] * s[1] + M[1][2] * s[2]) > 0; }
    const c = HR.hr[k] === hrSel ? "#ffcf6e" : up ? "#ffffff" : "#3d5277";
    const r = HR.hr[k] === hrSel ? 5 : up ? 1.8 : 1.3;
    dots += `<circle cx="${X(HR.bv[k]).toFixed(1)}" cy="${Y(HR.abs_mag[k]).toFixed(1)}" r="${r}" fill="${c}" ${HR.hr[k] === hrSel ? 'stroke="#000"' : ""}/>`;
  }
  let ax = "";
  for (const bv of [-0.4, 0, 0.5, 1, 1.5, 2]) ax += `<text x="${X(bv)}" y="${H - 8}" text-anchor="middle">${bv}</text>`;
  for (const m of [-6, -3, 0, 3, 6, 9, 12, 15]) ax += `<text x="${L - 4}" y="${Y(m) + 3}" text-anchor="end">${m}</text>`;
  el.innerHTML = `<svg width="${W}" height="${H}" style="font:10px system-ui;fill:#8d9bb3"><defs><linearGradient id="hrg" x1="0" x2="1"><stop offset="0" stop-color="#9bb0ff" stop-opacity=".12"/><stop offset="1" stop-color="#ff7b3a" stop-opacity=".12"/></linearGradient></defs>
    <rect x="${L}" y="6" width="${W - L - 6}" height="${H - 6 - B}" fill="url(#hrg)"/>${dots}${ax}
    <text x="${W / 2}" y="${H}" text-anchor="middle">B − V (colour → temperature)</text><text x="10" y="${H / 2}" transform="rotate(-90 10 ${H / 2})" text-anchor="middle">M_V</text></svg>`;
}

function passesTable(ps) {
  if (!ps || !ps.length) return `<p class="muted">No passes above 10° in the window.</p>`;
  return `<table><tr><th>Satellite</th><th>Rise</th><th class="r">max</th><th>Set</th><th class="r">mag</th></tr>` + ps.map((p) =>
    `<tr${p.visible ? ' style="color:#d8fff0"' : ' class="muted"'}><td>${esc(p.satellite.replace(" (ZARYA)", ""))}</td><td>${dfmt(p.rise_utc)} ${compass(p.rise_az ?? 0)}</td>
     <td class="r">${num(p.max_alt, 0)}° ${compass(p.culmination_az)}</td><td>${tfmt(p.set_utc)} ${compass(p.set_az ?? 0)}</td><td class="r">${p.visible ? num(p.brightest_mag, 1) : "—"}</td></tr>`).join("") +
    `</table><p class="small muted">Bright rows: visible (satellite sunlit, sky dark, above 10°). Magnitude from the standard magnitude at 1000 km and the diffuse-sphere phase law.</p>`;
}
async function renderSatsPanel() {
  const el = $("#tab-sats");
  el.innerHTML = `<p class="muted">Computing passes…</p>`;
  try {
    const p = await api(`/api/passes?${qs()}&t=${isoOf(S.simJD)}&days=3`);
    const src = p.source || {};
    const err = src.groups ? Object.values(src.groups).find((g) => g.error) : null;
    let h = `<h3>Next passes: ISS, Tiangong, Hubble</h3>` + passesTable(p.passes);
    if (!p.passes.length && err) h = `<h3>Satellites</h3><div class="card">No current TLEs: CelesTrak could not be reached from the server (${esc(err.error)}). The pass predictor is validated offline against an independent implementation (Engine tab).</div>`;
    h += `<p class="small muted">${esc(p.method)}. TLEs: CelesTrak GP data, refreshed every 6 h.</p><div id="decayBox"></div>`;
    el.innerHTML = h;
    renderDecay($("#decayBox"), ctx);
  } catch (e) { el.innerHTML = `<p>${esc(e.message)}</p>`; }
}

// --------------------------------------------------------------------------------------------- analyses
function renderAnalyses() {
  const el = $("#tab-analyses"); el.dataset.ready = 1;
  const opts = Object.entries(BNAMES).filter(([k]) => k !== "earth").map(([k, v]) => `<option value="${k}">${v}</option>`).join("");
  el.innerHTML = `
  <h3>Angular separation</h3><div class="row"><input id="aA" type="text" value="moon" list="blist"> ↔ <input id="aB" type="text" value="jupiter" list="blist"><button class="primary" id="aSep">Measure</button></div>
  <datalist id="blist">${opts}<option value="HR 2491">Sirius</option><option value="HR 7001">Vega</option><option value="HR 5340">Arcturus</option><option value="HR 2061">Betelgeuse</option><option value="HR 1713">Rigel</option><option value="HR 6134">Antares</option></datalist><div id="aSepOut" class="small"></div>
  <h3>Best observing window tonight</h3><div class="row"><select id="aTgt">${opts}</select><button class="primary" id="aPlan">Plan</button></div><div id="aPlanOut"></div>
  <h3>Orbit, light time & Doppler</h3><div class="row"><select id="aOrb">${opts.replace('<option value="sun">Sun</option>', "")}</select><button class="primary" id="aOrbit">Compute</button></div><div id="aOrbitOut"></div>
  <h3>Black-body spectrum</h3><div class="row"><input id="aT" type="range" min="2500" max="30000" value="5772" step="10" style="flex:1"><span id="aTv" class="pill">5772 K</span></div><div id="aBB"></div>
  <h3>Events: phases, oppositions, elongations, eclipses, transits</h3><div class="row"><select id="aDays"><option value="90">next 3 months</option><option value="365" selected>next 12 months</option><option value="730">next 2 years</option></select><button class="primary" id="aEv">List</button></div><div id="aEvOut"></div>
  <h3>Space weather and aurora</h3><div id="aSW"><button class="ghost" id="aSWgo">Kp forecast and aurora chance here</button></div>
  <h3>Observing conditions (next 48 h)</h3><div id="aWX"><button class="ghost" id="aWXgo">Clouds, transparency, seeing</button></div>
  <h3>Asteroids and close approaches</h3><div id="aSB"><button class="ghost" id="aSBgo">Apophis, Phaethon, Ceres…</button></div>
  <h3>Stellar ML on the HR diagram</h3><div id="aML"><button class="ghost" id="aMLgo">Cluster and classify the stars</button></div>
  <h3>Black hole with an accretion disc, in front of your sky</h3><div id="aBH"></div>
  <h3>Gravitational lensing: a black hole in front of your sky</h3><p class="small muted">A Schwarzschild black hole placed where you are looking; the real stars (Yale BSC) and Milky Way are lensed by null geodesics traced with the black-hole module of PINNeAPPle. Where the field has few bright stars the thin accretion disc is drawn too.</p>
  <div class="row"><label class="small">camera distance <input id="aRc" type="range" min="8" max="300" value="120"></label><select id="aDisc"><option value="auto">disc: auto</option><option value="off">no disc</option><option value="on">with disc</option></select><button class="primary" id="aLens">Render where I look</button></div><div id="aLensOut"></div>`;
  $("#aSWgo").onclick = () => renderSpaceWeather($("#aSW"), ctx);
  $("#aWXgo").onclick = () => renderConditions($("#aWX"), ctx);
  $("#aSBgo").onclick = () => renderSmallBodies($("#aSB"), ctx);
  $("#aMLgo").onclick = () => renderStarsML($("#aML"), ctx);
  renderBlackHole($("#aBH"), ctx);
  $("#aSep").onclick = async () => {
    try { const r = await api(`/api/analysis/separation?a=${encodeURIComponent($("#aA").value)}&b=${encodeURIComponent($("#aB").value)}&t=${isoOf(S.simJD)}&${qs()}`);
      $("#aSepOut").innerHTML = `<div class="card"><b>${r.separation_dms}</b> (${num(r.separation_deg, 5)}°), position angle ${num(r.position_angle_deg, 2)}°<br><span class="muted">${esc(r.method)}</span></div>`; }
    catch (e) { $("#aSepOut").textContent = e.message; }
  };
  $("#aPlan").onclick = () => planInto($("#aPlanOut"), $("#aTgt").value);
  $("#aOrbit").onclick = () => orbitInto($("#aOrbitOut"), $("#aOrb").value);
  const bb = async () => {
    const T = +$("#aT").value; $("#aTv").textContent = T + " K";
    const r = await api(`/api/analysis/blackbody?T=${T}`);
    const c = `rgb(${r.rgb.map((x) => Math.round(x * 255)).join(",")})`;
    const step = 5;
    PChart.mount($("#aBB"), { x: r.wavelength_nm.filter((_, i) => i % step === 0).map((x) => Math.round(x)), height: 170, yLabel: "B_λ", series: [{ name: `${T} K`, y: r.radiance.filter((_, i) => i % step === 0), color: c, width: 2 }] });
    $("#aBB").insertAdjacentHTML("beforeend", `<p class="small"><span class="swatch" style="color:${c};background:${c}"></span>peak (Wien) ${num(r.peak_nm, 0)} nm · σT⁴ = ${r.flux_w_m2.toExponential(3)} W/m² · x axis: wavelength (nm)</p>`);
  };
  $("#aT").onchange = bb; bb();
  $("#aEv").onclick = async () => {
    $("#aEvOut").innerHTML = `<p class="muted">Searching (eclipses need the full Sun–Moon geometry; ~5–20 s)…</p>`;
    try {
      const r = await api(`/api/analysis/events?t=${isoOf(S.simJD)}&days=${$("#aDays").value}&${qs()}`);
      const icon = { moon: "☾", season: "☀", planet: "♁", eclipse: "◐", transit: "●" };
      $("#aEvOut").innerHTML = `<table>${r.events.filter((e) => e.kind !== "moon" || /Full|New/.test(e.event)).map((e) =>
        `<tr${e.kind === "eclipse" || e.kind === "transit" ? ' style="color:#ffcf6e"' : ""}><td>${icon[e.kind] || ""}</td><td>${dfmt(e.utc)}</td><td>${esc(e.event)}${e.elongation_deg !== undefined && e.kind === "planet" ? ` <span class="muted">${num(e.elongation_deg, 1)}°, mag ${num(e.magnitude, 1)}</span>` : ""}${e.local ? `<br><span class="small">here: max ${dfmt(e.local.max_utc)}, ${num(e.local.obscuration * 100, 0)}% of the Sun covered${e.local.visible ? "" : " (Sun below horizon)"}</span>` : ""}</td></tr>`).join("")}</table><p class="small muted">${esc(r.method)}</p>`;
    } catch (e) { $("#aEvOut").textContent = e.message; }
  };
  $("#aLens").onclick = async () => {
    // RA/Dec (J2000) of the centre of the view
    const dir = new THREE.Vector3(0, 0, -1).applyQuaternion(sky.camera.quaternion);
    const M = sky.Mj2000; const v = [0, 1, 2].map((c) => M[0][c] * dir.x + M[1][c] * dir.y + M[2][c] * dir.z);
    const ra = ((Math.atan2(v[1], v[0]) * R2D) + 360) % 360, dec = Math.asin(v[2]) * R2D;
    $("#aLensOut").innerHTML = `<p class="muted">Tracing geodesics (first render ~15 s, then instant for any direction)…</p>`;
    const rc = $("#aRc").value;
    const url = `/api/analysis/lensing.png?ra=${ra.toFixed(3)}&dec=${dec.toFixed(3)}&fov=60&r_cam=${rc}&width=480&disc=${$("#aDisc").value}`;
    let res, info;
    try { [res, info] = await Promise.all([fetch(url), api("/api/analysis/lensing")]); if (!res.ok) throw new Error(res.statusText); }
    catch (e) { $("#aLensOut").textContent = "Render failed (server busy?)."; return; }
    const disc = res.headers.get("X-Disc") === "1", nb = res.headers.get("X-Bright-Stars"), bm = res.headers.get("X-Bright-Mag");
    const img = new Image(); img.className = "lens"; img.src = URL.createObjectURL(await res.blob());
    img.onload = () => {
      $("#aLensOut").innerHTML = ""; $("#aLensOut").appendChild(img);
      $("#aLensOut").insertAdjacentHTML("beforeend", `<p class="small">Centre RA ${fmtRA(ra)}, Dec ${fmtDec(dec)} · 60° field · camera at ${rc} GM/c² · ${nb} stars brighter than V ${bm} in the field${disc ? " · thin accretion disc drawn" : ""}.</p>
      <table><tr><th>Black hole</th><th class="r">shadow predicted</th><th class="r">EHT ring</th><th class="r">σ</th></tr>${info.eht.map((r) => `<tr><td>${r.object}</td><td class="r">${num(r.predicted_uas, 1)} µas</td><td class="r">${r.observed_uas} ± ${r.observed_err_uas}</td><td class="r">${num(r.sigma, 2)}</td></tr>`).join("")}</table>
      <p class="small muted">Sun's deflection at the limb: ${num(info.sun_deflection_arcsec, 3)}″ (Eddington 1919: 1.75″). ${esc(info.method)}.</p>`);
    };
    img.onerror = () => { $("#aLensOut").textContent = "Render failed (server busy?)."; };
  };
}

// ----------------------------------------------------------------------------------------------- engine
async function renderEngine() {
  const el = $("#tab-engine"); el.dataset.ready = 1;
  el.innerHTML = `<h3>Physics-informed orbit surrogate</h3>
  <p class="small">A PINN learns a planet's heliocentric orbit for the next year: <i>r(t) = r<sub>Kepler</sub>(t) + τ² g<sub>θ</sub>(τ)</i>, trained on the residual of the full N-body equation of motion plus 8 anchor states, then judged against the classical integrator with split-conformal bands and physics checks.</p>
  <div class="row"><select id="eBody"><option>mars</option><option>venus</option><option>mercury</option><option>jupiter</option><option>saturn</option><option>earth</option></select><button class="primary" id="eRun">Train / show</button><span id="eStat" class="small muted"></span></div>
  <div id="eOut"></div>
  <h3>Other PINNeAPPle models in the observatory</h3>
  <table class="small"><tr><th>Model</th><th>Checked against</th><th></th></tr>
  <tr><td>Orbit-decay PINN (drag, NRLMSIS 2.0)</td><td>RK integration, input-uncertainty envelope, SGP4</td><td><button class="ghost" data-go="sats">Satellites</button></td></tr>
  <tr><td>Learned NEO propagator</td><td>full N-body integrator, held-out bodies, Apophis</td><td><button class="ghost" data-go="aSBgo">Open</button></td></tr>
  <tr><td>Small-body propagator</td><td>JPL Horizons (Ceres), JPL close approaches (Apophis 2029)</td><td><button class="ghost" data-go="aSBgo">Open</button></td></tr>
  <tr><td>Kp forecaster + aurora</td><td>observed Kp, held-out months; persistence</td><td><button class="ghost" data-go="aSWgo">Open</button></td></tr>
  <tr><td>Stellar classifiers / clustering</td><td>BSC5 MK spectral types, cross-validation</td><td><button class="ghost" data-go="aMLgo">Open</button></td></tr>
  <tr><td>Observing-conditions ensemble</td><td>(not verified: needs an observation history)</td><td><button class="ghost" data-go="aWXgo">Open</button></td></tr></table>
  <h3>Classical integrators</h3><div id="eInt"><p class="muted">Loading…</p></div>
  <h3>Validation of every method</h3><div id="eVal"><p class="muted">Loading…</p></div>`;
  $("#eRun").onclick = () => runSurrogate($("#eBody").value);
  for (const b of el.querySelectorAll("[data-go]")) b.onclick = () => {
    const g = b.dataset.go;
    if (g === "sats") return openTab("sats");
    openTab("analyses"); const btn = $("#" + g); if (btn) { btn.click(); btn.scrollIntoView({ block: "start" }); }
  };
  runSurrogate("mars"); loadIntegrators(); loadValidation();
}
async function runSurrogate(body) {
  $("#eStat").textContent = "training on the server (~30 s the first time)…";
  try {
    const r = await api(`/api/engine/surrogate?body=${body}&t=${isoOf(S.simJD)}&days=365`);
    $("#eStat").textContent = `trained in ${num(r.train_seconds, 1)} s · ${r.model.params} parameters`;
    const v = r.verdict.status;
    const x = r.t_days.map((t) => Math.round(t));
    let h = `<div class="card"><span class="badge ${v}">${v.replace(/_/g, " ")}</span> &nbsp;max error (test) <b>${num(r.summary.max_error_km_test, 0)} km</b> = <b>${num(r.summary.max_angle_arcsec_test, 3)}″</b> seen from the Sun · Kepler alone: ${num(r.summary.kepler_max_error_km, 0)} km · ${Math.round(r.summary.coverage * 100)}% of held-out states inside the ${Math.round((1 - r.summary.alpha) * 100)}% conformal band (median ±${num(r.summary.band_km_median, 0)} km)</div>
      <table><tr><th>Check</th><th class="r">value</th><th class="r">limit</th><th></th></tr>${r.checks.map((c) => `<tr><td>${esc(c.name)}</td><td class="r">${fmtv(c.value)}</td><td class="r">${fmtv(c.threshold)}</td><td><span class="badge ${c.status}">${c.status}</span></td></tr>`).join("")}</table>
      <div id="eErr"></div><div id="eEn"></div><div id="eLoss"></div>
      <p class="small muted">Reference: ${esc(r.reference.method)}; its total energy drift ${r.reference.energy_drift.toExponential(2)}, angular momentum ${r.reference.angmom_drift.toExponential(2)}. Energy / angular-momentum checks compare the PINN's error with the size of the perturbation-induced variation.</p>`;
    $("#eOut").innerHTML = h;
    PChart.mount($("#eErr"), { x, height: 190, yLog: true, yLabel: "km", series: [
      { name: "PINN − integrator", y: r.error_km.map((e) => Math.max(e, 1)), color: "#7fb4ff", width: 2 },
      { name: "Kepler two-body − integrator", y: r.kepler_error_km.map((e) => Math.max(e, 1)), color: "#ff6b6b", dash: "4 3" },
      { name: "conformal band (90%)", y: r.band_km, color: "#3ccf8e", dash: "2 2" }] });
    const e0 = r.energy_ref[0];
    PChart.mount($("#eEn"), { x, height: 160, yLabel: "Δε/|ε|", series: [
      { name: "two-body energy, integrator", y: r.energy_ref.map((e) => (e - e0) / Math.abs(e0)), color: "#ffcf6e" },
      { name: "two-body energy, PINN", y: r.energy_pinn.map((e) => (e - e0) / Math.abs(e0)), color: "#7fb4ff", dash: "4 3" }] });
    PChart.mount($("#eLoss"), { x: r.history.map((h) => h.iter), height: 150, yLog: true, yLabel: "loss", series: [
      { name: "physics residual", y: r.history.map((h) => h.physics), color: "#c58cff" }, { name: "anchor data", y: r.history.map((h) => h.data), color: "#3ccf8e" }] });
  } catch (e) { $("#eStat").textContent = e.message; }
}
const fmtv = (v) => (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5 ? (+v).toExponential(2) : (+v).toPrecision(3));
async function loadIntegrators() {
  try {
    const r = await api("/api/engine/integrators?years=2&dt_hours=6");
    const x = r.t_days.map((t) => Math.round(t)), col = { leapfrog: "#ffcf6e", yoshida4: "#3ccf8e", rk4: "#ff6b6b" };
    $("#eInt").innerHTML = `<div id="eIntE"></div><div id="eIntP"></div><p class="small muted">Sun, 8 planets and the Moon from DE421 at J2000, 6-hour steps for two years. Symplectic schemes keep the energy error bounded; RK4 drifts. Against DE421 the Newtonian point-mass model misses relativity and asteroids (Mars: ~${num(r.methods.yoshida4.error_vs_de421_km.mars.at(-1), 0)} km after 2 yr).</p>`;
    PChart.mount($("#eIntE"), { x, height: 170, yLog: true, yLabel: "|ΔE/E|", series: Object.entries(r.methods).map(([m, d]) => ({ name: `${m} (${d.force_evals} force evals)`, y: d.rel_energy_error.map((v) => Math.max(Math.abs(v), 1e-16)), color: col[m] })) });
    PChart.mount($("#eIntP"), { x, height: 160, yLog: true, yLabel: "km", series: Object.entries(r.methods).map(([m, d]) => ({ name: `${m}: Mars − DE421`, y: d.error_vs_de421_km.mars.map((v) => Math.max(v, 1e-3)), color: col[m] })) });
  } catch (e) { $("#eInt").textContent = e.message; }
}
async function loadValidation() {
  const r = await api("/api/validation").catch(() => null);
  if (!r || r.running || !r.rows) { $("#eVal").innerHTML = `<p class="muted">Running the validation suite on the server…</p>`; return setTimeout(loadValidation, 4000); }
  $("#eVal").innerHTML = `<table><tr><th>Method</th><th>Reference</th><th class="r">result</th><th></th></tr>${r.rows.map((x) =>
    `<tr><td>${esc(x.method)}</td><td class="small">${esc(x.reference)}</td><td class="r">${fmtv(x.value)} ${esc(x.unit)}<br><span class="small muted">limit ${fmtv(x.threshold)} · n=${x.n}</span></td><td><span class="badge ${x.status}">${x.status}</span></td></tr>`).join("")}</table>
    <p class="small muted">Computed by pinneapple_physics.astro.validation at server start (${num(r.seconds, 1)} s); the same checks run in the test suite.</p>`;
}

// ------------------------------------------------------------------------------------------------- start
setViewButtons();
(async () => {
  try {
    const cat = await api("/api/catalog");
    sky.setCatalog(cat); sys.setCatalog(cat);
  } catch (e) { toast("Catalogue: " + e.message); }
  await loadSky(true);
  last = performance.now();
  requestAnimationFrame(frame);
  window.__ready = true;
  if (params.get("view") === "system") switchView("system").then(() => {
    if (params.get("focus")) setTimeout(() => focusSystem(params.get("focus")), +(params.get("focusDelay") || 7000));
  });
  if (params.get("tab")) openTab(params.get("tab"));
  if (params.get("select")) {                    // e.g. select=star:2491 or select=body:saturn
    const [kind, id] = params.get("select").split(":");
    S.selected = { kind, id: kind === "star" ? +id : id }; renderObject(S.selected); openTab("object");
  }
})();
