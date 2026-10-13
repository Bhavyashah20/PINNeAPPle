// CST airfoil geometry: a line-by-line port of pinneapple_design/aero/geometry.py (tested against Python fixtures
// in apps/playground/tests/geometry.test.mjs).
//
// A shape is 6 numbers: thickness weights T0..T2 and camber weights M0..M2 of a 2nd-order Bernstein CST,
// upper = M + T, lower = M - T, class function sqrt(x)(1-x), blunt trailing edge of 0.25 % chord.

export const PARAMS = ["T0", "T1", "T2", "M0", "M1", "M2"];
export const BOUNDS = [[0.10, 0.24], [0.08, 0.22], [0.08, 0.22], [-0.03, 0.08], [-0.02, 0.14], [-0.04, 0.12]];
export const TE_THICKNESS = 0.0025;
export const N_CST = 2;

const comb = (n, k) => { let r = 1; for (let i = 1; i <= k; i++) r = (r * (n - k + i)) / i; return r; };

export function bern(n, x) {
  const out = new Array(n + 1);
  for (let i = 0; i <= n; i++) out[i] = comb(n, i) * x ** i * (1 - x) ** (n - i);
  return out;
}

export function cstSurface(w, x, te) {
  const xc = Math.min(Math.max(x, 0), 1);
  const b = bern(w.length - 1, xc);
  let s = 0;
  for (let i = 0; i < w.length; i++) s += b[i] * w[i];
  return Math.sqrt(xc) * (1 - xc) * s + xc * te;
}

/** Upper and lower y at chord stations xs (chord 1, leading edge at 0). */
export function surfaces(shape, xs) {
  const T = shape.slice(0, 3), M = shape.slice(3, 6);
  const wu = M.map((m, i) => m + T[i]), wl = M.map((m, i) => m - T[i]);
  return [xs.map((x) => cstSurface(wu, x, TE_THICKNESS / 2)), xs.map((x) => cstSurface(wl, x, -TE_THICKNESS / 2))];
}

// as numpy.linspace: i * step + start, with the end point set exactly
export const linspace = (a, b, n) => {
  const step = (b - a) / (n - 1);
  return Array.from({ length: n }, (_, i) => (i === n - 1 && n > 1 ? b : n === 1 ? a : i * step + a));
};
export const cosine = (n) => linspace(0, Math.PI, n).map((t) => 0.5 * (1 - Math.cos(t)));

// small dense solver (Gaussian elimination, partial pivoting)
export function solve(A, b) {
  const n = b.length, M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / M[r][r];
  }
  return x;
}

function lstsq(rows, ys) {
  // normal equations with a QR-free 3x3 solve (the CST basis is well conditioned on these point sets)
  const n = rows[0].length, A = Array.from({ length: n }, () => new Array(n).fill(0)), b = new Array(n).fill(0);
  rows.forEach((r, k) => {
    for (let i = 0; i < n; i++) {
      b[i] += r[i] * ys[k];
      for (let j = 0; j < n; j++) A[i][j] += r[i] * r[j];
    }
  });
  return solve(A, b);
}

const basisRow = (x) => { const c = Math.sqrt(x) * (1 - x); return bern(N_CST, x).map((v) => v * c); };

/** Least-squares CST weights for given upper/lower coordinates -> shape vector (T, M). */
export function fitCst(xu, yu, xl, yl) {
  const one = (xs, ys, te) => {
    const xc = xs.map((x) => Math.min(Math.max(x, 1e-9), 1));
    return lstsq(xc.map(basisRow), ys.map((y, i) => y - xc[i] * te));
  };
  const wu = one(xu, yu, TE_THICKNESS / 2), wl = one(xl, yl, -TE_THICKNESS / 2);
  return [...wu.map((u, i) => (u - wl[i]) / 2), ...wu.map((u, i) => (u + wl[i]) / 2)];
}

/** CST fit of a NACA 4-digit airfoil, e.g. naca4("2412"). */
export function naca4(code, n = 200) {
  const m = +code[0] / 100, p = +code[1] / 10, t = +code.slice(2) / 100;
  const xs = cosine(n), xu = [], yu = [], xl = [], yl = [];
  for (const x of xs) {
    const yt = 5 * t * (0.2969 * Math.sqrt(x) - 0.126 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4);
    let yc = 0, dy = 0;
    if (m > 0) {
      yc = x < p ? (m / p ** 2) * (2 * p * x - x ** 2) : (m / (1 - p) ** 2) * (1 - 2 * p + 2 * p * x - x ** 2);
      dy = x < p ? ((2 * m) / p ** 2) * (p - x) : ((2 * m) / (1 - p) ** 2) * (p - x);
    }
    const th = Math.atan(dy);
    xu.push(x - yt * Math.sin(th)); yu.push(yc + yt * Math.cos(th));
    xl.push(x + yt * Math.sin(th)); yl.push(yc - yt * Math.cos(th));
  }
  return fitCst(xu, yu, xl, yl);
}

const trapezoid = (y, x) => { let s = 0; for (let i = 1; i < x.length; i++) s += 0.5 * (y[i] + y[i - 1]) * (x[i] - x[i - 1]); return s; };

/** Max thickness and camber (fraction of chord) and where they are, LE radius, area. */
export function properties(shape) {
  const x = linspace(0, 1, 401), [yu, yl] = surfaces(shape, x);
  const t = yu.map((u, i) => u - yl[i]), c = yu.map((u, i) => (u + yl[i]) / 2);
  let i = 0, k = 0;
  t.forEach((v, j) => { if (v > t[i]) i = j; });
  c.forEach((v, j) => { if (Math.abs(v) > Math.abs(c[k])) k = j; });
  const sel = (lo, hi) => t.filter((_, j) => x[j] >= lo && x[j] <= hi);
  return {
    thickness: t[i], thickness_at: x[i], camber: c[k], camber_at: x[k], le_radius: shape[0] ** 2 / 2,
    area: trapezoid(t, x), min_thickness_aft: Math.min(...sel(0.9, 1)), spar_depth: Math.min(...sel(0.2, 0.35)),
  };
}

// numpy.gradient on a uniform grid (2nd-order interior, 1st-order edges)
export function gradient(y, x) {
  const n = y.length, g = new Array(n);
  g[0] = (y[1] - y[0]) / (x[1] - x[0]);
  g[n - 1] = (y[n - 1] - y[n - 2]) / (x[n - 1] - x[n - 2]);
  for (let i = 1; i < n - 1; i++) g[i] = (y[i + 1] - y[i - 1]) / (x[i + 1] - x[i - 1]);
  return g;
}

/** [ok, reason]: rejects crossing surfaces and wavy ones, as geometry.valid(). */
export function valid(shape) {
  const x = linspace(0, 1, 401), [yu, yl] = surfaces(shape, x);
  for (let i = 1; i < x.length - 1; i++) if (yu[i] - yl[i] <= 0) return [false, "surfaces cross"];
  const d2 = gradient(gradient(yu, x), x);
  for (let i = 20; i < x.length - 5; i++) if (Math.abs(d2[i]) > 40) return [false, "wavy surface"];
  return [true, ""];
}

/** Closed outline (upper TE->LE, lower LE->TE) for drawing. */
export function outline(shape, n = 120) {
  const x = cosine(n), [yu, yl] = surfaces(shape, x), pts = [];
  for (let i = n - 1; i >= 0; i--) pts.push([x[i], yu[i]]);
  for (let i = 1; i < n; i++) pts.push([x[i], yl[i]]);
  return pts;
}

// ---- direct manipulation: three handles on each surface ------------------------------------------------------
// y(x_i) on a surface is linear in its 3 CST weights, so 3 handles per surface fix the weights exactly.
export const HANDLE_X = [0.06, 0.35, 0.75];
const HANDLE_A = HANDLE_X.map(basisRow);

export function handles(shape) {
  const [yu, yl] = surfaces(shape, HANDLE_X);
  return { upper: HANDLE_X.map((x, i) => [x, yu[i]]), lower: HANDLE_X.map((x, i) => [x, yl[i]]) };
}

/** Shape whose surfaces pass through the given handle heights (upper and lower y at HANDLE_X). */
export function shapeFromHandles(yUpper, yLower) {
  const wu = solve(HANDLE_A, yUpper.map((y, i) => y - HANDLE_X[i] * (TE_THICKNESS / 2)));
  const wl = solve(HANDLE_A, yLower.map((y, i) => y + HANDLE_X[i] * (TE_THICKNESS / 2)));
  return [...wu.map((u, i) => (u - wl[i]) / 2), ...wu.map((u, i) => (u + wl[i]) / 2)];
}

/** Per-parameter box check against the model's CST bounds. Returns the names out of range. */
export function outOfBounds(shape) {
  return PARAMS.filter((_, i) => shape[i] < BOUNDS[i][0] - 1e-9 || shape[i] > BOUNDS[i][1] + 1e-9);
}

/** The 8 network inputs of MLPEnsemble.inputs (kept for tests; the ONNX graph does this itself). */
export function encode(shape, alpha) {
  const s = shape.map((v, i) => ((v - BOUNDS[i][0]) / (BOUNDS[i][1] - BOUNDS[i][0])) * 2 - 1);
  return [...s, (alpha - 6) / 8, Math.sin((alpha * Math.PI) / 180) * 4];
}
