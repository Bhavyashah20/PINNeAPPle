// Pure validity / physics helpers shared by the modes and unit-tested against the Python adapters.

/** Heat-sink validity mask, as HeatsinkForced.in_range: the card's box plus fin gap, source fractions, n >= 2.
 *  row: the 12 inputs in card order. */
export function heatsinkInRange(row, names, vr) {
  const v = Object.fromEntries(names.map((k, i) => [k, row[i]]));
  for (const k of names) if (vr[k] && Array.isArray(vr[k])) {
    const [lo, hi] = vr[k], tol = 1e-9 * Math.max(Math.abs(lo), Math.abs(hi), 1);
    if (v[k] < lo - tol || v[k] > hi + tol) return false;
  }
  const gap = (v.base_width_mm - v.n_fins * v.fin_thickness_mm) / Math.max(v.n_fins - 1, 1);
  const inside = (x, k) => x >= vr[k][0] - 1e-9 && x <= vr[k][1] + 1e-9;
  return inside(gap, "fin_gap_mm") && inside(v.source_width_mm / v.base_width_mm, "source_frac_w")
    && inside(v.source_depth_mm / v.base_depth_mm, "source_frac_d") && v.n_fins >= 2;
}

/** Wong-Reece Mohr-Coulomb traction limit c b R theta + Fz tan(phi), theta = acos(1 - z/R). */
export function tractionLimit(Fz, z, c) {
  const th = Math.acos(Math.min(Math.max(1 - z / c.R_m, -1), 1));
  return c.c_Pa * c.b_m * c.R_m * th + Fz * Math.tan((c.phi_deg * Math.PI) / 180);
}
