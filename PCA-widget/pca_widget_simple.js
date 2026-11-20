// Simplified PCA widget as a standalone script.
// Usage: <div id="pca-widget"></div> <script src="pca_widget_simple.js"></script>
// It mounts into #pca-widget if that exists, otherwise at the end of <body>.
(() => {
const NS = "http://www.w3.org/2000/svg";

const CSS = `
.pcaw { --bg: #fff; --ink: #1f2328; --muted: #6b7280; --grid: #e5e7eb; --line: #b0b0b0;
        background: var(--bg); color: var(--ink); font: 14px system-ui, sans-serif; }
.pcaw .row { display: flex; gap: 24px; align-items: flex-start; flex-wrap: wrap; margin-bottom: 16px; }
.pcaw .matrix { display: grid; gap: 4px; }
.pcaw .matrix input { width: 56px; padding: 3px 4px; font: 13px ui-monospace, monospace; text-align: right; }
.pcaw .matrix .lbl { color: var(--muted); font: 12px ui-monospace, monospace; align-self: center; padding-right: 4px; }
.pcaw .dim { width: 56px; font-size: 14px; }
.pcaw h3 { margin: 0 0 6px; font-size: 14px; font-weight: 600; }
.pcaw label b { font-size: 14px; font-weight: 600; }
.pcaw .note { color: var(--muted); font-size: 12px; margin-top: 4px; max-width: 420px; }
.pcaw svg text { font: 11px system-ui, sans-serif; fill: var(--ink); }
.pcaw svg .muted { fill: var(--muted); }
.pcaw .quad { display: grid; grid-template-columns: 206px 206px; gap: 8px; }
.pcaw svg.small text { font-size: 9px; }`;

const HTML = `
<div class="row"><label><b>Dimensions</b> d = <input class="dim" type="number" min="1" max="20" value="7"></label></div>
<div class="row"><div class="matrix"></div></div>
<div class="row">
  <div><h3>Vectors projected on PC1, PC2</h3><svg class="sc" data-ky="1" width="420" height="420"></svg><div class="note"></div></div>
  <div><h3>PC3–PC6 vs PC1</h3>
    <div class="quad">
      <svg class="sc small" data-ky="2" width="206" height="206"></svg><svg class="sc small" data-ky="3" width="206" height="206"></svg>
      <svg class="sc small" data-ky="4" width="206" height="206"></svg><svg class="sc small" data-ky="5" width="206" height="206"></svg>
    </div>
  </div>
</div>`;

const style = document.createElement("style");
style.textContent = CSS;
document.head.appendChild(style);
const root = document.createElement("div");
root.className = "pcaw";
root.innerHTML = HTML;
(document.getElementById("pca-widget") || document.body).appendChild(root);

const q = sel => root.querySelector(sel);
const dimInput = q(".dim"), matrix = q(".matrix"), note = q(".note");
let d = 7, cells = [];

// d×d grid of inputs; default row k = (1, 0.001, 0, …, 0, 0.001) cyclically shifted right by k
function buildMatrix() {
  d = Math.max(1, Math.min(20, parseInt(dimInput.value) || 7));
  matrix.innerHTML = "";
  matrix.style.gridTemplateColumns = `auto repeat(${d}, auto)`;
  const v1 = Array.from({length: d}, (_, j) => j === 0 ? 1 : (j === 1 || j === d - 1) ? 0.001 : 0);
  cells = [];
  for (let i = 0; i < d; i++) {
    const l = document.createElement("div");
    l.className = "lbl"; l.textContent = `v${i + 1}`;
    matrix.appendChild(l);
    cells.push(Array.from({length: d}, (_, j) => {
      const inp = document.createElement("input");
      inp.type = "text"; inp.inputMode = "decimal"; inp.value = v1[((j - i) % d + d) % d];
      return matrix.appendChild(inp);
    }));
  }
  update();
}

const getX = () => cells.map(r => r.map(inp => parseFloat(inp.value.replace(",", ".")) || 0));

// Jacobi eigen-decomposition of symmetric A. Returns {vals, vecs} with vecs[:, i] the i-th eigenvector.
function jacobiEig(A) {
  const n = A.length, a = A.map(r => r.slice());
  const V = Array.from({length: n}, (_, i) => Array.from({length: n}, (_, j) => +(i === j)));
  for (let sweep = 0; sweep < 100; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] ** 2;
    if (off < 1e-24) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      if (Math.abs(a[p][q]) < 1e-300) continue;
      const th = (a[q][q] - a[p][p]) / (2 * a[p][q]);
      const t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) {          // columns p, q
        const akp = a[k][p], akq = a[k][q];
        a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
      }
      for (let k = 0; k < n; k++) {          // rows p, q
        const apk = a[p][k], aqk = a[q][k];
        a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
      }
      for (let k = 0; k < n; k++) {
        const vkp = V[k][p], vkq = V[k][q];
        V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
      }
    }
  }
  return {vals: a.map((r, i) => r[i]), vecs: V};
}

// PCA of the mean-centred rows via eigen-decomposition of XᵀX: σ_i = √λ_i, projection = X v_i
function update() {
  const X0 = getX();
  const mu = X0[0].map((_, j) => X0.reduce((s, r) => s + r[j], 0) / d);
  const X = X0.map(r => r.map((x, j) => x - mu[j]));
  const C = Array.from({length: d}, (_, i) => Array.from({length: d}, (_, j) =>
    X.reduce((s, r) => s + r[i] * r[j], 0)));
  const {vals, vecs} = jacobiEig(C);
  const order = vals.map((_, i) => i).sort((i, j) => vals[j] - vals[i]);
  const sig = order.map(i => Math.sqrt(Math.max(vals[i], 0)));
  const P = order.slice(0, 6).map(c => X.map(r => r.reduce((s, x, j) => s + x * vecs[j][c], 0)));
  root.querySelectorAll("svg.sc").forEach(svg => drawScatter(svg, P, 0, +svg.dataset.ky));
  const tol = 1e-6 * (sig[0] || 1);
  note.textContent = d > 2 && Math.abs(sig[1] - sig[2]) < tol
    ? "PC2 and PC3 have equal variance, so PC2 is not unique: the plot shows one arbitrary choice within the degenerate subspace."
    : "";
}

function el(tag, attrs, parent, text) {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (text !== undefined) e.textContent = text;
  parent.appendChild(e);
  return e;
}
const fmt = x => Math.abs(x) < 1e-10 ? "0" : (+x.toPrecision(3)).toString();

// PC(ky+1) vs PC(kx+1), equal scale on both axes, box always includes the origin
function drawScatter(svg, P, kx, ky) {
  const W = +svg.getAttribute("width");
  svg.innerHTML = "";
  if (ky >= P.length) {
    el("text", {x: W / 2, y: W / 2, "text-anchor": "middle", class: "muted"}, svg, `no PC${ky + 1} for d = ${d}`);
    return;
  }
  const px = P[kx], py = P[ky];
  const f = W / 420, pad = 36 * f, r = Math.max(5 * f, 3.5), off = 8 * f;
  const xs = px.concat([0]), ys = py.concat([0]);
  const cx = (Math.max(...xs) + Math.min(...xs)) / 2, cy = (Math.max(...ys) + Math.min(...ys)) / 2;
  const half = 1.12 * Math.max(Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys), 1e-9) / 2;
  const sx = x => pad + (x - cx + half) / (2 * half) * (W - 2 * pad);
  const sy = y => W - pad - (y - cy + half) / (2 * half) * (W - 2 * pad);
  el("line", {x1: pad, x2: W - pad, y1: sy(0), y2: sy(0), stroke: "var(--grid)"}, svg);
  el("line", {x1: sx(0), x2: sx(0), y1: pad, y2: W - pad, stroke: "var(--grid)"}, svg);
  el("text", {x: W - pad, y: sy(0) - 6 * f, "text-anchor": "end", class: "muted"}, svg, `PC${kx + 1}`);
  el("text", {x: sx(0) + 6 * f, y: pad + 4 * f, class: "muted"}, svg, `PC${ky + 1}`);
  // faint polygon v1 → v2 → … → v1
  const pts = px.map((x, i) => `${sx(x)},${sy(py[i])}`).join(" ");
  el("polygon", {points: pts, fill: "none", stroke: "var(--line)", "stroke-width": 1.5}, svg);
  px.forEach((x, i) => {
    const g = el("g", {}, svg);
    el("circle", {cx: sx(x), cy: sy(py[i]), r: r, fill: `hsl(${360 * i / d}, 75%, 45%)`, stroke: "var(--bg)", "stroke-width": 2 * f}, g);
    el("circle", {cx: sx(x), cy: sy(py[i]), r: 12, fill: "transparent"}, g);
    el("text", {x: sx(x) + off, y: sy(py[i]) - off}, g, `v${i + 1}`);
    el("title", {}, g, `v${i + 1}: (${fmt(x)}, ${fmt(py[i])})`);
  });
}

dimInput.addEventListener("change", buildMatrix);
matrix.addEventListener("input", update);
buildMatrix();
})();
