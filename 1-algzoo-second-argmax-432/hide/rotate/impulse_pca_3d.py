"""Interactive 3D view of the impulse response on the first three uncentered PCs (see impulse_pca_uncentered.py).
Fitted on the raw impulse h_T0..h_9 (T0 = 2), pulse units removed; all of h_0..h_9 is plotted, on equal axes.
Checkboxes add the inputs x = (1, 0, .., 2 at step p, .., 0), p = 1..9, projected on the same PCs.
Ten number boxes (pre-filled with 0) add any custom input (the RNN is re-run in the browser and projected on the same PCs).
A menu switches to a 2D plot of PC1 vs PC0 of the centred PCA (the top-left panel of impulse_pca.png); all trajectories
are then recomputed in the browser from their inputs.
Writes rotate/impulse_pca_3d.html (self-contained; drag to rotate, scroll to zoom)."""
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import numpy as np
import plotly.graph_objects as go

HIDE = Path(__file__).resolve().parent
ROOT = HIDE.parent.parent
sys.path.insert(0, str(HIDE))
from impulse_pca import run, pca, units, T0, INK, MUTED, W_ih, W_hh

H = run([1.0] + [0.0] * 9)[:, units]
U, S, Vt = np.linalg.svd(H[T0:], full_matrices=True)
Vt *= np.where(Vt.sum(1) < 0, -1, 1)[:, None]   # sign convention: loadings sum > 0 (mostly along +neurons)
var = S**2 / (S**2).sum()                       # fractions of the total second moment
P = H @ Vt[:3].T                                # scores (10, 3); t < T0 projected, not fitted
mean_c, Vt_c, _, var_c = pca()                  # centred PCA of impulse_pca.png (same fit steps and sign convention)
# Projections the menu switches between: scores = (h_units - mean) @ V.T.
TITLE_C = ("Impulse response x = (1, 0, …, 0): PC1 vs PC0, mean subtracted (as in the top-left panel of impulse_pca.png)"
           f"<br><sup>Fitted on h_{T0}..h_9 over the 13 non-pulse units (thick line: fitted steps). "
           f"PC0 + PC1 hold {100 * var_c[:2].sum():.2f}% of the variance. Drag to zoom, double-click to reset.</sup>")

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948", "#52514e"]   # p = 1..9
XS = {p: np.eye(10)[0] + 2 * np.eye(10)[p] for p in range(1, 10)}
EXTRA = {p: ("x = (" + ", ".join(str(int(v)) for v in x) + ")", COLORS[p - 1]) for p, x in XS.items()}
Q = {p: run(x)[:, units] @ Vt[:3].T for p, x in XS.items()}
ALL = np.vstack([P, *Q.values()])
HOVER = "%{customdata}<br>PC0 %{x:.2f}<br>PC1 %{y:.2f}<br>PC2 %{z:.2f}<extra></extra>"
HOVER2 = "%{customdata}<br>PC0 %{x:.2f}<br>PC1 %{y:.2f}<extra></extra>"
MARKER = dict(size=7, color=np.arange(10), colorscale="Viridis", cmin=0, cmax=9, line=dict(color="white", width=2))


def line(t, width):
    return go.Scatter3d(x=P[t, 0], y=P[t, 1], z=P[t, 2], mode="lines", hoverinfo="skip", showlegend=False,
                        line=dict(color=MUTED, width=width))


def extra(p):
    """Hidden trajectory for the second input at step p; t < p coincides with the impulse, so labels start at t = p."""
    name, color = EXTRA[p]
    return go.Scatter3d(x=Q[p][:, 0], y=Q[p][:, 1], z=Q[p][:, 2], mode="lines+markers+text", visible=False, showlegend=False,
                        line=dict(color=color, width=6), marker=MARKER, customdata=[f"{name}<br>t={t}" for t in range(10)],
                        text=[f"t={t}" if t >= p else "" for t in range(10)], textposition="top center",
                        textfont=dict(size=11, color=INK), hovertemplate=HOVER)


points = go.Scatter3d(
    x=P[:, 0], y=P[:, 1], z=P[:, 2], mode="markers+text", showlegend=False, customdata=[f"impulse<br>t={t}" for t in range(10)],
    text=[f"t={t}" for t in range(10)], textposition="top center", textfont=dict(size=13, color=INK),
    marker=dict(MARKER, colorbar=dict(title="t", len=0.6, thickness=14)), hovertemplate=HOVER)

axis = dict(backgroundcolor="white", gridcolor="#e4e4e4", zerolinecolor=MUTED, color=INK)
fig = go.Figure([line(np.arange(T0 + 1), 3), line(np.arange(T0, 10), 8), points, *map(extra, EXTRA)])
TITLE_U = ("Impulse response x = (1, 0, …, 0) on the first three PCs, no mean subtracted"
           f"<br><sup>Fitted on the impulse h_{T0}..h_9 over the 13 non-pulse units (thick line: fitted steps). "
           f"PC0–PC2 hold {100 * var[:3].sum():.2f}% of its second moment. Drag to rotate, scroll to zoom.</sup>")
MODES = {"uncentred": dict(label="3D: uncentred PC0–PC2", title=TITLE_U, mean=[0.0] * 13, V=Vt[:3].tolist(), hover=HOVER,
                           axes=[f"PC{k} ({100 * var[k]:.1f}%)" for k in range(3)]),
         "centred": dict(label="2D: centred PC1 vs PC0 (impulse_pca.png)", title=TITLE_C, mean=mean_c.tolist(), V=Vt_c[:2].tolist(),
                         hover=HOVER2, axes=[f"PC{k} ({100 * var_c[k]:.1f}%)" for k in range(2)])}
fig.update_layout(
    title=dict(font=dict(size=16, color=INK), yref="container", y=1, yanchor="top", pad=dict(t=20), text=TITLE_U),
    paper_bgcolor="white", margin=dict(l=0, r=0, t=80, b=0),
    scene=dict(aspectmode="data", dragmode="orbit", camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
               **{f"{a}axis": dict(title=f"PC{k} ({100 * var[k]:.1f}%)", range=[ALL[:, k].min() - 8, ALL[:, k].max() + 8], **axis)
                  for k, a in enumerate("xyz")}))

boxes = "".join(f'<label style="display:inline-block;margin:2px 14px 2px 0;cursor:pointer"><input type="checkbox" data-trace="{2 + p}"> '
                f'<span style="display:inline-block;width:18px;height:4px;border-radius:2px;background:{c};vertical-align:middle"></span> {n}</label>'
                for p, (n, c) in EXTRA.items())
panel = f"""
document.body.style.margin = 0; document.documentElement.style.overflowX = 'hidden';   // only the page height scrolls
const panel = document.createElement('div');
panel.id = 'inputs';
panel.style.cssText = 'position:absolute;top:80px;left:16px;right:16px;z-index:10;font:14px sans-serif;color:{INK};padding:4px 0';
panel.innerHTML = '<div class="base">' + {boxes!r} + '</div><div class="custom"></div>';
document.body.appendChild(panel);
const gd = document.getElementById('{{plot_id}}');
panel.addEventListener('change', e => Plotly.restyle(gd, {{visible: e.target.checked}}, [+e.target.dataset.trace]));
// Push the 3D scene below the checkbox panel; redone when the panel grows or rewraps.
// The scene height is fixed by the window and the built-in checkboxes; custom-input rows make the page taller (it scrolls).
const fit = () => {{ const base = panel.querySelector('.base').offsetHeight, extra = panel.offsetHeight - base;
  const h = Math.max(innerHeight - form.offsetHeight, 84 + base + 400) + extra;
  gd.style.height = h + 'px';   // the div must grow too, or the plot spills over the controls below it
  Plotly.relayout(gd, {{'margin.t': 84 + panel.offsetHeight, height: h}}); }};
window.addEventListener('resize', fit);
"""
# Custom input: the RNN runs in the browser on the same weights and is projected on the same three PCs.
custom = """
const W_IH = %s, W_HH = %s, UNITS = %s, MODES = %s, MARKER = %s;
let mode = 'uncentred';
const CUSTOM_COLORS = ['#7f3c8d', '#11a579', '#3969ac', '#f2b701', '#e73f74', '#80ba5a', '#e68310', '#008695'];
let nCustom = 0;
function scores(xs) {
  const {V, mean} = MODES[mode];
  let h = new Array(16).fill(0);
  return xs.map(x => {
    h = W_IH.map((w, i) => Math.max(0, w * x + W_HH[i].reduce((s, a, j) => s + a * h[j], 0)));
    return V.map(v => v.reduce((s, a, k) => s + a * (h[UNITS[k]] - mean[k]), 0));
  });
}
// Input sequence, plotted steps and 3D line width of every trace, in trace order: impulse lines t = 0..T0 and T0..9,
// impulse points, the nine two-pulse inputs, then custom inputs as they are added. Switching projection rebuilds every
// trace from these (scatter3d for the uncentred PCs, 2D scatter for the centred ones), keeping its style and visibility.
const ALLT = [...Array(10).keys()], IMPULSE = ALLT.map(t => +(t === 0));
const TRACES = [[IMPULSE, ALLT.slice(0, %d), 3], [IMPULSE, ALLT.slice(%d), 8], [IMPULSE, ALLT, 0],
  ...%s.map(xs => [xs, ALLT, 6])];
const SCENE = JSON.parse(JSON.stringify(gd.layout.scene));
const coords = ([xs, ts]) => { const Q = scores(xs); return MODES[mode].V.map((_, k) => ts.map(t => Q[t][k])); };
function shape(tr, i) {   // trace i in the current projection
  const C = coords(TRACES[i]), is3d = mode === 'uncentred', d = {...tr, x: C[0], y: C[1], hovertemplate: MODES[mode].hover};
  delete d.z; if (is3d) d.z = C[2];
  d.type = is3d ? 'scatter3d' : 'scatter';
  if (TRACES[i][2]) d.line = {...tr.line, width: TRACES[i][2] / (is3d ? 1 : 2)};
  if (d.marker) d.marker = {...tr.marker, size: is3d ? 7 : 9};
  return d;
}
function axes() {   // equal-aspect axes covering every trace, shown or not
  const C = TRACES.map(coords), M = MODES[mode], pad = mode === 'uncentred' ? 8 : 3;
  return M.axes.map((title, k) => { const v = C.flatMap(c => c[k]);
    return {title: {text: title}, range: [Math.min(...v) - pad, Math.max(...v) + pad], color: '%s', gridcolor: '#e4e4e4',
            zerolinecolor: '%s', backgroundcolor: 'white'}; });
}
function render() {
  const L = JSON.parse(JSON.stringify(gd.layout)), A = axes();
  delete L.scene; delete L.xaxis; delete L.yaxis; L.title.text = MODES[mode].title;
  if (mode === 'uncentred') { L.scene = {...SCENE}; ['x', 'y', 'z'].forEach((a, k) => L.scene[a + 'axis'] = {...SCENE[a + 'axis'], ...A[k]}); }
  else { L.xaxis = {...A[0], zeroline: true}; L.yaxis = {...A[1], zeroline: true, scaleanchor: 'x', scaleratio: 1}; L.plot_bgcolor = 'white'; }
  L.margin = {...L.margin, l: mode === 'uncentred' ? 0 : 70, b: mode === 'uncentred' ? 0 : 50};
  Plotly.react(gd, gd.data.map(shape), L);
}
function setMode(m) { mode = m; render(); }
const form = document.createElement('form');
form.style.cssText = 'min-height:48px;box-sizing:border-box;padding:4px 8px;display:flex;flex-wrap:wrap;align-items:center;justify-content:center;gap:6px;border-top:1px solid #ddd;font:14px sans-serif';
form.innerHTML = '<select class="mode" style="margin-right:16px">' +
  Object.entries(MODES).map(([k, M]) => '<option value="' + k + '">' + M.label + '</option>').join('') + '</select>' +
  '<span>Custom x = (x_0, …, x_9):</span>' +
  Array.from({length: 10}, (_, t) => '<input class="xt" type="number" step="any" value="0" title="x_' + t +
    '" style="width:52px;margin-right:2px;padding:1px">').join('') + ' <button type="button" class="rand" title="iid Unif(-1, 1), the training distribution">Random</button> <button>Add</button>' +
  '<span class="err" style="color:#c00;font-size:12px"></span>';
document.body.appendChild(form);
// Bold every box that is a new running max: x_t > max(x_0, ..., x_(t-1)); x_0 always counts.
function boldMax() {
  let m = -Infinity;
  form.querySelectorAll('.xt').forEach(b => { const x = Number(b.value), rec = b.value.trim() !== '' && x > m;
    b.style.fontWeight = rec ? 'bold' : 'normal'; if (rec) m = x; });
}
form.addEventListener('input', boldMax);
form.querySelector('.mode').addEventListener('change', e => setMode(e.target.value));
form.querySelector('.rand').addEventListener('click', () => {
  form.querySelectorAll('.xt').forEach(b => b.value = (2 * Math.random() - 1).toFixed(2)); boldMax(); });
boldMax();
form.addEventListener('submit', e => {
  e.preventDefault();
  const err = form.querySelector('.err');
  const xs = [...form.querySelectorAll('.xt')].map(b => b.value.trim() === '' ? NaN : Number(b.value));
  if (xs.some(isNaN)) { err.textContent = 'Every box needs a number.'; return; }
  err.textContent = '';
  const name = 'x = (' + xs.join(', ') + ')', color = CUSTOM_COLORS[nCustom++ %% CUSTOM_COLORS.length];
  TRACES.push([xs, ALLT, 6]);
  Plotly.addTraces(gd, shape({mode: 'lines+markers+text', showlegend: false, line: {color}, marker: MARKER,
    text: ALLT.map(t => 't=' + t), textposition: 'top center', textfont: {size: 11, color: '%s'},
    customdata: ALLT.map(t => name + '<br>t=' + t)}, TRACES.length - 1));
  render();
  const label = document.createElement('label');
  label.style.cssText = 'display:inline-block;margin:2px 14px 2px 0;cursor:pointer';
  label.innerHTML = '<input type="checkbox" checked data-trace="' + (gd.data.length - 1) + '"> <span style="display:inline-block;width:18px;height:4px;border-radius:2px;background:' + color + ';vertical-align:middle"></span> ' + name;
  panel.querySelector('.custom').appendChild(label); fit();
});
fit();
""" % (*(json.dumps(v) for v in (W_ih.tolist(), W_hh.tolist(), units, MODES, dict(MARKER, color=list(range(10))))),
       T0 + 1, T0, json.dumps([x.tolist() for x in XS.values()]), INK, MUTED, INK)
panel += custom
fig.write_html(ROOT / "rotate" / "impulse_pca_3d.html", include_plotlyjs=True, full_html=True,
               config=dict(displaylogo=False, responsive=True), default_height="calc(100vh - 49px)", post_script=panel)

for p in EXTRA:
    first = np.argmax(np.abs(Q[p] - P).max(1) > 1e-9)   # first step that differs from the impulse; should equal p
    print(f"p={p}: {EXTRA[p][0]}, differs from t={first}, h_9 scores {np.round(Q[p][9], 2)}")
print("axis ranges:", np.round([ALL.min(0) - 8, ALL.max(0) + 8], 1).T.tolist())
