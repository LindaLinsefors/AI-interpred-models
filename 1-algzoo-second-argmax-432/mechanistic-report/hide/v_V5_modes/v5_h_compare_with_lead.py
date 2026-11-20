"""Verifier V5, part H (run only AFTER my own numbers for M1-M6 existed): compare my per-cell table (v5_b_m5m6.txt)
with the lead's saved table e05_cell_traces.pkl (read-only), and summarise my table.

Run:  python -B v5_h_compare_with_lead.py > v5_h_compare_with_lead.txt
"""
import pickle
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
res = pickle.load(open(HERE.parent / "e05_cell_traces.pkl", "rb"))
txt = (HERE / "v5_b_m5m6.txt").read_text(encoding="utf-8")
m5 = txt.split("M5  clock")[1].split("summary per a1")[0]
m6 = txt.split("M6  level")[1].split("cells with a1 >= 0")[0]
pat5 = re.compile(r"a1=(\d) a2=(\d)\s+alpha=([\d.]+)@\s*([+-][\d.]+)\s+beta=([\d.]+)@\s*([+-][\d.]+)\s+R2u=([\d.]+)\s+R2c=([\d.]+)")
pat6 = re.compile(r"a1=(\d) a2=(\d)\s+alpha=(-?[\d.]+)\s+beta=(-?[\d.]+)\s+R2u=([\d.]+)")
rows = [(int(a), int(b), float(am), float(ap), float(bm), float(bp), float(ru), float(rc)) for a, b, am, ap, bm, bp, ru, rc in pat5.findall(m5)]
r = [x for x in rows if x[0] >= 1]

d_mod, d_ph, d_b, d_r = [], [], [], []
for A1, A2, am, ap, bm, bp, ru, rc in r:
    a, b, R2, n = res[(A1, A2, "clock")]
    mine = am * np.exp(1j * np.radians(ap))
    d_mod.append(abs(a) - am); d_ph.append(np.degrees(np.angle(a / mine))); d_b.append(abs(b) - bm); d_r.append(R2 - ru)
print("M5, cells with a1 >= 1 (81 cells): lead (seed 13, 1M sequences) minus mine (seeds 5010-5013, 800k sequences)")
print(f"   |alpha|: max abs diff {np.abs(d_mod).max():.4f}, mean diff {np.mean(d_mod):+.4f};  phase: max abs diff {np.abs(d_ph).max():.2f} deg;"
      f"  |beta|: max abs diff {np.abs(d_b).max():.4f};  R2u: max abs diff {np.abs(d_r).max():.4f}")
print("   lead's own table, |alpha| per a1: mean / median / min / max over a2 cells, and the a2 = a1 + 1 cell")
for A1 in range(1, 10):
    al = np.array([abs(v[0]) for (p, q, n), v in res.items() if n == "clock" and p == A1])
    nxt = res.get((A1, A1 + 1, "clock"))
    print(f"      a1={A1}: mean {al.mean():.3f}  median {np.median(al):.3f}  min {al.min():.3f}  max {al.max():.3f}   a2=a1+1: {abs(nxt[0]) if nxt else float('nan'):.3f}")
lb = np.array([abs(v[1]) for (p, q, n), v in res.items() if n == "clock" and p >= 1])
lr = np.array([v[2] for (p, q, n), v in res.items() if n == "clock" and p >= 1])
print(f"   lead's table: |beta| in [{lb.min():.3f}, {lb.max():.3f}], min R2u {lr.min():.4f}")
d_a, d_b, d_r = [], [], []
for A1, A2, a_, b_, r2 in pat6.findall(m6):
    A1, A2 = int(A1), int(A2)
    if A1 == 0:
        continue
    a, b, R2, n = res[(A1, A2, "level")]
    d_a.append(a.real - float(a_)); d_b.append(b.real - float(b_)); d_r.append(R2 - float(r2))
print(f"M6, cells with a1 >= 1: alpha max abs diff {np.abs(d_a).max():.4f}; beta max abs diff {np.abs(d_b).max():.4f}; R2u max abs diff {np.abs(d_r).max():.5f}")

print("\nSUMMARY OF MY M5 TABLE (81 cells with a1 >= 1)")
rc = np.array([x[7] for x in r]); ru = np.array([x[6] for x in r])
print(f"   centred R2: min {rc.min():.3f}, median {np.median(rc):.3f}, cells below 0.90: {int((rc < 0.90).sum())}, below 0.85: {int((rc < 0.85).sum())};"
      f"   uncentred R2: min {ru.min():.4f}, median {np.median(ru):.4f}, cells below 0.96: {int((ru < 0.96).sum())}")
for A1 in range(1, 10):
    c = [x for x in r if x[0] == A1]
    older = [x[2] for x in c if x[1] > A1]; younger = [x[2] for x in c if x[1] < A1]
    al = np.array([x[2] for x in c]); ph = np.array([x[3] for x in c]) % 360
    ph = np.where(ph > 300, ph - 360, ph) if A1 == 1 else ph
    print(f"   a1={A1}: mean |alpha| {al.mean():.3f}, median {np.median(al):.3f}, half-range {(al.max() - al.min()) / 2:.3f}; phase mean {ph.mean():.1f}, half-range {(ph.max() - ph.min()) / 2:.1f};"
          f" cells a2 > a1: mean |alpha| {np.mean(older) if older else float('nan'):.3f}; cells a2 < a1: {np.mean(younger) if younger else float('nan'):.3f}")
means = np.array([np.mean([x[2] for x in r if x[0] == A1]) for A1 in range(1, 10)])
older = np.array([np.mean([x[2] for x in r if x[0] == A1 and x[1] > A1]) for A1 in range(1, 9)])
print(f"   decay per step of the mean |alpha|, a1 = 3 -> 9: {(means[8] / means[2]) ** (1 / 6):.4f};  cells a2 > a1 only, a1 = 3 -> 8: {(older[7] / older[2]) ** (1 / 5):.4f};  eigenvalue modulus 0.9509")
print("   |beta| in cells a2 < a1 (runner-up arrived after the max), by age a2 of the runner-up:")
for A2 in range(0, 9):
    b = [x[4] for x in r if x[1] == A2 and x[1] < x[0]]
    print(f"      a2={A2}: mean |beta| = {np.mean(b):.3f}  range {min(b):.3f}..{max(b):.3f}  cells {len(b)}")
