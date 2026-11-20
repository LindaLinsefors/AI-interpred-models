"""Verifier V1, attack E2b: how much of a REAL no-event step is the resting map A that T1/T2 analyse?

Real no-event steps (x_t < S_{t-1}, S_{t-1} > 0.1).  Old positions 0..t-1 (indices 9-t..8 at time t), per-state
mean over those positions removed, R^2 with slope 1 of the actual s_t against
  P0  s_{t-1} shifted by one position
  P1  U_o A h_{t-1}[ORD]                                  (resting map: what T1/T2 are about)
  P2  P1 + U_o W[ORD,PULSE] h_{t-1}[PULSE]                 (+ what the pulse units send)
  P3  P2 + U_o w[ORD] x_t  = U_o z_t[ORD]                  (+ the input; still no ReLU)
  P4  U_o h_t[ORD]                                         (+ ReLU gating of the 13 ordinary units)
  (P5 = P4 + U[:,PULSE] h_t[PULSE] is the actual s_t)
Also: how often is the gate pattern of a real state the resting pattern (13 ordinary units on, pulses off)?
Seeds 5601..5603, 200k each.  Run: python -B v1_attack_resting_vs_real.py -> v1_attack_resting_vs_real.txt
"""
from vlib import *

P = Tee("v1_attack_resting_vs_real.txt")
Uo = U[:, ORD]; A = W[np.ix_(ORD, ORD)]
SEEDS = [5601, 5602, 5603]
NB = 200_000
preds = ["P0 shift", "P1 resting", "P2 +pulses", "P3 +input", "P4 +ReLU(ORD)"]
subsets = ["all old positions", "positions of age>=2 at t", "all old positions, max age>=3 at t", "age>=2 positions, max age>=3 at t"]
SS = {s: {p: np.zeros((10, 2)) for p in preds} for s in subsets}      # [t, (ss_res, ss_tot)]
gate_all13 = np.zeros((10, 3)); n_states = np.zeros(10)
n_off_hist = np.zeros((10, 14))
unit_on = np.zeros((10, 16))


def add(key, sub, y, p, t):
    y = y - y.mean(1, keepdims=True); p = p - p.mean(1, keepdims=True)
    SS[sub][key][t, 0] += ((y - p) ** 2).sum(); SS[sub][key][t, 1] += (y ** 2).sum()


for seed in SEEDS:
    x = sample(NB, seed)
    H, Z = run(x, return_pre=True)
    for t in range(1, 10):
        G = Z[:, t] > 0
        n_states[t] += NB
        ord_on = G[:, ORD].all(1); pul_off = ~G[:, PULSE].any(1)
        gate_all13[t] += [ord_on.sum(), pul_off.sum(), (ord_on & pul_off).sum()]
        np.add.at(n_off_hist[t], (~G[:, ORD]).sum(1), 1)
        unit_on[t] += G.sum(0)
    for t in range(2, 10):
        p, val = top3(x, t - 1)
        p1, S = p[:, 0], val[:, 1]
        m = (x[:, t] < S) & (S > 0.1)
        h0, h1, z1 = H[m, t - 1], H[m, t], Z[m, t]
        sl = slice(9 - t, 9)
        y = (h1 @ U.T)[:, sl]
        s_prev = (h0 @ U.T)[:, 10 - t: 10]
        P1 = (h0[:, ORD] @ (Uo @ A).T)[:, sl]
        P2 = P1 + (h0[:, PULSE] @ (Uo @ W[np.ix_(ORD, PULSE)]).T)[:, sl]
        P3 = (z1[:, ORD] @ Uo.T)[:, sl]
        P4 = (h1[:, ORD] @ Uo.T)[:, sl]
        a1_after = t - p1[m]
        for sub in subsets:
            mm = np.ones(m.sum(), bool) if "max age" not in sub else (a1_after >= 3)
            cols = slice(0, t) if sub.startswith("all") else slice(0, t - 1)
            if cols.stop - cols.start < 2 or mm.sum() < 10:
                continue
            for key, pr in zip(preds, (s_prev, P1, P2, P3, P4)):
                add(key, sub, y[mm][:, cols], pr[mm][:, cols], t)
    del H, Z

P("=== gate pattern of real states (all states, N(0,1)) ===")
P(" t | P(all 13 ordinary units on) | P(all 3 pulse units off) | P(both = resting pattern) | mean number of ordinary units OFF")
for t in range(1, 10):
    P(f" {t} | {gate_all13[t,0]/n_states[t]:.4f} | {gate_all13[t,1]/n_states[t]:.4f} | {gate_all13[t,2]/n_states[t]:.4f} | {(n_off_hist[t]*np.arange(14)).sum()/n_states[t]:.2f}")
P(" P(unit on) pooled over t=3..9, units 0..15:", np.round(unit_on[3:].sum(0) / n_states[3:].sum(), 3))
P(" distribution of the number of ordinary units OFF, pooled t=3..9 (0..13):", np.round(n_off_hist[3:].sum(0) / n_states[3:].sum(), 3))

P("\n=== centred R^2 (slope 1) of the actual s_t on real no-event steps (S_{t-1} > 0.1) ===")
for sub in subsets:
    P(f"--- {sub}")
    P("   t   " + "  ".join(f"{k:>14s}" for k in preds))
    for t in range(2, 10):
        if SS[sub][preds[0]][t, 1] == 0:
            continue
        P(f"   {t}   " + "  ".join(f"{1 - SS[sub][k][t,0]/SS[sub][k][t,1]:14.3f}" for k in preds))
