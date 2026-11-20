"""I4. Near ties. Exact coefficients (gates of the input held fixed) of each input in a logit margin.

  S-T ties:  m_ST = logit[pS] - logit[pT] = c_S*S + c_T*T + c_M*M + sum_rest c_p x_p
                  = k*(S - T) + (c_S + c_T)*(S + T)/2 + c_M*M + rest,          k = (c_S - c_T)/2
  M-S ties:  m_SM = logit[pS] - logit[pM] = c_M*M + c_S*S + c_T*T + rest
                  = k*(M - S) + (c_M + c_S)*(M + S)/2 + c_T*T + rest,          k = (c_M - c_S)/2
  'offset' b = margin - k*gap  = the margin the net would have at zero gap if the gates did not change.
"""
from icommon import *

q = lambda v: " / ".join(f"{a:.2f}" for a in np.quantile(v, [.1, .5, .9]))


def collect(kind, tie=0.05):
    out = {k: [] for k in ("pM", "pS", "pT", "M", "S", "T", "ans", "cA", "cB", "cC", "rest", "margin", "npos")}
    ntot = 0
    for s, x in batches():
        pM, pS, pT, M, S, Tv = top3(x)
        sel = (S - Tv < tie) if kind == "ST" else (M - S < tie)
        ntot += len(x)
        x = x[sel]; pM, pS, pT, M, S, Tv = (a[sel] for a in (pM, pS, pT, M, S, Tv))
        H, G = run_gates(x)
        lg = logits(H); r = np.arange(len(x))
        if kind == "ST":
            v = U[pS] - U[pT]; pa, pb, pc = pS, pT, pM          # A = S, B = T, C = M
        else:
            v = U[pS] - U[pM]; pa, pb, pc = pM, pS, pT          # A = M, B = S, C = T
        c = coefs(v, G)
        margin = lg[r, pa if kind == "ST" else pS] - lg[r, pb if kind == "ST" else pM]
        tot = (c * x).sum(1)
        assert np.abs(tot - margin).max() < 1e-8
        cA, cB, cC = c[r, pa], c[r, pb], c[r, pc]
        rest = tot - cA * x[r, pa] - cB * x[r, pb] - cC * x[r, pc]
        for k, val in (("pM", pM), ("pS", pS), ("pT", pT), ("M", M), ("S", S), ("T", Tv), ("ans", lg.argmax(1)),
                       ("cA", cA), ("cB", cB), ("cC", cC), ("rest", rest), ("margin", margin), ("npos", (x > 0).sum(1))):
            out[k].append(val)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["ntot"] = ntot
    return out


def report(kind):
    D = collect(kind)
    n = len(D["ans"])
    if kind == "ST":
        gap = D["S"] - D["T"]; hi, lo, third = D["S"], D["T"], D["M"]
        first = np.where(D["pS"] < D["pT"], "S first", "T first")
        nameA, nameB, nameC = "S", "T", "M"
        ageA, ageB = 9 - D["pS"], 9 - D["pT"]
    else:
        gap = D["M"] - D["S"]; hi, lo, third = D["M"], D["S"], D["T"]
        first = np.where(D["pM"] < D["pS"], "M first", "S first")
        nameA, nameB, nameC = "M", "S", "T"
        ageA, ageB = 9 - D["pM"], 9 - D["pS"]
    k = (D["cA"] - D["cB"]) / 2
    asym = (D["cA"] + D["cB"])
    t_asym = asym * (hi + lo) / 2
    t_C = D["cC"] * third
    b = D["margin"] - k * gap
    assert np.allclose(b, t_asym + t_C + D["rest"])
    print(f"\n# {kind} near ties: gap < 0.05, n = {n} of {D['ntot']} ({n/D['ntot']:.4f})")
    a = D["ans"]
    print(f"answers: pS {np.mean(a == D['pS']):.4f}, pT {np.mean(a == D['pT']):.4f}, pM {np.mean(a == D['pM']):.4f}, other {np.mean((a != D['pS']) & (a != D['pT']) & (a != D['pM'])):.4f}")
    clean = (D["T"] > 0) & ((D["M"] - D["S"] >= 0.1) if kind == "ST" else (D["S"] - D["T"] >= 0.1))
    print(f"clean subset (T > 0 and the other gap >= 0.1): n = {clean.sum()}; answers: pS {np.mean(a[clean] == D['pS'][clean]):.4f}, "
          f"pT {np.mean(a[clean] == D['pT'][clean]):.4f}, pM {np.mean(a[clean] == D['pM'][clean]):.4f}")

    print(f"\n## coefficients against gap and order of arrival (clean subset). Quantiles 10% / 50% / 90%.")
    print(f"   c_{nameA}, c_{nameB}: coefficient of that input in the margin; k = (c_{nameA} - c_{nameB})/2; asym = c_{nameA} + c_{nameB}")
    edges = [0, 0.0125, 0.025, 0.05]
    rows = []
    for od in np.unique(first):
        for i in range(3):
            sel = clean & (first == od) & (gap >= edges[i]) & (gap < edges[i + 1])
            rows.append([od, f"[{edges[i]}, {edges[i+1]})", sel.sum(), q(D["cA"][sel]), q(D["cB"][sel]), q(k[sel]), q(asym[sel]),
                         f"{np.mean(np.abs(asym[sel]) < 0.1 * np.abs(k[sel])):.3f}", f"{np.mean(k[sel] > 0):.4f}",
                         f"{np.mean(D['margin'][sel] > 0):.3f}"])
    print(fmt_table(rows, ["order", "gap", "n", f"c_{nameA}", f"c_{nameB}", "k", "asym", "P(abs(asym) < 0.1 abs(k))", "P(k > 0)", "P(margin > 0)"]))

    print(f"\n## size of the terms of the margin (clean subset): k*gap, offset b and its parts. 10% / 50% / 90% and mean absolute value")
    rows = []
    for name, v in ((f"k * gap", k * gap), ("offset b = margin - k*gap", b), (f"  asym * mean({nameA},{nameB})", t_asym), (f"  c_{nameC} * {nameC}", t_C),
                    ("  rest (7 other inputs)", D["rest"]), ("margin", D["margin"])):
        rows.append([name, q(v[clean]), f"{np.abs(v[clean]).mean():.3f}", f"{v[clean].mean():.3f}", f"{v[clean].std():.3f}"])
    print(fmt_table(rows, ["term", "10% / 50% / 90%", "mean abs", "mean", "sd"]))
    sel = clean
    print(f"sign(margin) == sign(b): {np.mean(np.sign(D['margin'][sel]) == np.sign(b[sel])):.4f};  "
          f"P(b > 0) = {np.mean(b[sel] > 0):.4f};  P(margin > 0) = {np.mean(D['margin'][sel] > 0):.4f}")
    print(f"corr(asym term, c_{nameC} term) = {np.corrcoef(t_asym[sel], t_C[sel])[0,1]:.3f}; corr(asym term, rest) = {np.corrcoef(t_asym[sel], D['rest'][sel])[0,1]:.3f}")
    # scale-free dead zone: b / (k * M)
    dz = -b[sel] / (k[sel] * D["M"][sel])
    print(f"indifference gap g* = -b/k (gap at which the local linear margin is 0), in units of M: quantiles 10/25/50/75/90% = "
          + " / ".join(f"{v:.4f}" for v in np.quantile(dz, [.1, .25, .5, .75, .9])))

    print(f"\n## slope k (logits per unit gap) by ages of {nameA} (rows) and {nameB} (columns), median, clean subset; '.' if n < 100")
    rows = []
    for i in range(10):
        row = [i]
        for j in range(10):
            s2 = sel & (ageA == i) & (ageB == j)
            row.append("." if s2.sum() < 100 else f"{np.median(k[s2]):.1f}")
        rows.append(row)
    print(fmt_table(rows, [f"age {nameA} \\ age {nameB}"] + [str(j) for j in range(10)]))
    print(f"\n## P(margin > 0) (net prefers the true runner-up over {'the third' if kind == 'ST' else 'the max'}) by ages, clean subset")
    rows = []
    for i in range(10):
        row = [i]
        for j in range(10):
            s2 = sel & (ageA == i) & (ageB == j)
            row.append("." if s2.sum() < 100 else f"{np.mean(D['margin'][s2] > 0):.2f}")
        rows.append(row)
    print(fmt_table(rows, [f"age {nameA} \\ age {nameB}"] + [str(j) for j in range(10)]))
    print(f"\n## median offset b by ages, clean subset")
    rows = []
    for i in range(10):
        row = [i]
        for j in range(10):
            s2 = sel & (ageA == i) & (ageB == j)
            row.append("." if s2.sum() < 100 else f"{np.median(b[s2]):.2f}")
        rows.append(row)
    print(fmt_table(rows, [f"age {nameA} \\ age {nameB}"] + [str(j) for j in range(10)]))
    return D


if __name__ == "__main__":
    report("ST")
    report("SM")
