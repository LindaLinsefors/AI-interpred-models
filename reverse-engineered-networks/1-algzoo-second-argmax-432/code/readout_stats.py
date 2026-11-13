import numpy as np, json
w = np.load("m16_10_weights.npz"); Wi, Wh, Wo = w["W_ih"][:, 0], w["W_hh"], w["W_out"]
X = np.random.default_rng(3).standard_normal((20000, 10))
h = np.zeros((len(X), 16)); H = []
for t in range(10):
    h = np.maximum(X[:, t:t+1] * Wi + h @ Wh.T, 0); H.append(h)
order = np.argsort(X, 1); mx, ru = order[:, -1], order[:, -2]
comp = h[:, [1, 6, 7, 8]] @ Wo[:, [1, 6, 7, 8]].T
rest_idx = [0, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15]
rest = h[:, rest_idx] @ Wo[:, rest_idx].T
logit = comp + rest; ans = logit.argmax(1)
R = np.arange(len(X))
out = {}
for name, sel in [("max and runner-up both in 0-8", (mx < 9) & (ru < 9))]:
    c, r = comp[sel][:, :9], rest[sel][:, :9]; m, s = mx[sel], ru[sel]; n = sel.sum(); i = np.arange(n)
    out["n"] = int(n)
    out["rest_argmax_is_max"] = float(np.mean(r.argmax(1) == m))
    out["comp_argmin_is_max"] = float(np.mean(c.argmin(1) == m))
    out["comp_argmax_is_runnerup"] = float(np.mean(c.argmax(1) == s))
    out["rest_argmax_is_runnerup"] = float(np.mean(r.argmax(1) == s))
    out["median_comp_at_max"] = float(np.median(c[i, m])); out["median_rest_at_max"] = float(np.median(r[i, m]))
    out["median_comp_at_runnerup"] = float(np.median(c[i, s])); out["median_rest_at_runnerup"] = float(np.median(r[i, s]))
    # how often is runner-up preferred over max by comp alone, by rest alone
    out["comp_prefers_runnerup_over_max"] = float(np.mean(c[i, s] > c[i, m]))
    out["rest_prefers_runnerup_over_max"] = float(np.mean(r[i, s] > r[i, m]))
    out["acc"] = float(np.mean(ans[sel] == s))
# newest input: position 9
sel9 = ru == 9
out["runnerup_is_last_frac"] = float(np.mean(sel9)); out["acc_when_runnerup_last"] = float(np.mean(ans[sel9] == 9))
selmax9 = mx == 9
out["acc_when_max_last"] = float(np.mean(ans[selmax9] == ru[selmax9]))
print(json.dumps(out, indent=1)); json.dump(out, open("readout_stats.json", "w"), indent=1)
