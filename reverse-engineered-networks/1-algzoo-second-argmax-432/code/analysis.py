import importlib.util, json
from types import SimpleNamespace
import numpy as np
w = np.load("m16_10_weights.npz"); W_ih, W_hh, W_out = w["W_ih"][:, 0], w["W_hh"], w["W_out"]
spec = importlib.util.spec_from_file_location("ex", "exact_rewrite.py"); ex = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)

def net_trace(x):
    h = np.zeros(16); H = []
    for t in range(10):
        h = np.maximum(x[t] * W_ih + W_hh @ h, 0); H.append(h)
    return np.array(H)

def rewrite_trace(x):
    carry = None; mem = []
    for t in range(10):
        carry, _ = ex.step(carry, SimpleNamespace(values=x, position=t), {})
        mem.append(carry[0].copy())
    return np.array(mem)

out = {}
# ---- example row
x = np.array([0.3, 1.4, -0.5, 0.2, 0.9, -1.0, 1.8, 0.1, -0.3, 0.4])
H = net_trace(x); mem = rewrite_trace(x)
logits = W_out @ H[-1]
groups = {"comparators (1,6,7,8)": [1, 6, 7, 8], "record pulse (4)": [4], "other 11 units": [0, 2, 3, 5, 9, 10, 11, 12, 13, 14, 15]}
contrib = {g: (W_out[:, u] @ H[-1, u]).tolist() for g, u in groups.items()}
M = S = 0.0; Ms = []; Ss = []; age = []; a = None
for t in range(10):
    if x[t] > M: S, M, a = M, x[t], 0
    elif x[t] > S: S = x[t]; a = a + 1 if a is not None else None
    else: a = a + 1 if a is not None else None
    Ms.append(M); Ss.append(S); age.append(a)
out["example"] = dict(x=x.tolist(), hidden=H.round(3).tolist(), logits=logits.round(2).tolist(),
    answer=int(logits.argmax()), true=int(np.argsort(x)[-2]), contrib={k: np.round(v, 2).tolist() for k, v in contrib.items()},
    level=mem[:, 12].round(3).tolist(), runner=mem[:, 11].round(3).tolist(),
    clock_deg=np.degrees(np.arctan2(mem[:, 9], mem[:, 8])).round(1).tolist(), clock_r=np.hypot(mem[:, 8], mem[:, 9]).round(3).tolist(),
    M=Ms, S=Ss, age_of_max=age)

# ---- bulk stats
rng = np.random.default_rng(2)
X = rng.standard_normal((20000, 10))
order = np.argsort(X, 1); srt = np.sort(X, 1)
label = order[:, -2]
h = np.zeros((len(X), 16))
for t in range(10): h = np.maximum(X[:, t:t+1] * W_ih + h @ W_hh.T, 0)
ans = (h @ W_out.T).argmax(1)
wrong = ans != label
out["accuracy"] = float(np.mean(~wrong))
out["errors_name_third"] = float(np.mean(ans[wrong] == order[wrong, -3]))
out["errors_name_max"] = float(np.mean(ans[wrong] == order[wrong, -1]))
gap = np.minimum(srt[:, -1] - srt[:, -2], srt[:, -2] - srt[:, -3])
out["errors_with_min_gap_lt_0.05"] = float(np.mean(gap[wrong] < 0.05))
out["rows_with_min_gap_lt_0.05"] = float(np.mean(gap < 0.05))
bins = [0, 0.01, 0.025, 0.05, 0.1, 0.2, 0.4, 10]
out["acc_by_gap"] = [dict(lo=bins[i], hi=bins[i+1], n=int(((gap >= bins[i]) & (gap < bins[i+1])).sum()),
    acc=float(np.mean(~wrong[(gap >= bins[i]) & (gap < bins[i+1])]))) for i in range(len(bins)-1)]
# clock phase vs age of max, all steps, 4000 rows
ph = {k: [] for k in range(10)}; rad = {k: [] for k in range(10)}
for x in X[:4000]:
    mem = rewrite_trace(x); M = -np.inf; a = 0
    for t in range(10):
        if x[t] > M: M, a = x[t], 0
        else: a += 1
        if M > 0:
            ph[a].append(np.degrees(np.arctan2(mem[t, 9], mem[t, 8])) % 360); rad[a].append(np.hypot(mem[t, 8], mem[t, 9]))
out["clock_by_age"] = [dict(age=k, n=len(ph[k]), median_deg=float(np.median(ph[k])), q25=float(np.percentile(ph[k], 25)), q75=float(np.percentile(ph[k], 75)), median_r=float(np.median(rad[k]))) for k in range(10) if ph[k]]
json.dump(out, open("analysis.json", "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "example"}, indent=1))
e = out["example"]; print({k: e[k] for k in ["logits", "answer", "true", "contrib", "level", "runner", "clock_deg", "clock_r", "age_of_max"]})
print(np.array(e["hidden"])[:, [1, 4, 6, 7, 8]])
