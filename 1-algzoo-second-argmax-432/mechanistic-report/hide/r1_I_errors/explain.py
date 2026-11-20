"""Step-by-step view of one input: thresholds of unit 4 and of the comparators on x_t, which of them the input
crossed, the pulse activations, and the final logits.

  theta_j(t) = -W[j].h_{t-1}/w_j.  Unit 4 is ON iff x_t > theta_4;  comparator j (1, 6, 7, 8) is OFF iff x_t > theta_j.
"""
from icommon import *
from i03_interventions import run_iv


def explain(x, title="", show_logits=True):
    x = np.asarray(x, dtype=np.float64)
    H, Z = run(x, return_pre=True); H, Z = H[0], Z[0]
    lg = U @ H[-1]
    order = np.argsort(-x, kind="stable")
    print(f"\n### {title}")
    print("x = (" + ", ".join(f"{v:g}" for v in x) + f")   max at {order[0]}, runner-up at {order[1]}, third at {order[2]}")
    rows = []
    hprev = np.zeros(N)
    for t in range(T):
        th = {j: -(W[j] @ hprev) / w[j] for j in [4] + COMP}
        off = [j for j in COMP if x[t] > th[j]]
        rows.append([t, f"{x[t]:g}", f"{th[4]:.3f}"] + [f"{th[j]:.3f}" for j in COMP] +
                    [",".join(map(str, off)) if off else "-", f"{H[t, 4]:.2f}", f"{H[t, 5]:.2f}", f"{H[t, 9]:.2f}"])
        hprev = H[t]
    print(fmt_table(rows, ["t", "x_t", "theta_4", "theta_1", "theta_6", "theta_7", "theta_8", "comparators switched off (x_t > theta_j)",
                           "h_t[4]", "h_t[5]", "h_t[9]"]))
    if show_logits:
        print("logits: " + "  ".join(f"{p}:{v:.2f}" for p, v in enumerate(lg)))
    ans = int(lg.argmax())
    print(f"answer {ans} (logit {lg[ans]:.2f}); correct {order[1]} (logit {lg[order[1]]:.2f}); margin {lg[order[1]] - np.max(np.delete(lg, order[1])):.2f}  ->  {'CORRECT' if ans == order[1] else 'WRONG'}")
    return ans, lg


def answer(x):
    return int(logits(run(np.asarray(x, dtype=np.float64))).argmax(1)[0])


def answer_iv(x, mode):
    lg, _ = run_iv(np.atleast_2d(np.asarray(x, dtype=np.float64)), mode)
    return int(lg.argmax(1)[0])


def seq(vals, b=-1.0):
    """vals: dict position -> value; all other positions get the background b."""
    x = np.full(T, float(b))
    for p, v in vals.items():
        x[p] = v
    return x
