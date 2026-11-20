"""I3. Mechanistic attribution by intervention on the pulse units 4 and 5.

Definitions (prefix statistics exclude the current input):
  Mprev_t = max(x_0..x_{t-1}) (-inf at t = 0),  Sprev_t = second largest of x_0..x_{t-1} (-inf for t <= 1)
  oracle_t = 10.157 * max(x_t - max(Mprev_t, 0), 0)
  (a)  h_t[4] := oracle_t at every step
  (b1) h_t[4] := 0 whenever x_t is not a new max (x_t <= Mprev_t)                       ['misfires' removed]
  (b1') h_t[4] := 0 whenever oracle_t = 0 (x_t <= max(Mprev_t, 0))                      [same, zero reference]
  (b2) h_t[4] := oracle_t whenever x_t > max(Mprev_t,0) but unit 4 is silent in the running net  ['misses' repaired]
  (c)  h_{t+1}[5] := 0 whenever x_t < Sprev_t  (x_t is below the runner-up so far; t >= 2)
  (c') h_{t+1}[5] := 0 whenever x_t < max(Sprev_t, 0)                                     [zero reference]
  (d)  (b1) + (b2) + (c)
All interventions act on the running (already intervened) network; later steps see the modified state.
"""
from icommon import *


def run_iv(x, mode):
    """mode: set of strings among a, b1, b1z, b2, c, cz. Returns logits and counts of interventions that changed something."""
    B = len(x)
    h = np.zeros((B, N))
    Mprev = np.full(B, -np.inf); Sprev = np.full(B, -np.inf)
    kill5 = np.zeros(B, bool)                      # decided at step t, applied at t + 1
    changed = np.zeros(B, bool)
    for t in range(T):
        xt = x[:, t]
        z = xt[:, None] * w + h @ W.T
        h = np.maximum(z, 0.0)
        ref = np.maximum(Mprev, 0.0)
        oracle = W4 * np.maximum(xt - ref, 0.0)
        new4 = h[:, 4].copy()
        if "a" in mode:
            new4 = oracle.copy()
        if "b1" in mode:
            new4 = np.where(xt <= Mprev, 0.0, new4)
        if "b1z" in mode:
            new4 = np.where(xt <= ref, 0.0, new4)
        if "b2" in mode:
            new4 = np.where((xt > ref) & (h[:, 4] <= 0), oracle, new4)
        changed |= np.abs(new4 - h[:, 4]) > 1e-12
        h[:, 4] = new4
        if ("c" in mode or "cz" in mode):
            changed |= kill5 & (h[:, 5] > 0)
            h[:, 5] = np.where(kill5, 0.0, h[:, 5])
            thr = Sprev if "c" in mode else np.maximum(Sprev, 0.0)
            kill5 = xt < thr
        # update prefix statistics
        newM = np.maximum(Mprev, xt)
        Sprev = np.where(xt > Mprev, Mprev, np.maximum(Sprev, xt))
        Mprev = newM
    return h @ U.T, changed


MODES = [("(a) unit 4 := oracle", {"a"}),
         ("(b1) unit 4 off unless new max", {"b1"}),
         ("(b1') unit 4 off unless x_t > max(M,0)", {"b1z"}),
         ("(b2) unit 4 := oracle at misses", {"b2"}),
         ("(b) = b1 + b2", {"b1", "b2"}),
         ("(c) unit 5 off after x_t < S_prev", {"c"}),
         ("(c') unit 5 off after x_t < max(S_prev,0)", {"cz"}),
         ("(d) = b1 + b2 + c", {"b1", "b2", "c"}),
         ("(a) + (c)", {"a", "c"})]

if __name__ == "__main__":
    tot = {name: np.zeros(8) for name, _ in MODES}
    n = 0; nerr = 0
    for s, x in batches():
        tgt = second_argmax(x)
        base = logits(run(x)).argmax(1)
        ok0 = base == tgt
        n += len(x); nerr += (~ok0).sum()
        lg0, ch0 = run_iv(x, set())
        assert np.array_equal(lg0.argmax(1), base) and not ch0.any()      # the harness reproduces the net
        for name, mode in MODES:
            lg, ch = run_iv(x, mode)
            a = lg.argmax(1); ok1 = a == tgt
            tot[name] += [(a == base).sum(), ok1.sum(), (~ok0 & ok1).sum(), (ok0 & ~ok1).sum(), ch.sum(),
                          (ch & ~ok0).sum(), (~ok0 & ~ok1 & (a != base)).sum(), (ok1 & ch).sum()]
    print(f"n = {n}, unmodified accuracy = {1 - nerr/n:.5f}, errors = {nerr}")
    rows = []
    for name, _ in MODES:
        v = tot[name]
        rows.append([name, f"{v[4]/n:.4f}", f"{v[0]/n:.4f}", f"{v[1]/n:.4f}", f"{v[2]/nerr:.4f}", f"{v[3]/nerr:.4f}",
                     f"{(v[2]-v[3])/nerr:+.4f}", f"{v[5]/nerr:.4f}", f"{v[6]/nerr:.4f}"])
    print(fmt_table(rows, ["intervention", "sequences touched", "agreement with net", "task accuracy",
                           "net's errors fixed (share of errors)", "new errors (as share of net's errors)",
                           "net change", "share of net's errors that are touched", "errors changed to another wrong answer"]))
