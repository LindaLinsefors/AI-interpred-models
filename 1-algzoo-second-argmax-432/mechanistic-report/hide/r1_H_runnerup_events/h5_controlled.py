"""H5. Controlled inputs: max M at p1, runner-up S at p2 > p1, constant background b.
The 'S record' is the difference between the run with S and the run with x[p2] = b (no runner-up)."""
from hcommon import *

Mv = 1.5
SR = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.97]          # S / M
BG = [-1.0, -0.3]
R = {n: Uo @ RIGHT[n] for n in NAMES}                      # readout of each mode


def seq(p1, p2, S, b, M=Mv, extra=None):
    x = np.full(10, b); x[p1] = M; x[p2] = S
    if extra: x[extra[0]] = extra[1]
    return x


def mode_logits(h):
    """logit contribution of every mode and of the pulse units for a final state h (16,)."""
    m = coords(h)
    out = {n: (2 * np.real(m[n] * R[n]) if IS_COMPLEX[n] else np.real(m[n]) * R[n].real) for n in NAMES}
    out["pulses"] = U[:, PULSE] @ h[PULSE]
    return out


if __name__ == "__main__":
    # ---------- 1. answers and margins ----------
    print("=== 1. final answer and margin of p2 over the best other position ===")
    rows = []
    for b in BG:
        for sr in SR:
            for p1 in range(9):
                for p2 in range(p1 + 1, 10):
                    lg = logits(run(seq(p1, p2, sr * Mv, b)))[0]
                    oth = np.delete(np.arange(10), p2)
                    best = oth[np.argmax(lg[oth])]
                    rows.append((b, sr, p1, p2, lg[p2] - lg[best], best, lg[p2] - lg[p1], lg.argmax() == p2))
    rows = np.array(rows, dtype=float)
    for b in BG:
        print(f"\n background b = {b}, M = {Mv}:   correct answers / 45 pairs, and margin / M: min, median over pairs")
        for sr in SR:
            s = (rows[:, 0] == b) & (rows[:, 1] == sr)
            print(f"   S/M={sr:4.2f}: correct {int(rows[s, 7].sum()):2d}/45   margin/M min {rows[s, 4].min() / Mv:7.2f}  median {np.median(rows[s, 4]) / Mv:7.2f}   "
                  f"margin over the max's position /M: min {rows[s, 6].min() / Mv:7.2f} median {np.median(rows[s, 6]) / Mv:7.2f}")
        for sr in (0.2, 0.5, 0.8, 0.97):
            print(f"   margin/M of p2 over the best other, S/M = {sr} (rows p1 = 0..8, columns p2 = 1..9; * = wrong answer; in brackets: which position is the best other)")
            for p1 in range(9):
                line = ""
                for p2 in range(1, 10):
                    if p2 <= p1: line += "            "; continue
                    s = (rows[:, 0] == b) & (rows[:, 1] == sr) & (rows[:, 2] == p1) & (rows[:, 3] == p2)
                    line += f" {rows[s, 4][0] / Mv:6.1f}{'*' if rows[s, 7][0] == 0 else ' '}[{int(rows[s, 5][0])}] "
                print(f"     p1={p1}: {line}")
    print("\n margin/M of p2 over the best other by a2 = 9 - p2 (median over p1 < p2) for each S/M:")
    for b in BG:
        print(f"  b = {b}:   a2 = " + " ".join(f"{a:6d}" for a in range(9)))
        for sr in SR:
            line = ""
            for a2 in range(9):
                s = (rows[:, 0] == b) & (rows[:, 1] == sr) & (rows[:, 3] == 9 - a2)
                line += f" {np.median(rows[s, 4]) / Mv:6.1f}"
            print(f"     S/M={sr:4.2f}: {line}")

    # ---------- 2. which modes carry the S record ----------
    print("\n=== 2. the S record at the final step: contribution of each mode to (logit[p2] - mean logit of the other positions except p1) ===")
    print("    record = run with S minus run without S (x[p2] = b).  Units: logits / S.   Averaged over p1 < p2 for each a2.")
    for b in BG:
        for sr in (0.2, 0.5, 0.8):
            S = sr * Mv
            print(f"  b = {b}, S/M = {sr}:   a2 | " + " ".join(f"{n:>7s}" for n in NAMES + ["pulses", "total"]))
            for a2 in range(0, 9):
                p2 = 9 - a2
                acc = {n: [] for n in NAMES + ["pulses"]}
                for p1 in range(p2):
                    h1 = run(seq(p1, p2, S, b))[0, -1]; h0 = run(seq(p1, p2, b, b))[0, -1]
                    c1, c0 = mode_logits(h1), mode_logits(h0)
                    oth = [p for p in range(10) if p not in (p1, p2)]
                    for n in acc:
                        dl = c1[n] - c0[n]
                        acc[n].append((dl[p2] - dl[oth].mean()) / S)
                vals = [np.mean(acc[n]) for n in NAMES + ["pulses"]]
                print(f"                     {a2}  | " + " ".join(f"{v:7.2f}" for v in vals) + f" {sum(vals):7.2f}")
