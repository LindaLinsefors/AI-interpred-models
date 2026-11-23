"""Shared definitions for the question "what does the tape store?" (2026-11-23).

The state of the 13 non-pulse units o = ORD is an exact sum of nine mode patterns of the resting map A = W[o, o]
(modes.py):   h[o] = sum_real zeta_m r_m + sum_pairs 2 Re(z_m r_m),   zeta_m = l_m . h[o]  (LEFT, RIGHT).

THE TAPE = the four modes  alt (real, lam = -0.928), short (0.735 at 42.99 deg), h2 (0.937 at 83.28 deg),
h3 (0.837 at 124.78 deg).  It is 7 real numbers per state:
    tape coordinates  c = (zeta_alt, Re z_short, Im z_short, Re z_h2, Im z_h2, Re z_h3, Im z_h3)      tape_coords(H)
    tape part of the state   h_tape[o] = P_TAPE h[o]   (zero on the pulse units 4, 5, 9)              tape_part(H)
    rest of the state        h - h_tape   (the modes level, runner, clock, trans, echo + pulse units)  rest_part(H)
    tape piece of the scores s_tape = U[:, o] h_tape[o]   (eq. 4a of ../tape.md)                      tape_scores(H)
All functions take states of shape (..., 16).  Everything is linear in the state and exact (checked in __main__).

A STATE AT STEP t of a sequence x is h_t(x_0..x_t).  M_t, S_t, T_t are the three largest of x_0..x_t, and the age
of an input x_p at step t is t - p.  QUIET STATE (the user's filter): no new maximum and no new runner-up in the
last two steps, and a positive runner-up:  a_M >= 2, a_S >= 2, S_t > 0.     prefix_stats(x, t)["quiet"]
The network is positively homogeneous: scaling all inputs by c > 0 scales the state and the tape by c.
"""
from common import *
from modes import LEFT, RIGHT, LAM, NAMES, IS_COMPLEX, coords

TAPE = ["alt", "short", "h2", "h3"]
REST = [n for n in NAMES if n not in TAPE]              # level, runner, clock, trans, echo


def projector(names):
    """13 x 13 real matrix P with P h[o] = the part of h[o] that lies in the given modes."""
    P = np.zeros((13, 13))
    for n in names:
        m = np.outer(RIGHT[n], LEFT[n])
        P += 2 * m.real if IS_COMPLEX[n] else m.real
    return P


P_TAPE, P_REST = projector(TAPE), projector(REST)


def tape_coords(H):
    """(..., 16) -> (..., 7): zeta_alt, Re z_short, Im z_short, Re z_h2, Im z_h2, Re z_h3, Im z_h3."""
    Ho = H[..., ORD]
    out = [np.real(Ho @ LEFT["alt"])]
    for n in ("short", "h2", "h3"):
        z = Ho @ LEFT[n]
        out += [z.real, z.imag]
    return np.stack(out, -1)


def coords_to_tape13(c):
    """(..., 7) tape coordinates -> (..., 13) tape part of h[o]."""
    out = np.multiply.outer(c[..., 0], RIGHT["alt"].real)
    for j, n in enumerate(("short", "h2", "h3")):
        out = out + 2 * np.real(np.multiply.outer(c[..., 1 + 2 * j] + 1j * c[..., 2 + 2 * j], RIGHT[n]))
    return out


def tape_part(H):
    """(..., 16) -> (..., 16): the tape part of the state (zero on the pulse units)."""
    out = np.zeros(H.shape)
    out[..., ORD] = H[..., ORD] @ P_TAPE.T
    return out


def rest_part(H):
    return H - tape_part(H)


def swap_tape(H_recipient, H_donor):
    """The recipient's state with its tape part replaced by the donor's tape part. Can have negative entries."""
    return rest_part(H_recipient) + tape_part(H_donor)


def set_tape(H, c):
    """The state H with its tape coordinates replaced by c (..., 7)."""
    out = rest_part(H)
    out[..., ORD] += coords_to_tape13(c)
    return out


def tape_scores(H):
    """(..., 16) -> (..., 10): the tape piece of the scores, eq. (4a) of tape.md.  Entry i belongs to age 9 - i."""
    return H[..., ORD] @ P_TAPE.T @ U[:, ORD].T


def state_at(x, t):
    """State after step t (inputs x[:, 0..t]), shape (B, 16)."""
    return run(x[:, : t + 1])[:, -1]


def continue_from(h, x_future):
    """Run on from state h (B, 16) with further inputs x_future (B, T'); returns all later states (B, T', 16)."""
    return run(x_future, h0=h)


def answer(h9):
    """The network's answer (a position 0..9) from final states (B, 16)."""
    return (h9 @ U.T).argmax(-1)


def prefix_stats(x, t):
    """Top three of x_0..x_t: values M, S, T3, positions pM, pS, pT, ages aM, aS, aT, and the quiet-state mask."""
    pre = x[:, : t + 1]
    order = np.argsort(-pre, 1, kind="stable")
    ar = np.arange(len(x))
    out = {}
    for k, nm in enumerate(("M", "S", "T3")[: min(3, t + 1)]):
        p = order[:, k]
        out["p" + nm[0]], out[nm], out["a" + nm[0]] = p, pre[ar, p], t - p
    if t >= 1:
        out["quiet"] = (out["aM"] >= 2) & (out["aS"] >= 2) & (out["S"] > 0)
    return out


if __name__ == "__main__":
    x = sample(20000, 1)
    H = run(x)
    h = H[:, 9]
    print("P_TAPE + P_REST = identity:", np.abs(P_TAPE + P_REST - np.eye(13)).max())
    print("P_TAPE is a projector:     ", np.abs(P_TAPE @ P_TAPE - P_TAPE).max(), " rank", np.linalg.matrix_rank(P_TAPE))
    print("coords -> tape part:       ", np.abs(coords_to_tape13(tape_coords(h)) - tape_part(h)[:, ORD]).max())
    c = coords(h); Uo = U[:, ORD]
    G = np.real(np.multiply.outer(c["alt"], Uo @ RIGHT["alt"]))
    for n in ("short", "h2", "h3"):
        G = G + 2 * np.real(np.multiply.outer(c[n], Uo @ RIGHT[n]))
    print("tape_scores = eq. (4a):    ", np.abs(G - tape_scores(h)).max())
    print("set_tape(H, own coords):   ", np.abs(set_tape(h, tape_coords(h)) - h).max())
    print("swap_tape(H, H) = H:       ", np.abs(swap_tape(h, h) - h).max())
    h4 = state_at(x, 4)
    print("state_at + continue_from:  ", np.abs(continue_from(h4, x[:, 5:])[:, -1] - h).max(), " state_at(4) vs run:", np.abs(h4 - H[:, 4]).max())
    xr = np.zeros_like(x); xr[:, 4:] = x[:, :6]
    print("zero-prefix identity:      ", np.abs(run(xr)[:, 9] - H[:, 5]).max())
    st = prefix_stats(x, 9)
    print("quiet final states:        ", st["quiet"].mean(), " accuracy:", np.mean(answer(h) == second_argmax(x)))
    print("rms of the tape coordinates / M on quiet final states:", np.round(np.sqrt(((tape_coords(h) / st["M"][:, None])[st["quiet"]] ** 2).mean(0)), 2))
    top = lambda A: A[:, :8].argmax(1)
    q = st["quiet"]
    print("runner-up is the top entry among ages >= 2:  tape piece", np.mean(top(tape_scores(h))[q] == st["pS"][q]).round(3),
          " full score", np.mean(top(h @ U.T)[q] == st["pS"][q]).round(3))
