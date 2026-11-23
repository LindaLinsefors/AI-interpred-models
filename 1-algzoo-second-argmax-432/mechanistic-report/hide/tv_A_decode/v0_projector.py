"""v0: my own tape projector / coordinates against tape_common.py, and my own top-three statistics against prefix_stats.
Output: v0_projector.txt"""
from vcommon import *
import tape_common as tc

out = Tee(__file__.replace(".py", ".txt"))
x = sample(50_000, 777); H = run(x); h = H[:, 9]
P = sum((2 if n in CPLX else 1) * np.real(np.outer(RV[n], LV[n])) for n in TAPE)
out("eigenvalues used (|lam|, angle in deg):", {n: (round(abs(LAM[n]), 4), round(float(np.degrees(np.angle(LAM[n]))), 2)) for n in TARGET})
out("my projector vs tape_common.P_TAPE, max abs difference:", np.abs(P - tc.P_TAPE).max(), " rank", np.linalg.matrix_rank(P))
out("my tape coordinates vs tape_common.tape_coords:", np.abs(tape_coords(H) - tc.tape_coords(H)).max())
out("my rest coordinates vs modes.coords (level, runner, clock, trans, echo):",
    max(np.abs(mode(H, n) - tc.coords(H)[n]).max() for n in ("level", "runner", "clock", "trans", "echo")))
c = np.random.default_rng(0).standard_normal((len(h), 7))
out("my set_tape vs tape_common.set_tape (random coordinates):", np.abs(set_tape(h, c) - tc.set_tape(h, c)).max())
out("set_tape with own coordinates returns the state:", np.abs(set_tape(h, tape_coords(h)) - h).max())
out("tape part + rest = state on the 13 units, 9 modes complete:",
    np.abs(sum((2 if n in CPLX else 1) * np.real(np.multiply.outer(mode(h, n), RV[n])) for n in TARGET) - h[:, ORD]).max())
for t in (3, 6, 9):
    a, b = stats(x, t), tc.prefix_stats(x, t)
    out(f"t = {t}: stats agree with prefix_stats:", all(np.array_equal(a[k], b[k2]) for k, k2 in
        (("M", "M"), ("S", "S"), ("T", "T3"), ("aM", "aM"), ("aS", "aS"), ("aT", "aT"), ("quiet", "quiet"))), " quiet share", a["quiet"].mean().round(4))
st = stats(x, 9); q = st["quiet"]
out("accuracy on N(0,1):", (answer(h) == second_argmax(x)).mean(), " on quiet final states: answer = runner-up position", (answer(h) == st["pS"])[q].mean().round(4))
s_tape = tape13(tape_coords(h)) @ U[:, ORD].T
out("tape piece alone, top entry among ages >= 2 is the runner-up (quiet final):", (s_tape[:, :8].argmax(1) == st["pS"])[q].mean().round(4),
    "; top entry among all ten:", (s_tape.argmax(1) == st["pS"])[q].mean().round(4))
