"""V8. Claim 5, the part not checked on data by the investigator: "for a new maximum the tape has a slow lobe on the
entries older than the maximum, cancelled by level + runner + clock".
Exact split by source input with the gates of each sequence held fixed (common.traces): h_9 = sum_p J[9, p] x_p.
The maximum's own contribution to each piece of the scores, J[9, p_M] x_{p_M}, per unit of M, on quiet final states,
at the entries OLDER than the maximum (ages a_M + 1 .. 9) and at the YOUNGER old entries (ages 2 .. a_M - 1),
each relative to the maximum's contribution at its own entry?  No: plain values, in logits per unit of M.
Run:  python -B v8_max_lobe.py > v8_max_lobe.txt"""
from vcommon import *
from common import traces

x = sample(60_000, 9801); st = stats(x, 9); q = st["quiet"]; x = x[q]; st = {k: v[q] for k, v in st.items()}; n = len(x); ar = np.arange(n)
out = []
for lo in range(0, n, 10_000):
    xs = x[lo:lo + 10_000]; H, J = traces(xs); out.append(J[np.arange(len(xs)), 9, st["pM"][lo:lo + 10_000]])       # (b, 16): d h_9 / d x_pM
g = np.concatenate(out)                                                                                               # times M / M = per unit of M
go = g[:, ORD]
pcs = {"tape": go @ (Uo @ P_TAPE).T, "level+runner+clock": go @ (Uo @ (PROJ["level"] + PROJ["runner"] + PROJ["clock"])).T,
       "trans+echo": go @ (Uo @ (PROJ["trans"] + PROJ["echo"])).T, "all (with pulse units)": g @ U.T}
print(f"{n} quiet final states (60000 sequences, seed 9801).  Contribution of the maximum's own input to each piece, logits per unit of M (gates fixed).")
i8 = np.broadcast_to(np.arange(8), (n, 8)); iM = 9 - st["aM"]
print("   a_M    n   | OLDER entries (mean over the entries, then mean / sd over states): tape, level+runner+clock, trans+echo, all | corr(tape, LRC) over states | the same for YOUNGER old entries | at the maximum's own entry")
for aM in range(3, 9):
    qa = st["aM"] == aM; row = f"   {aM}  {qa.sum():6d} |"
    for side in (i8 < iM[:, None], (i8 > iM[:, None])):
        cnt = side[qa].sum(1); vals = {k: np.where(side[qa], v[qa][:, :8], 0).sum(1) / np.maximum(cnt, 1) for k, v in pcs.items()}
        ok = cnt > 0
        row += "  ".join(f"{vals[k][ok].mean():6.2f} / {vals[k][ok].std():5.2f}" for k in pcs) + f" | corr {np.corrcoef(vals['tape'][ok], vals['level+runner+clock'][ok])[0, 1]:+.2f} |" if ok.sum() > 10 else "   --   |"
    row += "  ".join(f"{v[qa][np.arange(qa.sum()), iM[qa]].mean():6.2f}" for v in pcs.values())
    print(row)
print("   kernel prediction for a_M = 5 (resting gates, times the pulse of unit 4): tape +1.4..+2.3 and level+runner+clock -0.2..-1.8 per unit pulse on the four older entries")
