# Tape investigation, verifier of arm D_transplant

## Summary

Every headline number of the nine claims reproduces with my own code, my own tape projector (built from the eigen-decomposition; it agrees with tape_common.P_TAPE to 2e-15) and other seeds, to within 0.5 points. The investigator's scripts do what the claims say (they are in hide/t1_transplants/, not t1_D_transplant/); I found no wrong selection, no tape definition that differs, and no fitted quantity that could leak. The claim that the position of an old runner-up is read from the tape and not from the rest is solid: it holds at every step t = 4..9, for uniform inputs, and with positive later inputs, and moving the runner-up between two old slots changes the contrast between those slots by a median 40.0 M, of which 39.1 M is in the tape piece and 0.2 M in the rest piece. The summary sentence "the network reads the two ages and S/M from the tape and nothing else" does not survive. Three results show this: (1) the rest computed from the same zero-padded (M, S) sequence works as well as the tape computed from it (93.7% against 94.7% at t = 9), so that test is not specific to the tape; (2) two real states with the same (a_M, a_S, S/M) have tapes that differ by a median 0.6 of their norm, and exchanging them gives 89.5-90.2% at t = 9 against 96.3% intact (94.8-95.4% against 99.1% even when no third is near the runner-up); (3) the pooled "94.7% against 96.3%" is the net of a 3.2-point loss on the 70% of states without a near third and a 2.1-point gain on the 30% with one. The claim that far-below inputs are "not read" is too strong at t = 9 (the error rate rises from 3.6% to 5.9%), though a control shows those tape directions are 3-4 times less harmful than unrelated ones of equal size. The pooled threshold weight 0.24 ranges from 0.14 to 0.33 by configuration and is not larger when the probe arrives at once at t = 9. I did not recompute the T_pos pairs, the T_up_hi dose series or the investigator's ladder beyond its "top two kept" rung. I did not write tape_v2.md (subagents cannot write .md files here); the corrected answer below is returned for whoever assembles it.

## Corrected answer

**Terms.** A state $h$ is the 16 activations after step $t$. Its tape part is $P h$, where $P$ projects the 13 non-pulse units onto the four modes alt, short, h2, h3; the rest is $h - P h$. A transplant builds the hybrid $h_{A \leftarrow B} = h_A - P h_A + P h_B$ from a recipient A and a donor B. $M$, $S$, $T$ are the largest, second largest (runner-up) and third largest input so far, and $a_M$, $a_S$ their ages (steps since they arrived). "Old" means age $\ge 2$. All states are quiet ($a_M \ge 2$, $a_S \ge 2$, $S > 0$). The scores are $s = U h$; entry $i$ belongs to the input of age $9 - i$. The tape piece is $s_{\text{tape}} = U P h$ and the rest piece is $s - s_{\text{tape}}$. The contrast between two inputs is the difference of their entries.

**What the transplants establish.**

1. *Which old input is the runner-up is written in the tape.* Moving the runner-up from one old slot to another, with everything else fixed, changes the contrast between the two slots by a median $40.0\,M$; $39.1\,M$ of that is in the tape piece and $0.2\,M$ in the rest piece (10% to 90%: $-1.5\,M$ to $+4.5\,M$). Exchanging only the tape between such a pair moves the answer to the donor's runner-up in 85% of states at $t = 9$ and 92-94% at $t = 5..8$; it stays with the recipient's in at most 0.9%. This survives 1-5 further inputs below the runner-up (91-95% with negative inputs, 90-92% with small positive ones) and holds for uniform inputs (83-93%). Deleting the tape leaves 5.6% correct; deleting any other mode group leaves 49-97%.

2. *The exception.* A runner-up of age 2 is followed in 74% of cases when $a_M \le 7$, 22% when $a_M = 8$ and 4% when $a_M = 9$. In the last cell the contrast runner-up against maximum is $+23.9\,M$ in the rest piece and $-7.3\,M$ in the tape piece, so there the rest holds it.

3. *An old third close to the runner-up is in the tape too.* Raising an old third to $S + dM$ in the donor moves the hybrid's answer to it nearly as in the intact network (75% against 86% at $d = +0.02$, 13% against 14% at $d = -0.02$); the reverse hybrid gives at most 0.4%.

4. *Values.* For a later input $z$, the hybrid compares it with an effective runner-up value
$$\theta = S_{\text{rest}} + f\,(S_{\text{tape}} - S_{\text{rest}}), \qquad f = 0.24 \ \text{pooled},\ 0.14\text{-}0.33 \ \text{by configuration}.$$
So about a quarter of the runner-up's value, as a threshold, passes through the tape. For the maximum's value $f = 0.03$.

5. *The maximum's age.* The tape pattern depends on it: a tape that differs only in the maximum's position costs 4.6 points at $t = 6$ (13 points at $t = 9$ by the investigator's figure, which I did not recompute). After a later new maximum the hybrid names the rest's old maximum in 43%, the tape's in 18% and neither in 38%. The answer never follows the tape's maximum, so this is a consistency requirement and not a stored position that is read out.

6. *Encoding.* No single mode carries the runner-up's position (one transplanted mode yields the donor's answer in at most 12%), and a half dose $(c_A + c_B)/2$ of the tape coordinates yields neither position (22% and 19%; the maximum in 35%). The position is a pattern over the entry index built jointly from short, h2 and h3, and its size matters. Without alt the rate is 71% instead of 85%.

**What does not survive.** "The tape stores $V = \{a_M, a_S, S/M\}$ and nothing else."

- A tape computed from the sequence that keeps only $M$ and $S$ (zeros elsewhere) does work in real states: 94.7% against 96.3% intact at $t = 9$. That pooled figure is a 3.2-point loss on states without a third within $0.1\,M$ of the runner-up (96.0% against 99.2%) and a 2.1-point gain on states with one.
- The same test passes for the rest: a rest computed from that sequence with the real tape gives 93.7%. So the test does not single out the tape.
- Two real states with the same $(a_M, a_S, S/M)$ have tapes that differ by a median 0.6 of the tape's norm at $t = 9$. Exchanging them gives 89.5-90.2% against 96.3%; 94.8-95.4% against 99.1% when neither state has a third near the runner-up; 97.0-97.2% against 99.5% after a later new maximum.
- The two ages explain 64% of the variance of the tape piece's old entries (in units of $M$), and adding $S/M$ brings it to 74%.

**Statement from this angle.** The tape stores, as one pattern over the entry index, which old input is the runner-up; this is robust and almost exclusive to the tape once the runner-up is three or more steps old. The pattern's shape also depends on the maximum's age and on $S/M$, and it carries a graded imprint of the other old inputs: decisive for one within about $0.1\,M$ of the runner-up, worth 2-3 points of accuracy at $t = 9$ for inputs more than $0.4\,M$ below it, and at most 0.6 points once further steps have passed. It does not store the maximum's value, and it carries only about a quarter of the runner-up's value as a threshold. A tape that is a function of $(a_M, a_S, S/M)$ alone is enough to make the network answer, but the real tape is not that function.

**Limits of these tests.** 38% of the hybrids at $t = 9$ and 64% at $t = 6$ have a negative entry (median $-0.84\,M$ against activations of about $10\,M$); clipping at zero changes matched-pair rates by at most 5.4 points and no conclusion. Immediate readout at $t = 9$ is a linear statement about the two score pieces; the runs continued for 1-5 steps are the dynamical evidence. How $S/M$ is laid out in the seven coordinates was not measured.

## Verdicts (9)

### [confirmed] 1. The position (age) of the runner-up is read from the tape and not from the rest: exchanging only the tape between two states that differ in the runner-up's position moves the answer to the tape donor's runner-up.

- my numbers: S_pos, rest A + tape B, P(tape's runner-up) / P(rest's runner-up): t = 9 read out at once 85.4% / 0.3% (n = 307,289; clipped 81.9%; reverse hybrid 85.3%); t = 6 running answer 93.8% / 0.1% (n = 218,572; clipped 93.1%). Then low inputs: t = 6 + 3: 91.4% / 0.2% (n = 218,384); t = 4 + 5: 95.3% / 0.4% (n = 73,552); t = 8 + 1: 92.6% / 0.3% (n = 285,461). Unrelated donor rescaled to A's M, positions differ: 76.4% / 2.4% at t = 9 (n = 269,848), 89.4% / 1.5% at t = 6 (n = 178,642). By donor configuration at t = 9: 56 cells median 0.907, range 0.040 to 0.987; donors with 3 <= a_S <= 8: 91.0% / 0.04% (n = 230,618); a_S = 2: 74.3% (a_M <= 7), 21.6% (a_M = 8), 4.0% (a_M = 9, maximum named in 92.2%); a_S = 9: 80.4%. Third within 0.1 M of S: 75.7%, third named 16.4%.
- note: Extra tests that could have failed and did not: every step t = 4..9 (85.7, 92.7, 93.9, 93.4, 91.7, 85.4%; rest's at most 0.9%), uniform inputs (83.2-93.1%), positive later inputs (89.5-91.7% for inputs S*U(0.05, 0.5)). t = 9 is the weakest step, not a typical one. The exception at a_S = 2 with an old maximum is real (see claim 3).

### [confirmed_with_correction] 2. A tape that is a function of only (a_M, a_S, S/M) works almost as well as the real one when transplanted into real states; for the immediate answer the two ages are enough; without any tape the network fails.

- my numbers: Numbers reproduce. t = 9, n = 307,272: M,S-only donor 94.7% (clipped 93.2%) against 96.3% intact; S replaced by 0.6 M: 96.2% (clipped 95.7%); r = 0.1, 0.2, 0.4, 0.8, 0.95: 46.2, 73.3, 92.3, 95.3, 84.6%; tape = 0: 5.6%; S only 0.8%; M only 12.6%. t = 6 (n = 222,995), hybrid / intact: three low 97.0 / 97.3, new max at 7 98.0 / 99.5, new max at 9 95.4 / 98.5, new runner-up at 7 94.6 / 97.1, new third at 7 94.2 / 92.8, own inputs 95.2 / 96.1; with r = 0.6: 98.5, 91.5, 91.7, 99.0, 95.5, 92.6. Cells (a_M, a_S) for the true-S donor: median 0.961, range 0.645 at (9, 2) to 0.994. CORRECTIONS: (a) split hidden in the pooled number: no third within 0.1 M of S (70% of states) 96.0% against 99.2% intact; near third (30%) 91.7% against 89.6% intact. (b) Control: rest of the same artificial donor + A's real tape gives 93.7% at t = 9, and from t = 6: 97.6, 97.7, 97.6, 96.5, 88.4, 95.1% for the six futures. (c) Real donors with the same V: all inputs except the top two redrawn below S: 90.2% at t = 9 (n = 307,625); another real sequence with the same (a_M, a_S) and S/M within 0.02: 89.5% (n = 307,565); with no third within 0.1 M of S in either state 95.4 / 94.8% against 99.1%; from t = 6: three low 96.6 / 96.4 against 97.3, new max at 7 97.2 / 97.0 against 99.5, new max at 9 95.0 / 94.7 against 98.6. (d) Real tape against the artificial one: |difference| / |real tape| median 0.44 (quartiles 0.22, 0.82); held-out variance of the real tape piece (entries of age >= 2, in M) explained by the cell (a_M, a_S) 64%, by cell plus a quadratic in S/M 74%.
- note: What is shown is existence: one particular function of (a_M, a_S, S/M), the tape of the zero-padded sequence, gives a working hybrid. It is not shown, and it is false at the 4-7 point level at t = 9, that real tapes with the same V are interchangeable (the brief's sufficiency criterion). The control (b) shows the test does not single out the tape: a rest built from M and S alone works equally well. "Without any tape the network fails" is confirmed and is specific: deleting the runner mode leaves 48.9%, trans + echo 89.2%, clock 91.0%, level 96.8%, although trans + echo and runner have larger score pieces (17.8 M and 15.1 M rms against 12.1 M for the tape).

### [confirmed] 3. Among entries of age >= 2 the contrasts runner-up against maximum and runner-up against third sit in the tape piece of the scores; the rest piece is nearly flat there and does not veto its own maximum's position.

- my numbers: 307,836 quiet final states. Top entry among ages >= 2: tape piece runner-up 83.9%, maximum 3.5%, third 6.2%; rest piece 15.1%, 21.4%, 9.1%; full 97.3%, 0.7%, 1.8%. Runner-up entry minus maximum entry in M (10/25/50/75/90%): tape 2.6, 5.3, 9.8, 15.3, 21.6 (positive 95.4%); rest -8.5, -5.0, -1.5, 1.0, 3.4 (positive 36.4%). Runner-up minus third: tape -0.1, 4.7, 11.5, 19.2, 29.1; rest -6.8, -3.1, 0.2, 3.5, 8.0. Unrelated donor, tape's runner-up on the rest's maximum: named in 82.7% at t = 9 (n = 38,739), 91.2% at t = 6 (n = 44,335). By a_S (pooled over a_M), runner-up on top: tape piece 50, 82, 84, 94, 92, 91, 89, 89%; rest piece 69, 19, 17, 0, 0, 1, 0, 15% for a_S = 2..9.
- note: "Flat" is true of the median only: state by state the rest piece's runner-up-minus-maximum contrast has a 10-90% range of 12 M and is negative in 64% of states. Two additions. (i) Part of "the rest piece has the runner-up on top when a_S = 2" is a slot bias: the rest piece's top old entry is the age-2 entry in 31.8% of states in which neither the runner-up nor the maximum is there (68.8% when the runner-up is there). (ii) The exception is nevertheless real in one corner: for a_S = 2, a_M = 9 the runner-up-minus-maximum contrast is +23.9 M in the rest piece and -7.3 M in the tape piece (medians, n = 5,458), and a recipient whose runner-up sits elsewhere has -1.6 M there. That explains the unfollowed transplants the investigator listed as unexplained (hybrid contrast positive in 5.8% against 97.0% in the intact donor). For a_M = 8 the rest's share is 7.2 M of 14.5 M, for a_M = 5 it is 1.6 M of 9.4 M.

### [confirmed] 4. The tape holds the order of two close old inputs (third against runner-up) with a resolution of about 0.02 M; the rest holds none of it.

- my numbers: Old third set to S + d M in the donor, P(answer = that input), donor intact / rest A + tape B. t = 9, n = 14,915: d = -0.10: 0.7 / 1.5%; -0.05: 3.1 / 4.5%; -0.02: 14.0 / 12.6%; +0.02: 86.1 / 75.4%; +0.05: 97.3 / 86.7%; +0.10: 99.3 / 91.2%; +0.20: 99.6 / 91.6%; clipped for d > 0: 71.3, 82.2, 87.3, 88.8%. Reverse hybrid (rest B + tape A): 0.0-0.4% at every d. t = 6 then three low inputs, n = 21,684: +0.02: 84.4 / 71.2%; +0.05: 96.2 / 84.3%; +0.20: 99.6 / 95.2%.
- note: The 0.02 M is the intact network's own resolution (14% errors at d = -0.02 and +0.02); the hybrid is 8-13 points worse than the intact donor once the two have swapped roles. I did not recompute the T_pos pairs (9.4% against 0.2%); that part is not checked.

### [confirmed_with_correction] 5. The runner-up's value as the threshold for a later input is read mostly from the rest: effective threshold S_rest + 0.24 (S_tape - S_rest). The maximum's value is not read from the tape (weight 0.02).

- my numbers: Half-point f by interpolation on a grid of 0.05 in u, given for value_B > value_A / value_B < value_A. Runner-up, transplant at t = 6, probe at step 7 (n = 122,147): f = 0.24 / 0.24; reverse hybrid 0.76 / 0.76; intact A 0.01 / -0.01. By configuration: a_S = 2: 0.15 / 0.23; 3: 0.14 / 0.24; 4: 0.28 / 0.22; 5: 0.31 / 0.25; 6: 0.27 / 0.25; a_M = 2..6: 0.23/0.17, 0.20/0.24, 0.23/0.31, 0.23/0.25, 0.33/0.19; runner-up before the maximum 0.26 / 0.24, after 0.21 / 0.23. Other delays: probe at step 8: 0.24 / 0.22; at step 9: 0.18 / 0.28; transplant at t = 8, probe at 9: 0.20 / 0.24 (n = 140,148); transplant at t = 4, probe at 5: 0.14 / 0.26 (n = 74,832). Maximum (n = 112,486): f = 0.03 / 0.03; at most 0.07 by configuration except a_M = 6 (0.13 / 0.14) and a_S = 2 with M_B > M_A (0.10).
- note: The pooled 0.24 and 0.02-0.03 reproduce. Corrections to the caveat: across configurations f lies between 0.14 and 0.33, never above one third, and the two signs of the change can differ by 0.10, so it is a blur and not a clean shift of one threshold; the weight is not larger when the probe arrives at once at t = 9 (0.20 / 0.24). The oldest maximum (a_M = 6 at t = 6, the first input) has a tape weight of 0.13-0.14, not 0.02.

### [confirmed] 6. The position of the maximum, as needed after a later new maximum, is not cleanly in the tape or in the rest: the rest dominates, the tape contributes, a mismatch breaks the answer.

- my numbers: M_pos at t = 6, new maximum 1.3 M at step 7, two low inputs, n = 218,447: rest's old maximum 43.4%, tape's old maximum 18.4%, old runner-up 19.9%, new maximum 8.1%, other 10.3% (intact 99.5%); clipped 44.4 / 17.7 / 19.1 / 8.6 / 10.2%. New maximum at step 9: 37.7, 19.4, 26.3, 0.0, 16.5% (intact 98.6%). Present answer of the M_pos hybrid at t = 6: runner-up 92.0%, tape's maximum 0.9%. Tape deleted at t = 6 then new maximum at 7: 72.4% correct; M,S-donor tape 98.0%.
- note: Split by configuration, which the investigator had not done: by recipient's a_M the rest's old maximum is named in 30-61% and the tape's in 9-27%; by donor's a_M the tape's is named in 1-35% (lowest, 1-3%, when the donor's maximum is the first input). It is not clean in any cell. I did not recompute the t = 9 present-answer figure (83.8%).

### [confirmed_with_correction] 7. The network does not read the values or positions of inputs far below the runner-up from the tape, although the tape changes with them.

- my numbers: low_val: tape changes by a median 0.16 of its norm (quartiles 0.09, 0.28); hybrid 94.1% against 96.4% intact at t = 9 (n = 307,331); 96.1 against 96.4% at t = 6. low_pos: 95.0 against 96.5% at t = 9 (n = 302,392); 96.8 against 96.9% at t = 6. From t = 6, low_val: three low 97.2 against 97.4; new max at 7 99.0 against 99.5; at 9 98.0 against 98.6. All but the top two redrawn below S: 90.1-90.2% at t = 9, 95.5% at t = 6 (running answer, intact 96.3%). CONTROL: a tape change of the same size taken from another pair (rescaled to this M) gives 88.1% (low_val) and 90.6% (low_pos) at t = 9, 92.8% and 94.8% at t = 6. States in which every other input is more than 0.4 M below S in both recipient and donor: 95.2% against 98.5% at t = 9 (n = 14,459).
- note: The control supports the claim's content: the directions in which the tape varies with far-below inputs cost 2.3 points, unrelated changes of the same size cost 8.3. But "does not read" is too strong at t = 9: the error rate rises from 3.6% to 5.9% (low_val), and from 1.5% to 4.8% in the states where all other inputs are far below. From t = 6 onward the loss is at most 0.6 points. Suggested wording: "weakly read at t = 9, not read after further steps".

### [confirmed] 8. No single tape mode carries the runner-up's position, and the code is graded: a half dose of the donor's tape yields neither position.

- my numbers: S_pos at t = 9, n = 307,226, P(A's runner-up) / P(B's runner-up). Only one mode from B: alt 79.5 / 0.2%, short 31.4 / 12.1%, h2 43.0 / 3.0%, h3 45.2 / 6.3%. All but one from B: alt 0.7 / 70.6%, short 13.6 / 30.9%, h2 3.6 / 38.8%, h3 7.1 / 42.0%. t = 6 (n = 218,437): all but alt 0.4 / 81.7%; only h3 33.1 / 14.4%. Dose q at t = 9: 0.2: 71.1 / 0.4%; 0.35: 41.6 / 7.2%; 0.5: 22.1 / 18.6% (maximum 35.3%, third 20.5%); 0.65: 9.4 / 36.6%; 0.8: 1.4 / 63.5%; 1.0: 0.3 / 85.4%. Clipped values differ by at most 7 points.
- note: Consistent with my deletion test at t = 9: without alt 90.6% correct, without short 46.1% (maximum named in 32.3%), without h2 68.5%, without h3 65.0%. The T_up_hi dose series was not recomputed.

### [confirmed_with_correction] 9. The hybrids leave the reachable set often but only slightly, and the conclusions do not depend on it.

- my numbers: S_pos: negative entry in 38.4% of hybrids at t = 9 and 63.8% at t = 6. t = 9: tape's runner-up named in 87.0% without a negative entry and 82.8% with one; most negative entry among the latter: median -0.84 M, 10% quantile -3.15 M; rms of intact activations 11.9 M (rms of h / M; the investigator's 10.3 M is a ratio of means). Clipping: 85.4 to 81.9% at t = 9, 93.8 to 93.1% at t = 6; at t = 7 and 8: 93.4 to 88.0% and 91.7 to 87.2%; unrelated donors at t = 6: 89.4 to 79.9%.
- note: Small corrections: at steps the investigator did not test the clipping effect on matched pairs reaches 5.4 points (t = 7), not 4.5, and for unrelated donors at t = 6 it is 9.5 points, not 8. "Only slightly" holds for the median; one in ten of the affected hybrids has an entry below -3 M. None of the conclusions changes sign under clipping. As the investigator says, non-negativity is necessary, not sufficient, for reachability.

## Attacks (11)

### Control for claim 2 (v2): build the REST, not the tape, from the same zero-padded (M, S) sequence and keep A's real tape.

- result: t = 9: 93.7% correct (clipped 93.5%), against 94.7% for the artificial tape with the real rest and 96.3% intact. From t = 6: three low 97.6%, new maximum at 7 97.7%, at 9 97.6%, new runner-up 96.5%, new third 88.4%, own inputs 95.1% (artificial tape: 97.0, 98.0, 95.4, 94.6, 94.2, 95.2%).
- consequence: The test "a part computed from M and S alone still works" passes equally for the rest, so it says little about what the tape in particular stores. It shows that either piece of a zero-padded sequence is compatible with the real other piece to 1-3 points.

### Sufficiency with REAL donors of the same V = (a_M, a_S, S/M) (v4): (R1) the recipient with every input except the top two redrawn below S; (R2) another real quiet sequence with the same (a_M, a_S) and S/M within 0.02, rescaled to the recipient's M.

- result: The two tapes differ by a median 0.59-0.61 of the tape's norm at t = 9 (0.28-0.30 at t = 6). Hybrid accuracy at t = 9: 90.2% (R1), 89.5% (R2) against 96.3% intact. With no third within 0.1 M of the runner-up in either state: 95.4 / 94.8% against 99.1%. Near third in the donor only: 84.5 / 83.2% against 99.3%. From t = 6: three low 96.6 / 96.4% against 97.3%; new maximum at 7: 97.2 / 97.0% against 99.5%; at 9: 95.0 / 94.7% against 98.6%. The reverse hybrids give the same rates.
- consequence: The real tape is not a function of (a_M, a_S, S/M) in the brief's sense: states with the same V have clearly different tapes and exchanging them costs 6-7 points at t = 9, 4 points even without any near third, 2.5-4 points after a later new maximum. The summary's "it reads nothing else" is refuted at that level; the artificial donor works better than a real one because it adds no wrong imprint of other inputs.

### Is the real tape close to the artificial (M, S)-only tape? (v2, direct measurement instead of a functional test.)

- result: |real - artificial| / |real| has median 0.44 (quartiles 0.22, 0.82) over 307,272 quiet final states. For the tape piece of the scores at ages >= 2, in units of M: rms real 13.3, artificial 16.2, difference 8.6. Held-out share of variance explained by the cell (a_M, a_S): 64%; by the cell plus a quadratic in S/M: 74%.
- consequence: About three quarters of the tape piece is a function of the two ages and S/M; a quarter is not, and that quarter matters functionally (previous attack). "The tape stores V" holds for the bulk, not exactly.

### Pooled number split (v2): the artificial-tape hybrid by whether a third lies within 0.1 M of the runner-up.

- result: No near third (70% of states): 96.0% against 99.2% intact, i.e. errors rise from 0.8% to 4.0%. Near third (30%): 91.7% against 89.6% intact, the hybrid is better because the artificial tape has no third. With S replaced by 0.6 M the pooled 96.2% "equal to intact" is the same kind of cancellation.
- consequence: The headline "94.7% against 96.3%" hides a five-fold rise in errors on ordinary states, offset by removing the network's near-tie errors. It should be reported split.

### Control for claim 7 (v1): replace the matched low-input tape change by an equally large tape change taken from another pair.

- result: t = 9: matched change 94.1% (low_val) and 95.0% (low_pos); unrelated change of the same median size 88.1% and 90.6%; intact 96.4-96.5%. t = 6: 96.1 / 92.8% and 96.8 / 94.8%.
- consequence: The claim could have been trivial ("a 16% change of the tape never matters") and is not: the tape's dependence on far-below inputs lies in directions that cost 3-4 times less than generic ones. It still costs 2.3 points at t = 9, so "not read" should be "weakly read".

### Deletion controls (v2): delete other mode groups instead of the tape, quiet final states.

- result: Tape deleted 5.6%; runner 48.9%; trans + echo 89.2%; clock 91.0%; level 96.8%. Single tape modes: alt 90.6%, short 46.1%, h2 68.5%, h3 65.0%.
- consequence: "Without any tape the network fails" is specific to the tape and not a generic effect of removing a large piece of the state. Among the tape modes short matters most and alt least.

### Other populations (v2): every step t = 3..9 with the running answer, and inputs uniform on [-1, 1].

- result: S_pos follow rate, Gaussian t = 4..9: 85.7, 92.7, 93.9, 93.4, 91.7, 85.4% (rest's at most 0.9%); uniform: 85.4, 92.1, 93.1, 92.2, 89.7, 83.2%. Artificial (M, S) tape against intact, Gaussian t = 3..9: 94.0/95.7, 93.9/94.7, 95.6/95.8, 96.5/96.3, 97.1/96.7, 97.0/96.8, 94.7/96.3; uniform t = 9: 93.1/95.3. No S_pos pair exists at t = 3.
- consequence: Claims 1 and 2 are not artefacts of t = 9 or of Gaussian inputs. t = 9 is the least favourable step for both.

### Later inputs of the other sign (v4): S_pos transplant followed by positive inputs below the runner-up, which the investigator had not tested.

- result: Inputs S*U(0.05, 0.5): tape's runner-up 89.5% (t = 6 + 3), 90.9% (t = 4 + 5), 91.7% (t = 8 + 1), rest's at most 0.8% (intact 95.2-96.5%). Zeros: 90.2, 92.4, 92.2%. Inputs S*U(0.5, 0.9): 80.2, 70.0, 84.7%, a later input named in 14.2, 25.8, 8.7% (intact 92.9, 89.3, 94.6%).
- consequence: The answer keeps following the tape with positive later inputs. When those inputs come within a factor two of the runner-up the hybrid is 10-19 points below the intact network, so the transplanted runner-up is held less firmly than a native one.

### Is "the answer follows the tape" trivial because the rest barely changes? (v5: exact split of the score change of S_pos pairs at t = 9, n = 246,024.)

- result: Change of the contrast between the slot the runner-up enters and the slot it leaves: full scores median 40.0 M (10-90%: 22.8 to 82.0), tape piece 39.1 M, rest piece 0.2 M (-1.5 to 4.5). Share of the squared score change over ages >= 2: rest 6.6%, nearly all in the trans mode; clock 0.3%, runner 0.2%, level, echo and pulse units 0.0%.
- consequence: This is the claim in its most direct form and it gives the size: the old runner-up's position is written in the tape piece to within a few percent. The clock, the only mode of the rest that could hold an age, does not carry it.

### Alternative reading of the age-2 exception (v6): is "the rest piece has the runner-up on top when a_S = 2" a slot bias instead of stored information?

- result: The rest piece's top old entry is the age-2 entry in 31.8% of states where neither runner-up nor maximum is there, and in 68.8% when the runner-up is there. In the cell a_S = 2, a_M = 9 the runner-up-minus-maximum contrast is +23.9 M in the rest piece and -7.3 M in the tape piece; a recipient with its runner-up elsewhere has -1.6 M in the rest piece, so the hybrid's contrast is positive in 5.8% (97.0% in the intact donor). For a_M = 8: rest 7.2 M, tape 9.5 M; for a_M = 5: rest 1.6 M, tape 7.9 M.
- consequence: My alternative fails in part: about half of the pooled age-2 hit rate is slot bias, but in the corner (young runner-up, maximum 8-9 steps old) the rest really carries the contrast against the maximum. The investigator's limit stands and its "unexplained" failure now has a proximate cause.

### Claim 5 split by configuration and by the probe's delay (v3), which the investigator had flagged as not done.

- result: f = 0.24 pooled; 0.14-0.33 across a_S and a_M cells; 0.24 / 0.22 with the probe two steps after the transplant, 0.18 / 0.28 three steps after; 0.20 / 0.24 for transplant at t = 8 with the probe at t = 9; 0.14 / 0.26 for t = 4. Maximum: 0.03 pooled, 0.13-0.14 when the maximum is the first input.
- consequence: The quarter share is stable to within about 0.1 and never exceeds one third. The investigator's remark that it looks larger when the probe arrives at once at t = 9 is not supported.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v_common.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v_common.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v1_positions.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v1_positions.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v2_minimal_donor.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v2_minimal_donor.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v3_values_modes.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v3_values_modes.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v4_same_V_real_donor.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v4_same_V_real_donor.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v5_where_the_change_lives.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v5_where_the_change_lives.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v6_base_rate.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/tv_D_transplant/v6_base_rate.txt