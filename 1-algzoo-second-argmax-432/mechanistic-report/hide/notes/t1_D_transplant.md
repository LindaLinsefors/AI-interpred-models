# Tape investigation, arm D_transplant

## Answer

Terms. A transplant takes a recipient state A and a donor state B (both quiet, same step $t$) and builds the hybrid $h = \mathrm{rest}(h_A) + \mathrm{tape}(h_B)$. "The answer follows the tape" means the network then names the position that is correct for B. $a_M$ and $a_S$ are the ages of the maximum $M$ and the runner-up $S$; "old" means age $\ge 2$.

From the transplants, the rest of the network reads three things from the tape: the two ages $a_S$ and $a_M$, and (weakly) the ratio $S/M$. It reads nothing else that I could detect, except the position and value of an old third input when it lies within about $0.1M$ of the runner-up.

1. **Which old input is the runner-up.** When B is A with the runner-up moved to another old position, the hybrid names the tape's runner-up in 85% of states at $t=9$ and 94% at $t=6$, and the rest's runner-up in 0.3% and 0.1%. This persists: 91-95% after 1-5 further low inputs. For donors with $3 \le a_S \le 8$ the rate is 91%, with 0.0% for the rest's.

2. **The contrast against the maximum.** Among old entries the tape piece puts the runner-up above the maximum's own entry by a median $9.8M$ and above the third's by $11.5M$. The rest piece is flat there ($-1.6M$ and $+0.1M$). The rest does not veto its own maximum: when an unrelated tape's runner-up sits on the rest's maximum, the hybrid names that position in 83%.

3. **Constructive test.** A tape taken from an artificial sequence with only $M$ and $S$ at their positions and zeros elsewhere is a function of $(a_M, a_S, S/M)$ times $M$. Transplanted into real states it gives 94.7% correct at $t=9$ (intact 96.3%), and 94-98% after chosen futures from $t=6$ (intact 93-99.5%). With $S$ replaced by $0.6M$ it gives 96.3% at $t=9$, equal to intact, but only 91.5% after a later new maximum. Deleting the tape gives 5.7%.

4. **Graded values near the top.** Raising an old third to $S + dM$ moves the hybrid's answer to it almost as in the intact network (75% against 86% at $d=+0.02$, 13% against 14% at $d=-0.02$); the rest contributes at most 0.5%. For a later input, the effective threshold "is it above the runner-up" is $S_{\text{rest}} + 0.24\,(S_{\text{tape}} - S_{\text{rest}})$, so the tape carries about a quarter of the runner-up's value. For the maximum's value the tape's weight is 0.02.

5. **Not read from the tape.** Values and positions of inputs more than $0.4M$ below the runner-up cost at most 2.2 points at $t=9$ and 0.6 points from $t=6$. After a later new maximum, the old maximum's position follows the rest in 43% and the tape in 18%, so it is not cleanly in either.

6. **Encoding.** No single tape mode carries the runner-up's position: one transplanted mode destroys A's answer (31-45% left for short, h2, h3; 80% for alt) but yields B's in at most 12%. A half dose $(c_A + c_B)/2$ gives neither position (59% other). So it is a bump over the entry index built jointly from short, h2 and h3, and its height matters.

Limits: a runner-up of age 2 is still partly held by the rest. Donors with $a_S = 2$ and $a_M = 8, 9$ are followed in only 21% and 4%, and 30-65% of hybrids have a slightly negative entry (clipping changes matched-pair rates by at most 4.5 points). How $S/M$ is laid out in the seven coordinates was not measured here.

I did not write `tape_v2.md`; these findings are returned for it.

## Claims (9)

### 1. [strong] The position (age) of the runner-up is read from the tape and not from the rest: when only the tape is exchanged between two states that differ in the runner-up's position, the answer goes to the tape donor's runner-up and essentially never stays with the recipient's.

- evidence: S_pos pairs (B = A with the runner-up swapped with an old input more than 0.4 M below it), hybrid = A's rest + B's tape. t = 9, read out at once, n = 368,373 pairs from 600,000 N(0,1) sequences: tape's runner-up 85.4%, rest's runner-up 0.3%, maximum 4.2%, third 7.0%, other 3.0% (intact 96.4%). t = 6, running answer, n = 261,979: 93.9% against 0.1%. The reverse hybrid (B's rest + A's tape) gives the mirror image (85.3% and 93.9% for the tape owner). After further low inputs: transplant at t = 6 then 3 low inputs 91.4% against 0.2% (n = 262,297); at t = 4 then 5 low inputs 95.2% against 0.4% (n = 88,015); at t = 8 then 1 low input 92.5% against 0.3% (n = 341,659). Unrelated donor rescaled to A's M, rows where the two runner-up positions differ: 76.6% against 2.4% at t = 9 (n = 214,855), 89.5% against 1.5% at t = 6 (n = 142,668). Split by donor configuration at t = 9 (800,000 sequences, 56 cells of (a_M, a_S)): median 0.904, range 0.039 to 0.989; donors with 3 <= a_S <= 8: 91.0% tape's, 0.0% rest's (n = 369,288).
- script: t1_transplants/a1_immediate.py (a1_immediate.txt); a3_future.py (a3_future_t4.txt, a3_future_t6.txt, a3_future_t8.txt); a2_modes_dose.py (a2_modes_dose.txt); a9_by_configuration.py (a9_by_configuration.txt); a10_weak_cells.py (a10_weak_cells.txt)
- caveats: At t = 9 the readout is linear, so the immediate result is a statement about the sizes of the two score pieces' contrasts; the runs continued for 1-5 steps are the dynamical test. Exceptions by configuration: donors with a_S = 2 are followed in 74.0% (a_M <= 7), 21.0% (a_M = 8) and 3.8% (a_M = 9, where the hybrid names the maximum in 92.7%); donors with a_S = 9 in 80.8%. When the third is within 0.1 M of the runner-up the rate falls to 75.7% (t = 9) and the third is named in 16.5%. 38% (t = 9) to 64% (t = 6) of these hybrids have a negative entry; clipped at zero the rates are 81.9% and 93.2%.

### 2. [strong] A tape that is a function of only (a_M, a_S, S/M) works almost as well as the real one when transplanted into real states, and for the immediate answer the two ages alone are enough. Without any tape the network fails.

- evidence: Donor = artificial sequence, zero everywhere in the prefix except M and S at A's positions; hybrid = A's rest + that tape. t = 9 read out at once, n = 369,358 quiet states: 94.7% correct (clipped 93.2%) against 96.3% intact. With the runner-up's value replaced by 0.6 M (tape depends on the two ages only): 96.2% (a second sample, n = 369,467: 96.3%; clipped 95.7%). By fixed ratio r = S'/M: 0.1: 46.3%, 0.2: 73.5%, 0.4: 92.4%, 0.6: 96.3%, 0.8: 95.4%, 0.95: 84.6%. Transplant at t = 6 (n = 267,692) with the true S, hybrid against intact: three low inputs 97.0 / 97.3%, new maximum at step 7 98.1 / 99.5%, new maximum at step 9 95.4 / 98.5%, new runner-up at step 7 94.6 / 97.1%, new third at step 7 94.1 / 92.9%, A's own inputs 95.2 / 96.1%. With r = 0.6 instead: 98.5, 91.5, 91.7, 99.0, 95.4, 92.6%. Controls at t = 9: tape set to zero 5.7% (answers scattered: maximum 16.4%, third 20.3%, inputs of age 0 or 1 31.0%); donor with only S 0.9%; donor with only M 12.6%. By (a_M, a_S) at t = 9 with r = 0.6 (n = 492,417): cells 0.709 to 0.998, median 0.982; intact 0.853 to 0.991, median 0.969.
- script: t1_transplants/a7_minimal_donor.py (a7_minimal_donor.txt); a8_two_number_tape.py (a8_two_number_tape.txt); a9_by_configuration.py (a9_by_configuration.txt)
- caveats: This shows sufficiency of V = {a_M, a_S, S/M} to within 0.3-3 points depending on the future, not that the real tape contains nothing else. The donors are artificial sequences, and 35-61% of the hybrids have a negative entry (clipped values agree within 1.5 points). Adding more of the real inputs to the donor did not help at t = 9: top three only 93.1%, top four only 93.2%. The fixed-ratio tape is good only for the present answer; it loses 8 points after a later new maximum, so S/M in the tape is used there. Weak cells for r = 0.6: (a_M, a_S) = (2, 9) 0.709, (2, 8) 0.827.

### 3. [strong] Among entries of age >= 2, the whole contrast 'runner-up against maximum' and 'runner-up against third' sits in the tape piece of the scores; the rest piece is nearly flat there, and the rest does not veto its own maximum's position.

- evidence: 369,576 quiet final states. Top entry among ages >= 2: tape piece runner-up 84.0%, maximum 3.4%, third 6.2%, other 6.3%; rest piece 15.0%, 21.6%, 9.0%, 54.4%; full scores 97.4%, 0.7%, 1.7%, 0.2%. Entry of the runner-up minus entry of the maximum, in units of M (10/25/50/75/90%): tape piece 2.6, 5.3, 9.8, 15.4, 21.7 (positive in 95.5%); rest piece -8.5, -5.0, -1.6, 1.0, 3.4 (positive in 36.1%). Runner-up minus third (third of age >= 2): tape piece -0.1, 4.7, 11.5, 19.3, 29.1; rest piece -6.8, -3.2, 0.1, 3.4, 8.0. Transplant test with an unrelated donor at t = 9: in the 46,158 pairs where the tape's runner-up sits on the rest's maximum, the hybrid names that position in 83.0%. A donor containing only one input (S alone, which is then that sequence's maximum) makes the hybrid name that position in 0.9%, so a lone maximum's tape pattern does not promote its own entry.
- script: t1_transplants/a4_breakdown.py (a4_breakdown.txt); a7_minimal_donor.py (a7_minimal_donor.txt); a10_weak_cells.py (a10_weak_cells.txt)
- caveats: Strongly configuration dependent for young runner-ups: with a_S = 2 the rest piece alone has the runner-up on top in 34-99% of states (91-99% for a_M = 3..5) and the tape piece alone in only 22-72%; with a_S = 3, 4 the rest piece alone still has it in 5-56%; with a_S = 5..8 the rest piece never has it (0.0-4.7%) and the tape piece has it in 81-99.7%. So the runner-up's position is fully in the tape only from about age 3-5 on.

### 4. [strong] The tape holds the order of two close old inputs (third against runner-up) with a resolution of about 0.02 M; the rest holds none of it. When a third is within 0.1 M of the runner-up, the hybrid's errors name the tape's third, not the rest's.

- evidence: A: third more than 0.4 M below S, of age >= 2, M - S > 0.25 M. B_d: that input set to S + d M. P(answer = that input) for B intact / A's rest + B's tape, t = 9, n = 17,688: d = -0.10: 0.8 / 1.6%; -0.05: 3.1 / 4.5%; -0.02: 14.0 / 12.8%; +0.02: 85.7 / 75.5%; +0.05: 97.0 / 87.1%; +0.10: 99.3 / 91.6%; +0.20: 99.8 / 92.2%. Reverse hybrid (B's rest + A's tape): 0.0-0.5% at every d. Transplant at t = 6 then three low inputs (n = 26,190): +0.02: 84.4 / 71.6%; +0.05: 96.1 / 84.3%; +0.20: 99.6 / 95.6%. T_pos pairs (third within 0.1 M of S, moved to another old position; t = 9, n = 83,662): hybrid names the tape's third in 9.4% and the rest's third in 0.2% (intact B names its third in 7.6%); the runner-up is still named in 85.4% (intact 91.0%).
- script: t1_transplants/a5_values.py (a5_values.txt); a1_immediate.py (a1_immediate.txt)
- caveats: On the side where the two swap roles the hybrid is 8-10 points below the intact donor, so a small part of that comparison needs the matching rest. Clipped rates are up to 4.5 points lower for d > 0. The third's position and value are functionally irrelevant to the task, so this content shows up as the network's near-tie errors rather than as something it needs.

### 5. [strong] The runner-up's value as the threshold for a later input is read mostly from the rest: the effective threshold is S_rest + 0.24 (S_tape - S_rest). The maximum's value is not read from the tape (weight 0.02).

- evidence: S_val pairs (runner-up's value redrawn uniformly between the third and M, |S_B - S_A| > 0.15 M, n = 146,473), transplant at t = 6, then z = S_A + u (S_B - S_A) at step 7 and two low inputs; P(z counted as a new runner-up, i.e. answer = step 7) for A's rest + B's tape: S_B > S_A: 14.5% at u = 0.1, 36.8% at 0.2, 81.6% at 0.35, 95.2% at 0.5; S_B < S_A: 86.0%, 61.1%, 16.8%, 4.1%. Half-point by linear interpolation u = 0.24 in both; for the reverse hybrid u = 0.76 in both. At u = 0.5 the answer is correct for the rest's owner in 91.9% and for the tape's owner in 5.1% (t = 4: 91.7 / 3.8%; t = 8, z arriving at once: 79.6 / 12.6%). M_val pairs (M' = S + g (M - S), g log-uniform in [1/3, 3], |M_B - M_A| > 0.15 M, n = 134,833), z = M_A + u (M_B - M_A): half-point u = 0.02 (intact A: 0.00); at u = 0.5 the answer follows the rest's owner in 99.2% and the tape's in 0.3%.
- script: t1_transplants/a5_values.py (a5_values.txt); a3_future.py (a3_future_t6.txt, a3_future_t4.txt, a3_future_t8.txt)
- caveats: The 0.24 is one pooled number over all configurations at t = 6 with z arriving one step after the transplant; it was not split by a_M, a_S and it is larger when z arrives immediately at t = 9. For the present answer a changed runner-up value in the tape costs little (S_val at t = 9: 96.0% against 96.4%), but changes above 0.23 M cost 4-6 points.

### 6. [moderate] The position of the maximum, as needed after a later new maximum, is not cleanly in the tape or in the rest: the rest dominates, the tape contributes, and a mismatch between them breaks the answer.

- evidence: M_pos pairs (maximum swapped with an old far-below input), transplant at t = 6, new maximum 1.3 M at step 7, then two low inputs, n = 262,674: hybrid names the rest's old maximum 43.1%, the tape's old maximum 18.4%, the old runner-up 20.0%, the new maximum 8.1%, other 10.3% (intact 99.5%). New maximum at step 9 after two low inputs: 37.6%, 19.5%, 26.4%, 0.0%, 16.5%. With the tape deleted at t = 6, a new maximum at step 7 is still followed by the correct answer in 72.4%; with the two-input donor tape in 98.1%. For the present answer, exchanging a tape that differs only in the maximum's position leaves the runner-up named in 83.8% at t = 9 and 91.9% at t = 6 (intact 96.4, 96.5%); the tape's maximum is named in 1.8% and 0.9%.
- script: t1_transplants/a4_breakdown.py (a4_breakdown.txt); a3_future.py (a3_future_t6.txt); a1_immediate.py (a1_immediate.txt); a7_minimal_donor.py (a7_minimal_donor.txt)
- caveats: The tape depends on the maximum's position (13 points lost when it mismatches), but no transplant made the answer follow the tape's maximum cleanly. Not split by configuration.

### 7. [moderate] The network does not read the values or positions of inputs far below the runner-up from the tape, although the tape changes with them.

- evidence: low_val (every input more than 0.4 M below S redrawn below that bound): the tape changes by a median 16% of its norm (quartiles 8%, 30%); hybrid accuracy at t = 9 94.2% against 96.4% intact (n = 368,691); at t = 6 96.1 against 96.4%. low_pos (those inputs permuted among their positions): 94.9 against 96.5% at t = 9 (n = 362,718), 96.8 against 97.0% at t = 6. After futures from t = 6 the loss is at most 0.6 points (new maximum at step 7: 99.0 against 99.5%; at step 9: 97.9 against 98.5%; new runner-up at step 7: 99.9 against 100.0%; new third: 99.2 against 99.5%). Ladder of donors at t = 9 (n = 369,102): unrelated with the same runner-up position by chance 81.2%, unrelated with the same positions of M and S 85.5%, top two kept and the others redrawn 90.2%, top three kept 92.4%, top four kept 94.0%, far-below redrawn 94.1%, identical 96.3%. The same ladder at t = 6 with three low inputs: 83.7, 93.1, 96.6, 96.8, 97.2, 97.2, 97.3%.
- script: t1_transplants/a1_immediate.py (a1_immediate.txt); a3_future.py (a3_future_t6.txt); a6_ladder.py (a6_ladder.txt)
- caveats: At t = 9 the loss is 1.5-2.2 points, not zero, and it grows to 6 points when everything below the runner-up is redrawn. The far-below inputs enter both the tape and the rest and largely cancel in the full scores, so a real tape is not exactly a function of the top few inputs. At t = 8 a new maximum at step 9 costs 1.5 points (96.3 against 97.8%).

### 8. [strong] No single tape mode carries the runner-up's position, and the code is graded: a half dose of the donor's tape yields neither position.

- evidence: S_pos pairs at t = 9, n = 245,932; P(answer = A's runner-up) / P(answer = B's runner-up). Only one mode from B: alt 79.6 / 0.2%, short 31.5 / 12.2%, h2 42.9 / 3.0%, h3 45.1 / 6.2%. All modes but one from B: all but alt 0.7 / 70.7%, all but short 13.5 / 30.8%, all but h2 3.7 / 38.8%, all but h3 7.1 / 42.2%. At t = 6 (n = 174,548): all but alt 0.4 / 81.7%; only h3 32.9 / 14.4%. Dose, tape coordinates (1-q) A + q B, t = 9: q = 0.2: 71.0 / 0.4%; 0.35: 41.7 / 7.2%; 0.5: 22.3 / 18.6%; 0.65: 9.6 / 36.5%; 0.8: 1.4 / 63.4%; 1.0: 0.4 / 85.3%. At q = 0.5 (n = 368,648) the answer is the maximum in 35.3% and the third in 20.7%. For the role swap (T_up_hi, n = 23,561) the switch comes late: q = 0.65: 85.1 / 3.6%; 0.8: 52.9 / 33.6%; 1.0: 8.4 / 87.0%.
- script: t1_transplants/a2_modes_dose.py (a2_modes_dose.txt); a6_ladder.py (a6_ladder.txt)
- caveats: Single-mode hybrids have a negative entry in 36-76% of cases; clipped values differ by at most 7 points and do not change the picture. alt matters least but is not irrelevant (85.3% with it, 70.7% without).

### 9. [moderate] The hybrids leave the reachable set often but only slightly, and the conclusions do not depend on it.

- evidence: Share of hybrids with a negative entry: 26-47% at t = 9 and 46-75% at t = 6 across pair kinds. S_pos at t = 9 (n = 368,648): tape's runner-up named in 87.1% of the 227,558 hybrids without a negative entry and in 82.7% of those with one; the most negative entry has median -0.84 M among the latter, against an rms activation of 10.3 M in intact states. Clipping the hybrid at zero changes the matched-pair rates by at most 4.5 points (S_pos 85.4 to 81.9% at t = 9, 93.9 to 93.2% at t = 6; role swap 86.9 to 82.4%); for unrelated donors at t = 6 by 8 points (89.8 to 82.0%).
- script: t1_transplants/a6_ladder.py (a6_ladder.txt); a1_immediate.py (a1_immediate.txt)
- caveats: Non-negativity is necessary, not sufficient, for a state to be reachable; I did not test reachability in any stronger sense.

## Did not work

- Single-mode transplants (only alt, short, h2 or h3 from the donor) never moved the answer to the donor's runner-up (0.2-12% at t = 9), so the runner-up's position cannot be assigned to one mode.
- I expected the rest to veto its own maximum's position; it does not (the hybrid names the tape's runner-up in 83.0% of the 46,158 pairs where it sits on the rest's maximum).
- No transplant made the answer follow the tape's maximum position: after a later new maximum the hybrid is right for the rest's owner in 43%, for the tape's owner in 18% and wrong otherwise.
- Adding the real third or the real top four to the artificial two-input donor did not improve the hybrid at t = 9 (93.1% and 93.2% against 94.7% with M and S only).
- A two-age tape with a fixed ratio S/M = 0.6 matches the intact network for the present answer (96.3%) but fails after a later new maximum (91.5% against 99.5%) and on A's own remaining inputs (92.6% against 96.1%).
- Transplants from donors whose runner-up is 2 steps old and whose maximum is 8 or 9 steps old are not followed (21.0% and 3.8%); I did not find out why.
- The T_pos, T_up_lo and T_up_hi pairs have no valid 'new third' future (the third is already adjacent to the runner-up), and T_pos has no valid states at all at t = 4.

## Open

- How S/M is laid out in the seven tape coordinates (bump height, or a relation between modes) was not measured; the transplants only show that the hybrid tolerates ratios 0.4-0.8 for the present answer and needs the true ratio after a new maximum.
- Why the tape carries 0.24 of the runner-up's value in the later comparison, and whether that weight depends on a_M, a_S or on how long after the transplant the input arrives (it looks larger when the input arrives at once at t = 9).
- The handover at young ages: at a_S = 2 the runner-up's score is mostly in the rest, at a_S = 3-4 it is shared, from a_S = 5 it is in the tape only. Which part of the rest holds it (pulse units, trans mode, clock) and how it moves into the tape was not examined.
- The failure of donors with a_S = 2 and a_M = 8, 9 (the hybrid names the maximum in 93% for a_M = 9) is unexplained.
- After a new maximum the old maximum's position is found in 72% without any tape and in 99.5% with the right one; what the tape adds there, and why a mismatched tape leaves the old runner-up named in 20-26%, is open.
- The real tape depends on inputs below the top two (2-6 points lost at t = 9 when they are redrawn in the donor), while a tape built from M and S alone works to within 1.6 points; how the rest compensates for the missing lower inputs was not analysed.
- Chosen futures used only negative low inputs (-0.05 M to -0.6 M) and A's own inputs; small positive low inputs were not tested separately. Immediate tests were run at t = 9 and 6, futures from t = 4, 6 and 8; other steps were not checked.
- tape_v2.md has not been written by this investigator; these findings are returned for whoever assembles it.

## Files

- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/tp.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a1_immediate.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a1_immediate.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a2_modes_dose.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a2_modes_dose.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a3_future.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a3_future_t4.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a3_future_t6.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a3_future_t8.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a4_breakdown.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a4_breakdown.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a5_values.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a5_values.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a6_ladder.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a6_ladder.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a7_minimal_donor.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a7_minimal_donor.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a8_two_number_tape.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a8_two_number_tape.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a9_by_configuration.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a9_by_configuration.txt
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a10_weak_cells.py
- c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/t1_transplants/a10_weak_cells.txt
