# Report review N3_sections6to12

## Summary

Checked every number in sections 6–12 of report.md against v_V1–V8, rev_R4, the arm notes (A–I), integration_round1.md and lab_notebook.md, and recomputed six items (scripts and outputs: c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\rev2_N3\n3_checks.py, n3_checks_out.txt, n3_margin.py, n3_margin_out.txt; nothing else was touched, no __pycache__ written).

What holds: the counts of section 9 are right (33 claims = T1–5, S1–6, I1–2, K0–6, M1–6, D1–4, O1–3; 31 CONFIRMED, 2 CORRECTED: S5 and D2). Nearly all numbers of sections 6, 7, 8 and 10 match their sources to the printed digit (last-step formula, interval edges, ablations, the overtaking law of V6, census, shift, repairs, arrangement shares, phase law, level-removal numbers, V4/V5 figures). The numbers repeated from sections 1–5 agree, except where listed.

No major error. Four moderate issues: (1) section 6, the bookkeeping sentence quotes −32 for "the other sources" where arm E has −32 for unit 5 alone and −50 for all other sources; (2) section 6, the margin "3–14 M" contradicts section 3.4's own list (up to 21.9) and omits that the margin grows with the gap for a maximum one step old (I get 3.5–24 by distance and gap for older maxima, 5–28 for a one-step-old one); (3) section 6, the 76% / 24% split of the overtaking law is a split of the D1 records only, which leave out the 20% of overtakings whose old maximum arrived on the step just before; (4) section 7, "the worst arrangements" are only the worst among those 7 or 8 positions apart (V6 tabulated nothing else): (7, 8, 9) is worse in the two smallest gap bins (0.96, 0.73) and (2, 1, 9), (6, 5, 9), (7, 6, 9) match or exceed (8, 0, 9) and (7, 0, 9) in the 8–20% bin.

The rest are minor or nits: numbers attached to a slightly different condition (19% in section 8; "the other ten" in section 10; 84–97% in section 9 against 87–99% in sections 3.4 and 5.4; "at most 2%" in section 8; unit 15 in "6–10 per unit of S"), table rows or sentences that overstate what was replicated (mirror write; "the two arms agree"; section 10 item 16), an age-convention mix in the sag of unit 4's threshold, and small differences between arm and verifier values.

## Issues (24)

### [moderate] section 6

- quote: against the other sources ($-32$)
- problem: Number attached to the wrong selection. In arm E (E3.2, runner-up after the maximum, n = 262,833) the margin over the maximum splits as MARK +67.3, TAPE (unit 5) −32.2, CLIP (inputs + comparator cuts) −14.8, unit 9 −3.0, total +17.3. So −32 is unit 5's writes alone; all other sources together give −50.0. As written, +67 − 32 = +35 does not give the margin.
- suggested fix: "... (unit 4 plus the wave-unit cuts: +67 on average, positive in 99.7% of sequences) against unit 5's writes (−32), the inputs and comparator cuts (−15) and unit 9 (−3); the total is +17."

### [moderate] section 6

- quote: by 3–14 $M$ (medians; less when the two are adjacent)
- problem: Wrong range and a missing condition. The range is the lab notebook's e16 summary for distances 1–5 only (3–5, 4–8, 5–10, 4–14). Section 3.4 itself lists medians of 16.8, 16.4, 17.9, 21.9 for D = 5..8 (V2), and section 3.3 gives 5–14. Recomputed (6M sequences, hide/rev2_N3/n3_margin_out.txt), medians of (s[p_S] − s[p_M])/M per distance and gap bin, gap above 0.05 M, maximum at least two steps old: D = 1: 3.5–4.8; D = 2: 5.7–7.4; D = 3: 7.6–10.0; D = 4: 9.1–14.5; D = 5: 8.9–18.3; D = 6: 7.8–15.9; D = 7: 6.4–15.5; D = 8: 9.8–23.9. Pooled over D by age of the maximum (2–9): 5.4–13.4. "Fixed size, not the gap" also fails for a maximum that is one step old (11% of these sequences): there the median margin is 5, 8, 13, 19, 25 M (D = 1) up to 9, 17, 21, 25, 29 M (D = 8) for gap bins 0.05–0.15, 0.15–0.3, 0.3–0.5, 0.5–0.8, 0.8–1.
- suggested fix: "The runner-up's score exceeds that of a maximum two or more steps old by about 4–5 M when the two are adjacent, 6–14 M at distances 2–4 and 6–24 M at distances 5–8 (medians per distance and gap; 5–13 M by age of the maximum), with no systematic growth with the gap unless the gap is below 0.05 M (section 3.4). A maximum that is only one step old is different: its deficit grows with the gap, from about 5 M to 28 M."

### [moderate] section 6

- quote: If the old runner-up is at least two steps old (76% of cases)
- problem: Missing condition. The 76% and 24%, the 99.5% / 97.2% and the whole law are for V6's D1 records, which require the old maximum to be at least two steps old at the overtaking (g ≥ 1), with M0 > 0.3 and S0 > 0.1. Overtakings whose old maximum arrived on the step just before the new one are excluded; they are 20% of overtakings (V6, D2 composition, share g = 0). For them V6's table gives c = 20.4, 30.7, 53.7, 92.2, 99.7, 101.6, 84.0, 63.2 for an old runner-up 2..9 steps old, with no law. Section 7 then uses one such arrangement, (8, 0, 9), next to "the law of section 6".
- suggested fix: Add after "The comparison is made at the moment of overtaking.": "The numbers below are for overtakings in which the old maximum is at least two steps old (80% of them). If it arrived on the step just before the new maximum, c is 20–102 depending on the old runner-up's age, and I have no law."

### [moderate] section 7

- quote: The worst arrangements, written as (position of runner-up, of third, of maximum)
- problem: Unsourced as "worst". V6 tabulated only the eight arrangements with the two candidates 7 or 8 positions apart; the three listed are the worst of those. V6's own rows for the 8–20% bin already show (6, 5, 9) at 0.250 and (7, 6, 9) at 0.217, as bad as (8, 0, 9) at 0.224 and (7, 0, 9) at 0.214. Recomputed over all 240 arrangements (12M sequences, hide/rev2_N3/n3_checks_out.txt part B; my values for the three listed agree with V6 within 0.03): (7, 8, 9) has 0.96, 0.73, 0.14, 0.001 and is the worst in the two smallest bins; in the first bin (6, 3, 7), (5, 3, 6), (8, 6, 9) have 0.78–0.80, above (0, 7, 9) at 0.64; in the 8–20% bin the order is (0, 7, 9) 0.55, (2, 1, 9) 0.25, (6, 5, 9) 0.25, (8, 0, 9) 0.23, (7, 6, 9) 0.23, (7, 0, 9) 0.21; in the 20–50% bin (0, 7, 9) 0.37, then (2, 1, 9) 0.056 and (6, 5, 9) 0.048. (7, 0, 9) is not among the three worst in any bin.
- suggested fix: "Among the arrangements with the two candidates 7 or 8 positions apart, the worst are (0, 7, 9) ...; (8, 0, 9) ...; (7, 0, 9) .... Others are as bad at small gaps: (7, 8, 9), the third arriving just before a last-step maximum, has 0.96, 0.73, 0.14, 0.001; and (2, 1, 9), (6, 5, 9), (7, 6, 9) have 0.23–0.25 in the 8–20% bin. Only (0, 7, 9) stays above 6% in the 20–50% bin."

### [minor] section 9

- quote: the two arms agree (slopes 50–80)
- problem: Not a faithful summary of arm I. Arm H: regression slopes 77 / 50 / 64–66 by order of arrival (N(0,1): 77.6, 54.0, 63.9, 72.7, 51.8, 67.3; R² 0.36–0.81). Arm I (I4-a, I4-b): exact local slopes with medians 70–79 that do not depend on which of the two came first (78 against 79, 72 against 73, 70 against 71), and a strong dependence on the pair of ages: 62–85 for most pairs, 24–46 when the two are adjacent and old, 100–155 when they are about four positions apart and old. So the arms agree on the typical size near a tie but not on the range or on what the slope depends on. Section 6 ("The slope depends on the order of arrival") and section 10 item 4 give arm H's reading only.
- suggested fix: Row: "arm H: regression slopes 50–78 by order of arrival; arm I: exact local slopes, medians 70–79, by pair of ages 24–155, no dependence on which of the two came first. Same size near a tie; the dependence is not settled." In section 6 add after the three slopes: "(arm H; arm I finds the same typical size, 70–79, but attributes the variation to the pair of ages: 24–46 for adjacent old pairs, 100–155 for old pairs about four positions apart)."

### [minor] section 8

- quote: by 6–10 per unit of $S$
- problem: Wrong for unit 15 and marginal for unit 14. Runner eigenvector times w₄ × 5.2 (recomputed, same as arm D's D-23 and reviewer R1's rev05): −8.7, −6.0, −6.8, −7.9, −6.1 for units 0, 10, 11, 12, 14 and −1.7 for unit 15 (its entry is −0.032 against −0.11 to −0.17 for the others). Measured change of the centre per unit S: −10.4, −5.3, −5.9, −8.7, −5.6, −3.7 for units 0, 10, 11, 12, 14, 15 (mine); R1: −10.6, −5.8, −6.5, −9.3, −4.5, −4.7. Arm D's own detail lists five units (−10, −6.5, −6.8, −8.3, −5.5) and not unit 15.
- suggested fix: "it lowers the centre of units 0, 10, 11, 12, 14 by 5–10 per unit of S (unit 15 by 2–5) and raises unit 2 and the comparators."

### [minor] section 8

- quote: Without this supply accuracy is 19%.
- problem: Number attached to a wider cut than the sentence describes. The bullet is about unit 2's weights into the comparators. V4: cutting those alone gives 27.9%; 19.2% is with unit 2's weight into unit 4 cut as well (unit 4 alone: 60.0%). Section 4.1 states the condition correctly.
- suggested fix: "Without this supply accuracy is 28% (19% if unit 2's weight into unit 4 is cut as well)."

### [minor] section 10

- quote: The other ten add 0.003–0.175 $M$ to each comparator threshold
- problem: The numbers are not for the ten satellites. V4 (K3 and objection 5): 0.003–0.175 M by age is the part from the eight wave units other than unit 2, and 54.5% is the accuracy with those eight units' weights into the comparators and unit 4 removed. Units 5 and 9 are separate (cutting them from the core: 84.6%); cutting all ten gives 50.6%. Section 4.1 has the same wording.
- suggested fix: "The eight wave units other than unit 2 add 0.003–0.175 M to each comparator threshold, depending on the age of the maximum, and the network needs that term (54.5% accuracy without it; 50.6% with the inputs from units 5 and 9 cut as well)."

### [minor] section 9

- quote: Added: comparators all zero in 84–97% only
- problem: Two ranges in the report for the same quantity. Sections 1, 3.4 and 5.4 say 95% overall, 99% at steps 1–3 falling to 87% at step 8 (R4, every positive old maximum). The table row gives V2's 84–97%, which is for an old maximum above 0.3 that is at least two steps old, steps 3–9. Recomputed (2M sequences): every positive old maximum, steps 1..9: 0.988, 0.988, 0.980, 0.958, 0.931, 0.903, 0.884, 0.868, 0.873 (94.3% pooled over steps 1–8); V2's selection, steps 2..9: 0.986, 0.970, 0.938, 0.904, 0.875, 0.857, 0.843, 0.853. Both are right; the selections differ and neither place says so.
- suggested fix: "Added: comparators all zero in 95% of new maxima only (99% falling to 87% by step; 84–97% if the old maximum is at least two steps old)."

### [minor] section 9

- quote: new maximum: gate sequence, pulse sizes, mirror write, unit 0 (5.4)
- problem: The row says "R4: replicated" for all four items. R4 (claim 4) replicated the gate rates, the regressions for h₅ and h₉, the exact products 12.06 and 1.48, the thresholds at s+1 and the cut of unit 0. It did not test the mirror write (168°–189°, arm F only), nor the mode coordinates after step s+2 (0.92 w₄d, 1.04 w₄d at 36°, 4% / 11% / 8% left).
- suggested fix: Outcome column: "R4: gate sequence, pulse sizes, thresholds and the cut of unit 0 replicated. Corrected: the cut of unit 0 fails if the next input is another new maximum. The mirror write and the mode coordinates after s+2 rest on arm F alone. Also arms B, E".

### [minor] section 10

- quote: Their numbers were replicated from scratch by one reviewer (ten claims, none failed)
- problem: Overstates the coverage for sections 6 and 7. Of section 6, R4 replicated only the last-step formula, the interval of x₉ and the two readout-weight ablations; of section 7, the census, the shift and the repairs. Not replicated or verified by anyone, and without a row in the section 9 table (the header of the report says section 9 covers each claim): the bookkeeping by source in section 6 (+67, 99.7%, +26, 88.6% / 45.6% / 11.8%; arm E), the takeover offsets in section 7 (quartiles of ε*, 70%, r = 0.91; arm I), the threshold-rotation test in section 8 (0.030 against 0.331; arm C) and the mode fits for a late runner-up in section 8 (median R² ≥ 0.998; arm H).
- suggested fix: "Sections 5.2 and 5.4, the last-step part of section 6 and the census, shift and repairs of section 7 come from one arm each; their numbers were replicated from scratch by one reviewer (ten claims, none failed). The bookkeeping by source (section 6, arm E), the takeover offsets ε* (section 7, arm I), the threshold-rotation test (section 8, arm C) and the mode fits for a late runner-up (section 8, arm H) rest on one arm each and were not replicated." Add rows for these four to the table.

### [minor] section 8

- quote: These cuts change the clock coordinate by at most 2%.
- problem: Missing condition. Arm D (D-20, D-21): the grazing cuts cost 1.8% over seven steps on the impulse; removing the wave units' ReLUs from age 3 on changes |z| by 0.9–1.7% for S/M up to 0.3, but by 3.9%, 8.6% and 17% for S/M of 0.3–0.6, 0.6–0.9 and 0.9–1 (upper bounds that include the cuts after a new runner-up). The test starts after age 2, so unit 0's deep cut at age 2, named in the same sentence, is not covered by it.
- suggested fix: "The grazing cuts of units 11 and 10 change the clock coordinate by 1–2% when the runner-up is small (S/M up to 0.3); for larger S/M removing the wave units' ReLUs changes it by 4–17%, which includes the cuts after a new runner-up."

### [minor] section 6

- quote: changes $K$ by less than 3%
- problem: True but selective. V6's same test also scaled the short clock (K = 68.0, 63.4, 57.6, 51.4 for factors 0, 0.5, 1, 1.5: +18% to −11%), the runner mode (61.4, 59.9, 57.6, 53.3) and the level (12.7, 38.8, 57.6, 38.9). A reader will take the sentence to mean no other mode matters. Also, K = 57.6 at factor 1 is a third value next to 65.7 and 42.5 in the previous bullet: it is the single-amplitude fit over both orders on V6's intervention sample, which the text does not say.
- suggested fix: "Scaling that mode by 0, 0.5, 1, 1.5 one step earlier gives K = −7.8, 27.6, 57.6, 73.2 (one amplitude for both orders). Scaling the second harmonic, the third harmonic or the alternating mode changes K by less than 3%; the short clock changes it by 10–18% and the runner mode by 4–7%. Scaling the level also lowers it (12.7 at factor 0), because the level holds the thresholds."

### [minor] section 11

- quote: slope 61 for the newest input by three routes
- problem: Left over from before the verification. The notebook's three routes were: the comparators for the newest input (58.25, which V1 says is not the slope of the score difference), unit 4 (−50.7, a fall for a new maximum) and unit 5 for the input one step old (61.2). Only the first concerns the newest input, and the corrected section 3.3 no longer speaks of three routes.
- suggested fix: "slope 61 for the newest input (61.0–62.1 from the weights, 61.9 measured, unchanged under six other input distributions) and the same 61.2 per unit for the input before last through unit 5".

### [minor] section 7

- quote: with $\kappa$ up to 0.26 when the maximum is 5–6 steps old, because 16% of the threshold rides on the clock
- problem: Age convention and wording. The κ list is arm B's, indexed by the age of the maximum one step before the input arrives. In the convention of the table in section 4.1 and of section 5.3 (age when the input arrives) I get κ = 0.003, 0.002, 0.023, 0.100, 0.197, 0.256, 0.254, 0.201, 0.124 for ages 1..9, false-firing rates of new runners-up 1.1, 2.3, 3.5, 10.0, 19.3, 28.0, 29.1, 22.3, 12.8%, and median θ₄/M 0.988, 0.978, 0.969, 0.929, 0.880, 0.842, 0.842, 0.879, 0.928 (n3_checks_out.txt part D). So the sag and the false firings peak at ages 6–7, as the unit-4 row of the table in 4.1 shows (0.64, 0.63); "19–29%" is the range over ages 5–7. Section 4.1's "sags to 0.84 when it is 5–6 steps old" has the same shift. Also, the law is in M − S⁺ (arm B), and "16% of the threshold" is not what arm B says: θ₄'s slow part is 0.82 M + 0.16 S⁺ and its clock term has amplitude 0.16 times the clock radius, negative at these ages.
- suggested fix: "M − θ₄ = ε(a) M + κ(a)(M − S⁺) with κ = 0.20, 0.26, 0.25 when the maximum is 5, 6, 7 steps old as the input arrives, because the slow part of θ₄ is only 0.82 M + 0.16 S⁺ and its clock term (0.16 of the clock radius) is negative at those ages. Unit 4 then fires for 19%, 28%, 29% of new runners-up."

### [nit] section 7

- quote: raises accuracy to 0.9588
- problem: Two values in the report for the same intervention: section 5.2 says "from 95.30% to 95.89%" (arm H, 1.2M sequences), section 7 says 0.9588 from a baseline of 0.9529 (arm I, 2²¹). R4 gets 0.9590 and 0.9589 on two samples. Sampling noise, but the two sections differ in the last digit and in the baseline.
- suggested fix: Use one sample in both places, for example "from 95.29% to 95.88%" in section 5.2, or write 0.959 in both.

### [nit] section 6

- quote: (slope 1.013 against the true one
- problem: The arm's values are kept although the replication differs: R4 gets slope 1.008 and R² 0.986 for the lower edge (arm E: 1.013, 0.990), and notes that this R² depends on the tail. The upper edge agrees (0.998, 0.996).
- suggested fix: "(slope 1.01 against the true one, R² = 0.99)".

### [nit] section 9

- quote: all 33 claims reproduced (31 as stated; 2 were too wide in scope)
- problem: The counts are right (31 CONFIRMED, 2 CORRECTED). "Too wide in scope" fits S5 (the bound 0.997 holds only up to a relative gap of 0.8). D2 is a different case: V6 marks it CORRECTED because its numbers match only if the stated condition g ≥ 1 is dropped; under the stated selection they are 8%, 11%, 16% lower at k = 3, 4, 5, and both V6 and V8 found its reading as a decay to be a cohort artefact. D2 no longer appears in the report.
- suggested fix: "all 33 claims reproduced (31 as stated; one bound held only up to a relative gap of 0.8, and one set of numbers held for a wider selection than the one stated)".

### [nit] section 10

- quote: 38–68% for rotations, 98–100% for enlargements
- problem: V3: 98.4% and 99.6% are for factors 1.5 and 2; at factor 1.25 it is 95.3% (V3's summary says 95–100%). For rotations V3 has 38–68% and V8 33–68%. "Changed by at most 0.05" holds for the raw rates (V8); the conditional one-step rate moves from 0.404 to 0.460 when clipped (V3).
- suggested fix: "33–68% for rotations, 95–100% for enlargements (98–100% from a factor of 1.5). The reductions were unchanged with the state clipped at zero and the rotations changed by at most 0.06."

### [nit] section 8

- quote: $S + 1.45\,(M - S)$
- problem: The table in section 4.1 that the sentence cites runs from −0.11 (unit 6, age 1) to 1.46 (unit 8, age 8). 1.45 is arm H's median; the table is V7's mean.
- suggested fix: "the thresholds lie between about S − 0.1 (M − S) and S + 1.46 (M − S)".

### [nit] section 10

- quote: the shift experiment
- problem: Only partly sourced for uniform inputs: arm I ran one shifted uniform distribution, Unif(0,1), with accuracy 0.892 (i02b), not the scan of c. Computed (2²⁰ sequences each, n3_checks_out.txt part E): Unif(−1,1) + c gives 0.120, 0.763, 0.906, 0.938, 0.934, 0.921, 0.892, 0.830 for c = −1, −0.5, −0.25, 0, 0.25, 0.5, 1, 2. So the outcome is the same as for Gaussian inputs, and c = 1 reproduces arm I's 0.892.
- suggested fix: Keep the sentence; if a number is wanted: "(Unif(−1,1) + c: 0.120, 0.763, 0.938, 0.921, 0.892 for c = −1, −0.5, 0, 0.5, 1)".

### [nit] section 10

- quote: and nothing for unit 3
- problem: Section 4.2 does report something for unit 3: letting it go negative costs 1.0 point of accuracy (arm A: 0.9529 to 0.9426). Arm A also found no simple gate rule for units 3 and 15 (best lookup 0.84, 0.83). For unit 13 the forcing test (age 4: 0.672, V7) is not in section 4.2.
- suggested fix: "For units 12, 13, 14, 15 I have forcing tests at single ages and no account; for unit 3 only the cost of removing its ReLU (1.0 point)."

### [nit] section 8

- quote: from age 3 on, the short clock holds the largest share of that record
- problem: Arm H's table (H5b, contribution to the runner-up's logit per unit S for a runner-up 2..8 steps old): short 9.3, 23.6, 19.5, 14.2, 10.6, 15.2, 20.2; second harmonic 10.6, 9.8, 5.5, 11.4, 13.5, 9.4, 15.3; third harmonic 27.5, 18.5, 17.8, 10.3, 6.7, 5.9, 3.9. The short clock is largest at ages 3, 4, 5, 7, 8 but not at age 6 (second harmonic 13.5 against 10.6). It is a share of the logit, for one controlled condition (S/M = 0.5, background −1).
- suggested fix: "from age 3 on, the short clock gives the largest share of that record's logit at most ages (in a controlled example; at age 6 the second harmonic is larger)".

### [nit] section 10

- quote: units 0 and 14 push
- problem: Ambiguous. V4: unit 0 lowers both θ₁ and θ₆ (−0.0102, −0.0084 per unit of activation) and unit 14 raises both (+0.0075, +0.0078). The sentence can be read as θ₁ and θ₆ moving in opposite directions.
- suggested fix: "unit 0 lowers θ₁ and θ₆ and unit 14 raises them".

## Files
