# Extra figures: results of the makers and of the independent checkers (2026-11-20)

One section per plot script. Each figure has the maker's own checks and the checker's verdict (`ok` = nothing changed, `fixed` = the checker edited the script and redrew, `problem` = an open fault). Dumped from the workflow journal by `hide/dump_plot_results.py`; the text is the agents', unedited.

## plot_two_waves.py

### Maker

**notes:** No plotted number disagrees with the report by more than sampling noise (largest difference: 0.4 in a median, 0.2 in a regression coefficient).

Files (all three figures come from one script):
- c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_two_waves.py (about 5 s for the full run; `--redraw` uses the cache; both print the comparison numbers)
- c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_two_waves.npz
- c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\plots\waves_by_age.png, readout_columns.png, margin_by_distance.png

Three things the figures show that the report's wording does not (none is a numerical disagreement):
1. waves_by_age, panel c: per (age, index) cell the largest residual in s is 10.1 (age 6, index 3, the maximum's own index). The report's "cancel to within 8" refers to the values averaged over ages by distance from the maximum (here −7.8 and −7.9). The residual is negative and sits on the maximum's index and the one after it.
2. margin_by_distance, panel b: "hardly with the gap" holds for relative gaps above 0.15 or D ≤ 4. For D = 5..8 the smallest bin (0.05–0.15) has a median margin of 12.0, 9.2, 9.2, 10.5, against 13–22 in the other three bins. e16's own output shows the same (D = 6: 9.2 against 16.3). I titled the panel "depends on the distance more than on the gap".
3. D = 7 and 8 involve position 8 (always for D = 8, in half the pairs for D = 7), which is outside the indices 0–7 where the wave holds. The V part dips at D = 7 and rises again at D = 8.

Also visible: in readout_columns (left) the points at i = 8, 9 lie well off the extrapolated curves (drawn thin), in line with verifier V2's remark that the fit holds on 0..7 only. The frequency is fixed at the clock's 42.75° and not fitted; the legend gives amplitude, phase, offset and R² per unit.

Design decisions the checker may question:
- Every heatmap cell carries its value, because a static PNG has no hover or table view.
- Blue = V / comparators' part, orange = m / wave units' part, black = sum in readout_columns (right) and margin_by_distance, as in panel c of mechanism_summary.png. The four comparators keep the colours of its panel b.
- The gap bins in margin_by_distance panel b use a grey ordinal ramp and not RAMP, because blue already means V in that figure.
- The two darkest lines in that panel end at the same point (22.0 and 22.1), so their end labels are pushed 1 unit apart.
- Palette run through the dataviz validator: the four categorical colours pass (aqua and yellow are below 3:1 contrast, covered by direct labels plus legend); the grey ramp passes as ordinal.
- Each figure has a two-line subtitle with the definitions and the selection, so it can be read without the report.

Housekeeping: no __pycache__ was written and nothing outside the allowed files was touched; `reverse-engineered-networks/` is unchanged. The session scratchpad is shared between the parallel agents: a helper I had named crop.py there was replaced by another agent's file of the same name (or I replaced theirs first), so I moved my temporary files to scratchpad\two_waves\. If another agent reports an odd crop error around 13:05–13:15, that is the cause.

Views I did not make but could: the two clock readouts as points in the complex plane per index (shows the 175–180° opposition directly); the per-sequence cancellation |Z_m + Z_V|/|Z_V| (median 0.36 in the report); P(margin > 0) by D and gap bin.

#### `waves_by_age.png`

*caption:* Coefficient of the gap M−S in V, in m and in their sum at t=9, by age of the maximum and logit index. V has a trough on the maximum's index (outlined), m a crest; the sum is at most about 10 in size.

*checks:* All from 1M N(0,1) sequences, seed 3415. (1) Mean over the eight outlined cells (distance 0 from the maximum's index): V −62.3, m +54.5, sum −7.8; report table of section 3.4: −62, +55, −8. (2) Distance −4: V +81.9, m −79.6, sum +2.3; report: +82, −80, +2. (3) Distance +4: V +59.7, m −59.4; report: +60, −59. (4) Cosine fitted to the 15 distance-averaged values: V amplitude 66.2 with trough at +0.09, m amplitude 62.6 with crest at +0.07; report: 66.2 at +0.09, and 62.6 with crest on the maximum. (5) All 15 distance-averaged values of V, m and sum agree with the report's table after rounding, except the sum at distance +7 (a single cell): −1.4 against −2 (e15 has −1.6). (6) Run with e15's seed (1515, 500k), the script reproduces every V coefficient in hide/outputs/e15_two_score_vectors.txt to the printed digit (age 2: −32.6, 17.6, 64.6, 84.7, 48.9, −10.0, −45.9, −60.9).

#### `readout_columns.png`

*caption:* Left: readout weights U[i,j] of comparators 1, 6, 7, 8 with sinusoids at 42.75° per index fitted on i≤7 (R² 0.94–0.99). Right: amplitude of the clock pattern in each logit through comparators, through wave units, and their sum: 5.5% remains.

*checks:* Weights only, no sampling. (1) Fitted amplitudes u = 5.81, 5.09, 6.21, 6.08; report: the same. (2) Phases −29.3°, 25.7°, 130.9°, −130.2°; report: −29°, 26°, 131°, −130°. (3) R² = 0.974, 0.943, 0.991, 0.988; report: the same. (4) Clock readout sizes on indices 0–7: comparators 5.56–8.42, wave units 5.70–9.54, sum 0.37–1.63; report section 4.3: 5.6–8.4, 5.7–9.5, 0.4–1.6. (5) Phase difference of the two parts 175.5°–179.9°; report: 175°–180°. (6) Ratio |sum|/(|comparators| + |wave units|) = 0.0550 on indices 0–7 and 0.0575 on all ten; report: 5.5% and 5.75%. (7) All ten complex values per part agree with hide/outputs/e07_clock_readout_split.txt (e.g. index 0: 6.59 at −161°, 8.18 at +17°, sum 1.63 at +6°).

#### `margin_by_distance.png`

*caption:* Median margin of the runner-up over the maximum in units of M, with its V part and m part (bands: quartiles), against the distance D between the two, for four relative-gap bins. Large opposite parts leave a margin of 4–22 that grows with D.

*checks:* 1M N(0,1) sequences, seed 3415; medians as V part / m part / margin. (1) D = 3, gap bins 0.05–0.15, 0.15–0.30, 0.30–0.50, 0.50–0.80: 20.4 / −12.9 / 7.7, 32.7 / −24.0 / 8.8, 49.0 / −39.4 / 10.2, 69.0 / −59.1 / 10.4; report table of section 3.4: 20.3 / −12.8 / 7.8, 32.8 / −24.1 / 8.8, 49.0 / −39.3 / 10.2, 68.7 / −58.7 / 10.5. (2) D = 4, gap 0.30–0.50: 61.8 / −47.9 / 13.7; report: 61.7 / −47.7 / 13.7. (3) D = 1, gap 0.30–0.50: 11.4 / −6.2 / 4.8; report: the same. (4) Margin for D = 1..8 at gap 0.30–0.50: 4.8, 7.8, 10.2, 13.7, 16.8, 16.3, 17.9, 22.0; report: 4.8, 7.8, 10.2, 13.7, 16.8, 16.4, 17.9, 21.9. (5) Run with e16's seed (1616, 800k), the script reproduces hide/outputs/e16_m_scores.txt exactly for D = 1..6, including quartiles and cell counts (D = 3, gap 0.30–0.50: 10.2 [7.7, 11.7], −39.3, 49.0, n = 28160). Cell counts in the figure: 3,603 to 46,676.

### Checker

#### `waves_by_age.png` (ok)

*caption:* Final-step coefficient of the gap in V (comparators), in m (other units) and in their sum, by age of the maximum and logit index. V's trough and m's crest on the maximum cancel.

*remark:* No error found. My own recomputation (different code path) matches every cell to 1e-13 on the same sample and within 0.35 on another. Against the report: mean of the eight outlined cells V -62.3, m +54.5, sum -7.8 (report -62, +55, -8); distance -4: +81.9 / -79.6 / +2.3 (report +82 / -80 / +2); cosine fits 66.2 with trough at +0.09, and 62.6 (report the same). Text only was added: the subtitle defines V and m, and the legend gives the mean of the outlined cells. That reconciles the title's 'at most 10' (largest single cell: age 6, index 3) with the report's 'within 8' (averages over ages). Script for all three figures: c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_two_waves.py (about 5 s; PNGs in ...\mechanistic-report\plots\).

#### `readout_columns.png` (fixed)

*caption:* Left: readout weights of the four comparators with sinusoids at the clock's frequency fitted on indices 0–7. Right: size of the clock pattern's readout through comparators, through wave units and summed; 5.5% remains.

*remark:* Plotted values are right: u = 5.81, 5.09, 6.21, 6.08; phases -29.3°, 25.7°, 130.9°, -130.2°; R² = 0.974, 0.943, 0.991, 0.988; clock readout 5.56–8.42 and 5.70–9.54, sum 0.37–1.63; 175.5°–179.9° out of phase; 5.50% and 5.75% (all as in the report and e07). Fixed: (1) the unit of z was undefined; the subtitle now says a pulse of unit 4 adds its size to z, 10.157 per unit of input excess. (2) A gridline ran through the legend text of panel b; legends and notes now have a surface-coloured background. (3) The legend now says the frequency is fixed, and gives the fit with the frequency free: 48.6°, 49.1°, 44.4°, 44.9° per index with R² 0.980, 0.996, 0.992, 0.994 (verifier V2: 48.65, 49.05, 44.35, 44.90). Title a now reads 'close to sinusoids'. The report states the sinusoids without this caveat.

#### `margin_by_distance.png` (fixed)

*caption:* Median margin of the runner-up over the maximum at the last step, in units of M, with its V part and m part, against their distance, for four relative-gap bins. Dashed: without one-step-old inputs.

*remark:* Pooled medians match the report (V part / m part / margin): D = 3: 20.4/-12.9/7.7, 32.7/-24.0/8.8, 49.0/-39.4/10.2, 69.0/-59.1/10.4 (report 20.3/-12.8/7.8, 32.8/-24.1/8.8, 49.0/-39.3/10.2, 68.7/-58.7/10.5); D = 4: 61.8/-47.9/13.7 (61.7/-47.7/13.7); margin for D = 1..8 at gap 0.3–0.5: 4.8, 7.8, 10.2, 13.7, 16.8, 16.3, 17.9, 22.0 (report ...16.4, 17.9, 21.9). Found, not in the report: at large D the pooled medians mix two kinds of pairs. Position 8 (an input one step old) is in 12–33% of the pairs at D ≤ 6, 50% at D = 7 and all at D = 8. Without those pairs the V part at D = 7 is 7–19 (pooled 15–39), 0.22–0.25 of its value at D = 4 where 1 - cos(42.75° D) gives 0.26; the m part is +1 to -9 (pooled -6 to -31); the margin peaks at D = 4–5 (12–18) and falls to 8–13 at D = 7. So section 3.4's 'grows with the distance' up to 22 at D = 8 holds for the pooled pairs only, and the pooled D = 7 median, of a bimodal mixture, moved by up to 5 between two 1M samples. Fixed in the figure: dashed lines show the two parts without position 8, a note in panel b gives the shares and the margin, the subtitle defines the relative gap. The report was not touched.

## plot_interventions.py

### Maker

**notes:** All three figures are drawn and no plotted value disagrees with the report by more than sampling noise.

**Files written**
- One script for all three figures, `hide/plot_interventions.py` (the name `plots/index.md` already lists), with its cache `hide/plot_interventions.npz`.
- A full run takes about 43 s; `--redraw` takes about 1 s.
- Nothing else was edited, and no `__pycache__` was created.

**Things worth knowing**
- **e29 now has a replication of its numbers.** The report flags e29 as a single unverified experiment. This script recomputes it with another seed and 2.5 times the sample, and every number reproduces within 0.001. It reuses the mode decomposition of `modes.py`, so it does not check the decomposition itself.
- **Rank correlation, not in the figure.** Across the nine parts, accuracy without a part has a Spearman correlation of −0.93 with the rms difference it makes between the thresholds, against −0.68 with the rms of its common part. I can add a contrast panel if that is wanted.
- **Transient part kept alone.** Keeping only the transient mode's part gives 0.419, below the 0.551 with no feedback at all. The report's range of 0.42–0.60 includes it without comment.
- **Wave-only removal, printed but not plotted.** Removing only the wave units' share reproduces the whole outcome split of the full removal: new maximum 0.751 and old runner-up 0.169, against 0.776 and 0.165.
- **Wipe at step 8.** The report says every miss answers position 8. My count is 98% of the answers that are not the maximum; the rest are mostly correct answers.

**Additions beyond the task text**
- `intervention_wipe.png`, panel b: a "no wipe" series next to the wiped runs, and a fourth category "another position".
- `intervention_clock.png`, panel b: next to each bar, how much of |z| is left after the change.
- `feedback_by_mode.png`, panel b: a second series, accuracy with only that part kept.
- Each PNG has a footer of two or three lines giving the selection, the intervention and the sample size, so it can be read alone.

**Design deviations**
- The palette validator of the dataviz skill could not be run because `node` is not installed. The colours are the skill's documented default palette, which it lists as already validated: blue, orange, aqua in fixed order, a neutral grey, and the blue sequential ramp.
- Bars have square ends.
- The stacked bars are wider than the skill's cap so that the values fit inside the segments.
- `feedback_by_mode.png`, panel b, has a legend but no direct series labels, because the legend sits directly above the first row.

#### `intervention_wipe.png`

*caption:* Left: P(answer = maximum) after the four comparators are zeroed once, by age k of the maximum and wipe step t₀. Right: outcomes pooled over t₀ ≤ 7, with and without the wipe: the answer flips from runner-up (0.967) to maximum (0.986).

*checks:* 1M N(0,1) sequences, seed 2101 (not e17's); 69.6% selected (report: 70%). (1) P(max) pooled over t0 ≤ 7, k = 1..6: 0.999, 1.000, 0.975, 0.941, 0.992, 1.000; report: the same six values. (2) Pooled over t0 ≤ 7 and all k: maximum 0.986, runner-up 0.002; report: 98.6%, 0.2%. (3) Column t0 = 8, k = 1..6: 0.795, 0.997, 0.419, 0.708, 0.931, 1.000; report: 0.80, 1.00, 0.42, 0.71, 0.93, 1.00. (4) Without the wipe the runner-up is answered in 0.9688 of sequences; report: 96.9%. The grey bar shows 0.967 because it is counted per run (a sequence counts once for each wipe step), like the wipe bars; the footer gives both. (5) Row k = 3, t0 = 3..9: 0.974, 0.981, 0.979, 0.977, 0.964, 0.419, 0.000; verifier V3: 0.973, 0.982, 0.979, 0.976, 0.963, 0.420, 0.000. (6) In-figure notes are computed: at t0 = 8, 98% of the answers that are not the maximum are position 8 (report: every miss answers position 8); at t0 = 9, position 9 is answered in 0.978, 0.998, 1.000 for k = 1, 2, 3 (report: answers position 9 for k ≤ 3). My selection and wipe give identical answers to e17's code on a 200k test sample.

#### `intervention_clock.png`

*caption:* Left: where the answer goes when the clock mode is scaled by f one step before a new maximum (old maximum answered: 0.991 at f = 1, 0.019 at f = 0). Right: removing only the wave units' share gives 0.022, the comparators' share 0.998.

*checks:* 2M N(0,1) sequences, seed 2102 (not e18's); 203,583 selected = 10.2% (report: 10%). (1) P(old maximum answered) for f = 0, 0.25, 0.5, 0.75, 1: 0.0188, 0.0946, 0.3417, 0.8895, 0.9908; report: 1.9%, 9.5%, 34%, 89%, 99.1%. (2) At f = 0: new maximum 0.776, old runner-up 0.165; report: 78%, 16%. (3) Dissociation, in the order unperturbed / clock mode removed / wave units' share removed / comparators' share removed: 0.991, 0.019, 0.022, 0.998; the task's reference values: 0.991, 0.019, 0.022, 0.998. (4) Median |z| after the change relative to before: 0.920 for the wave-only removal, 0.488 for the comparators-only removal; report: z keeps 92% of its size, and loses half of it. The two shares are implemented as in section 2: 2 Re(z r_clock) subtracted on the nine wave units or on units 1, 6, 7, 8 only, and the two changes add up to the full removal exactly. My selection and the f = 0.5 edit give identical answers to e18's code on a 200k test sample.

#### `feedback_by_mode.png`

*caption:* Left: accuracy with one mode's part of the wave units' feedback into the comparators removed, against the rms difference that part makes between the four thresholds. Right: the same accuracies, and accuracy with only that part kept. Larger differences cost more; the level's part most.

*checks:* 1M N(0,1) sequences, seed 2103 (e29 used 400k, seed 2929). Order below: level, transient, runner, short clock, third harmonic, second harmonic, clock, echo, alternating. (1) Accuracy without the part: 0.598, 0.736, 0.712, 0.709, 0.814, 0.877, 0.893, 0.930, 0.946; report: 0.598, 0.736, 0.712, 0.709, 0.813, 0.877, 0.894, 0.930, 0.947. (2) Rms difference between the four thresholds, in M: 0.037, 0.034, 0.029, 0.021, 0.021, 0.013, 0.008, 0.008, 0.004; report: the same nine values. (3) Intact 0.953, all nine parts removed 0.551; report: 0.953, 0.550. (4) Only one part kept: 0.419 to 0.603; report: 0.42–0.60. (5) Printed by the script but not plotted, mean contribution to a threshold in M: +0.125, −0.031, −0.031, −0.005, +0.019, +0.007, +0.010, +0.003, +0.004; report: the same nine values. The differences in the third decimal (0.814/0.813, 0.893/0.894, 0.946/0.947, 0.551/0.550) are at most 0.001, about one standard error of e29's 400k sample.

### Checker

#### `intervention_wipe.png` (fixed)

*caption:* Left: P(answer = maximum) after the comparators are zeroed once, by age k of the maximum and wipe step t₀. Right: pooled over t₀ ≤ 7, the answer flips from runner-up (0.967) to maximum (0.986).

*remark:* Fixed (in-plot note only). Selection (final runner-up > 0.2, gap > 0.2: 69.6%), wipe (h_t0[1,6,7,8] = 0 at t0 = p_M + k) and outcome classes match e17. My own recomputation (other code and seed, 600k sequences) agrees with every cell within 0.003. Against report.md: pooled t0 ≤ 7, k = 1..6: 0.999, 1.000, 0.975, 0.941, 0.992, 1.000 (same); all k: maximum 0.986, runner-up 0.002 (98.6%, 0.2%); column t0 = 8: 0.795, 0.997, 0.419, 0.708, 0.931, 1.000 (0.80, 1.00, 0.42, 0.71, 0.93, 1.00); row k = 3 matches verifier V3 within 0.001. Change: the note now also explains the light cell at k = 9, t0 = 9 (position 9 is answered in 96%; the note covered only k ≤ 3). Left as computed, not bent: at t0 = 8 the script finds 98% of the non-maximum answers at position 8 where the report says every miss (the rest are mostly correct runner-up answers); the no-wipe bar is 0.967 because it is counted per run, and the footer gives the report's 96.9% per sequence. All three figures come from c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_interventions.py (full run 43 s, cache unchanged) and are in ...\mechanistic-report\plots\.

#### `intervention_clock.png` (fixed)

*caption:* Left: where the answer goes when the clock mode is scaled by f one step before a new maximum. Right: P(old maximum answered) is 0.022 without the wave units' share, 0.998 without the comparators' share.

*remark:* Fixed (wording only). Selection equals e18's (re-implemented with running_stats: 10.2% selected). The edit is h[o] − (1 − f)·2Re(z r_clock) on all 13, the nine wave-unit or the four comparator entries, as in section 2; the mode projector equals an independently computed spectral projector to 1e-14. Own recomputation (1.5M sequences, other seed) within 0.003. Against report.md: old maximum answered 0.019, 0.095, 0.342, 0.890, 0.991 for f = 0, 0.25, 0.5, 0.75, 1 (1.9%, 9.5%, 34%, 89%, 99.1%); at f = 0 new maximum 0.776, old runner-up 0.165 (78%, 16%); dissociation 0.991, 0.019, 0.022, 0.998 (same as the reference values); median |z| left 0.92 and 0.49 (92%, half). Changes: panel b's labels use the report's terms (wave units' share, comparators' share, with unit numbers) and mark the |z| ratios as medians; the footer now says the old maximum is at least two steps old at the step before the new maximum. That is what e18 and the script select (t_r − 1 − p_0 ≥ 2). report.md's 'is then at least two steps old' reads as the age at t_r, where it is at least three; the lead may want to reword that sentence in section 3.5.

#### `feedback_by_mode.png` (fixed)

*caption:* Left: accuracy with one mode's part of the wave units' feedback into the comparators removed, against the rms difference it makes between the thresholds. Right: the same accuracies, and those with only that part kept.

*remark:* Fixed (layout only). Same computation as e29: a part is taken out of the comparators' pre-activations at every step; sizes over h_0..h_8 with M > 0.3 and a maximum at least one step old, divided by g_j and by M. Own step-by-step implementation (400k sequences, other seed) within 0.003. Against report.md, in its column order (level, transient, runner, short clock, third harmonic, second harmonic, clock, echo, alternating): accuracy without the part 0.598, 0.736, 0.712, 0.709, 0.814, 0.877, 0.893, 0.930, 0.946 (0.598, 0.736, 0.712, 0.709, 0.813, 0.877, 0.894, 0.930, 0.947); rms differences 0.037, 0.034, 0.029, 0.021, 0.021, 0.013, 0.008, 0.008, 0.004 (same); intact 0.953 (same); all parts removed 0.551 (0.550; ten samples of mine give 0.548–0.551); one part kept 0.419–0.603 (0.42–0.60). Changes: panel b now has direct series labels in the first row, with the legend moved to the lower left and the reference-line labels to the top; the tick labels 0.000 and 0.5 no longer overlap at the origin of panel a; the footer says this is a new sample, so last digits can differ from the table in section 4.1. The dataviz palette validator could not be run (node is not installed); the colours are the skill's documented default. Aside: another checker overwrote my temporary crop helper in the shared scratchpad folder chk/; no project file was affected.

## plot_clock_laws.py

### Maker

**notes:** All three figures come from one script, `hide\plot_clock_laws.py` (the name `plots\index.md` already lists), with cache `hide\plot_clock_laws.npz`. A full run takes 28 s; `--redraw` reuses the cache. I wrote nothing else and did not touch `index.md`.

No number disagrees with the report beyond sampling noise. Five things are worth knowing:

1. **Phase lag (clock_laws, panel c).** One step after a new maximum there is no systematic lag: the signed median of φ(a) − arg z is 0.4–2.0° in every gap bin. The median distance there is still 23° for gaps below 0.1, so it is scatter to both sides. The report's "It lags" therefore holds from age 2 on. The lag is larger at ages 2–4 (28°, 15°, 10° in the first three gap bins) than at ages 5–9 (18°, 8.5°, 6.4°). In the cache of `mechanism_summary.png`, the inner ring is 12° behind the outer one only at ages 2–3; it is 7–11° at ages 4–9 and 0° at age 1.

2. **Radius at a tie, age 8.** I get 0.144 (four seeds: 0.142–0.146), where the report says 0.15 (v_V5: 0.147). That is about 2σ and rounds the other way. Likewise α(9) = 0.805 rounds to 0.80 against the report's 0.81.

3. **Level one step after a new maximum.** Over all states the median is 1.01·w₄M; the report's 0.99 is for states with S > 0.05. From age 2 on the level stays at 0.91–0.94 and shows no sag, while unit 2 falls to 0.85. The sag belongs to unit 2, not to the level mode.

4. **The sag is not monotone in age.** Unit 2 is lowest at ages 7–8 (κ₂ = 0.40, 0.37) and recovers at age 9 (0.27). The thresholds of units 4 and 7 do the same (0.61 and 0.55 at ages 6–7, back to 0.81 and 0.83 at age 9). The report gives the range for unit 2 but not the recovery.

5. **Thresholds table.** The report's table is for controlled sequences. The figure shows random-input medians, which differ from the table by up to 0.08; the script recomputes the table itself and matches it.

Additions beyond the task text:
- `clock_laws.png` (a) has an inset with φ(a) minus the 42.75° line.
- `clock_laws.png` (b) also shows α for cells where no runner-up arrived after the maximum, with the eigenvalue decay of 0.951 per step.
- `level_and_unit2.png` has four panels: the requested unit-2 panel is split into maxima 1–4 and 5–9 steps old, and panel d (ratio against S⁺/M by age) shows that the sag is towards the runner-up.
- `thresholds_by_age.png` is a single panel; the quartile bands are narrow enough for that.

Legends sit on an opaque patch of the surface colour (a small wrapper around `fig_mechanism.legend`), so grid lines do not run through their text. The categorical and ordinal palettes pass the dataviz validator.

#### `clock_laws.png`

*caption:* Clock coordinate z on random inputs: (a) phase φ(a) steps 36°, then 43° per step of the maximum's age; (b) α(a) and c(a) shrink with age; (c) arg z lags φ(a) at small gaps; (d) radius follows the gap, leaving 0.14–0.21 w₄M at ties.

*checks:* All computed from 1M sequences (seeds 4101-4104); report values in brackets. φ(a) = 0.5, 36.4, 79.3, 123.2, 166.0, 208.7, 251.0, 294.0, 337.8° [0, 36, 79, 123, 166, 209, 251, 294, 338]. Steps 35.9°, then 42.3-43.9° [36°, then 42.3-43.9°]. Half-range over the runner-up's age 1.2-1.8° for a ≤ 6, 5.2 and 5.6° for a = 8, 9 [±1.3-1.7°, up to ±5.5°]. α(a) = 1.01, 1.03, 0.99, 0.95, 0.92, 0.88, 0.85, 0.82, 0.80 [same, last 0.81; mine is 0.805]. α without a later runner-up falls by 0.949 per step [0.949]. c(a) = 0.87, 0.84, 0.87, 0.86, 0.82, 0.80, 0.77, 0.75, 0.75 [identical]. Median |arg z − φ| over ages for gap ≤ 0.1, 0.1-0.3, 0.3-0.6, > 0.6: 18.1-33.2°, 7.8-15.0°, 3.2-6.5°, 0.6-2.5° [18-33, 8-15, 3-7, 0.6-2.6]; the plotted signed medians are 27.8, 14.8, 10.1° (ages 2-4) and 18.1, 8.5, 6.4° (ages 5-9) in the first three gap bins. Radius at age 1: 0.95, 0.56, 0.32, 0.21 for S/M in (0, 0.1], (0.5, 0.6], (0.8, 0.9], (0.95, 1] [v_V5: 0.947, 0.557, 0.316, 0.213]; at a tie 0.21 at age 1 [0.21] and 0.144 at age 8 [0.15].

#### `thresholds_by_age.png`

*caption:* Threshold positions f = (θ−S)/(M−S) of comparators 1, 6, 7, 8 and unit 4 by age of the maximum: medians and quartile bands (0 = runner-up, 1 = maximum). Shaded: inputs below the maximum that fire unit 4, widest at ages 6–7.

*checks:* Plotted medians (random inputs, S > 0, M − S > 0.2), ages 1..9: unit 4 = 0.98, 0.96, 0.94, 0.85, 0.72, 0.62, 0.61, 0.69, 0.81 [report table: 0.98, 0.97, 0.95, 0.86, 0.74, 0.64, 0.63, 0.72, 0.83]; unit 7 = 1.03, 1.02, 1.01, 0.87, 0.68, 0.55, 0.55, 0.65, 0.83 [1.02, 1.02, 1.01, 0.87, 0.68, 0.56, 0.56, 0.68, 0.85]; unit 1 = 0.84, −0.07, 0.22, 0.80, 1.31, 1.43, 1.14, 0.58, 0.06 [0.81, −0.08, 0.23, 0.81, 1.30, 1.41, 1.12, 0.61, 0.14]; unit 8 peaks at 1.50 at age 8 [1.46]; unit 6 peaks at 1.20 at age 4 [1.19]. The report's table is for 240 controlled sequences; the script recomputes it and reproduces all 45 entries to 0.01. Random medians differ from it by at most 0.08 (unit 1, age 9). They equal the medians behind panel b of mechanism_summary.png (fig_mechanism.npz) within 0.002. Lowest comparator median at ages 5-8: 0.31-0.55 [table: 0.35-0.56]. θ₄/M_old by age: 0.99, 0.98, 0.97, 0.93, 0.88, 0.84, 0.84, 0.88, 0.93 [0.98 young, 0.84 at ages 6-7].

#### `level_and_unit2.png`

*caption:* (a) Level coordinate / w₄ against the previous maximum: slope 0.936, R² 0.993. (b, c) Unit 2 / 17.3 against the same, for maxima 1–4 and 5–9 steps old. (d) Its ratio to the maximum against S⁺/M by age: old maxima sag towards the runner-up.

*checks:* Level/w₄ = 0.9363·max(M_(t−1), 0), centred R² 0.9929 [0.936, 0.993]. Median level/(w₄M): 1.01 one step after a new maximum, then 0.91-0.94 [0.99, then 0.92-0.94]. h_t[2] = 17.31·max(M_(t−1), 0), centred R² 0.974 [17.3, 0.974]. Median h_t[2]/(17.3 M) by age 1..9: 1.02, 1.02, 1.07, 1.05, 0.98, 0.90, 0.85, 0.86, 0.92 [v_V4, ages 1..8: 1.024, 1.018, 1.068, 1.050, 0.979, 0.900, 0.853, 0.860]. Plotted ratio at S ≤ 0 and at S⁺/M > 0.9, ages 5..8: 0.86, 0.71, 0.63, 0.66 and 1.07, 1.04, 1.02, 1.02 [v_V4: identical]. Regression at age 7: 10.4 M + 7.0 S⁺ [10.5 M + 6.9 S⁺]. κ₂ = 0.21, 0.34, 0.40, 0.37 for ages 5-8 [0.21-0.40]. The line 0.63·x in panel c is the age-7 median for states without a positive runner-up [v_V4: 0.629].

### Checker

#### `clock_laws.png` (fixed)

*caption:* Clock coordinate z on random inputs: phase φ(a) of the gap term (36°, then 43° per step), α(a) and c(a), phase lag at small gaps, and radius against S⁺/M (0.14–0.21 w₄M left at ties).

*remark:* Fixes are cosmetic; no plotted value changed. Grid lines ran through the inset's title and tick labels and through the tie annotation in (d). In (b) the light α dots are hidden under the dark ones for a ≤ 4, so I added a note saying so. Recomputed with my own code and other seeds (1M sequences): φ(a) = 0.5, 36.4, 79.3, 123.3, 166.0, 208.7, 251.0, 294.1, 337.9° [report: 0, 36, 79, 123, 166, 209, 251, 294, 338]; c(a) identical to the report; median |arg z − φ| by gap bin 18–33°, 8–15°, 3–7°, 0.6–2.6° [same]; tie radius at age 1 0.213 [0.21]. Three points for the report, none an error in the plot. (1) Panel c: one step after a new maximum there is no systematic lag (signed median 0.4–2.1°, median distance 23° at gaps below 0.1). The report's 'it lags' holds from age 2: 28°, 15°, 10° at ages 2–4 and 18°, 9°, 6° at ages 5–9 in the first three gap bins. In the summary figure's cache the inner ring is 0.2° behind at age 1, 12° at ages 2–3 and 7–11° at ages 4–9. (2) Tie radius at age 8 is 0.1451 ± 0.0004 on 8M sequences. The figure's sample gives 0.144 and prints 0.14; the report says 0.15. (3) α(9) = 0.8047 ± 0.0004 on 8M sequences; the report's 0.81 is 0.005 high.

#### `thresholds_by_age.png` (ok)

*caption:* Threshold positions f = (θ−S)/(M−S) of comparators 1, 6, 7, 8 and unit 4 by age of the maximum: medians, quartiles (0 = runner-up, 1 = maximum). Shaded: unit 4 fires falsely.

*remark:* Unchanged (the regenerated file is byte-identical). Medians and quartiles reproduced with my own code and other seeds within 0.004; the medians equal panel b of mechanism_summary.png within 0.002. The script also recomputes the report's table (240 controlled sequences) and matches all 45 entries to 0.01, so the threshold formulas and the age convention (age at the step the threshold acts) are the report's. The figure shows random-input medians, which differ from that table by up to 0.08 (unit 1, age 9: 0.06 against 0.14); that is the different input set, not an error. Plotted unit 4: 0.98, 0.96, 0.94, 0.85, 0.72, 0.62, 0.61, 0.69, 0.81 [table: 0.98, 0.97, 0.95, 0.86, 0.74, 0.64, 0.63, 0.72, 0.83]. My own check of the shaded zone, same selection: 23–31% of unit 4's firings at ages 5–8 are for inputs below the maximum, 2–5% at ages 1–3.

#### `level_and_unit2.png` (fixed)

*caption:* Level coordinate/w₄ against the previous maximum (slope 0.936, R² 0.993); unit 2/17.3 for maxima 1–4 and 5–9 steps old; its ratio to the maximum against S⁺/M: old maxima sag towards the runner-up.

*remark:* Two fixes. (1) Panel a's line '1.01 · x, median one step after a new maximum' read as a contradiction of the report's 0.99. Both are right: the median is 1.006 over all states and 0.992 among states with a runner-up above 0.05, which is the verifier's selection behind the report's number (1.015 where the old maximum was not positive). The legend now gives both, computed in the script. (2) The colour bars lacked their top tick (10⁶). Recomputed with my own code and other seeds: level slope 0.9363, R² 0.9929 [0.936, 0.993]; unit 2 slope 17.31, R² 0.974 [17.3, 0.974]; unit-2 ratio at age 7 without a positive runner-up 0.628 [v_V4: 0.629], with quartiles 0.620–0.634, so the 0.63·x line in panel c is the sharp lower edge of the cloud; κ₂ = 0.21, 0.34, 0.40, 0.37 at ages 5–8 [0.21–0.40]. Two things the report does not say: the sag is not monotone in age (median ratio 0.85–0.86 at ages 7–8, back to 0.92 at age 9), and the level coordinate itself does not sag (0.91–0.94 from age 2 on).

## plot_modes.py

### Maker

**notes:** All three figures come from one script, `hide/plot_modes.py` (the name `plots/index.md` already lists), with cache `hide/plot_modes.npz`. A full run takes about 11 s, `--redraw` about 1 s, and both print every value listed under checks. Nothing else was written or edited; no `__pycache__` was created.

One number disagrees with the report by more than sampling noise:
- Section 4.3 says the resting pattern holds "in 2% of states at ages 5 and 6". Over all states with a positive maximum it is 4.2% at age 5 and 2.6% at age 6, which is what the heatmap's last row shows (4 and 3). The 2% comes back only under verifier V5's condition S > 0.05 (2.5% and 2.2%), which the report sentence does not state.
- "Never when the maximum is 1 or 2 steps old" also needs a positive maximum. Over all states, including those whose running maximum is not positive, it is 0.8% and 1.2%.

Three last-digit rounding differences, from the weights alone:
- Level r on unit 12 relative to unit 2 is 1.585 (report 1.59).
- The comparator-7 coefficient in the level column of the section 5 table is −0.055 (report −0.06).
- The same coefficient in the runner column is −5.385 (report −5.39).

One addition to the report, not a disagreement: the title of panel a in `modes_spectrum.png` lists the echo as absent from Σ as well as level, runner and clock. The list is computed (no eigenvalue of Σ within 0.06), and Σ's smallest eigenvalue is 0.257.

Choices I made that the task left open:
- The heatmap uses states with a positive maximum so far (90% of states), because the network does not track the age of a non-positive maximum. Cell numbers appear only where a unit departs from the resting pattern in more than 5% of states.
- The clock dials show −arg r in the top row (the phase of z at which the unit peaks) and arg ℓ in the bottom row (the direction in which the unit pushes z). For a normal matrix the two rows would be equal.
- I added a bold figure header and rounded bar ends so the figures match the sibling figures already in `plots/`.
- The right panel of `gate_patterns.png` also lists the ten patterns that cover half of all steps.

Not made, available on request: a third row in `mode_patterns.png` with the products ℓ[u]·r[u], which do not depend on how units are scaled; and the same heatmap over all states.

#### `modes_spectrum.png`

*caption:* Eigenvalues of the resting map A (dots, named) and of the shift matrix Σ (rings): Σ has no level, runner or clock. Bars: rms size of each mode's piece of the final logits and answers changed when it is deleted; the pieces cancel.

*checks:* All recomputed (weights; 400k sequences, seeds 7101-7104, accuracy 0.9527). (1) Eigenvalues of A: level 1.0099, runner 0.9735, clock 0.951∠42.75°, second harmonic 0.937∠83.28°, alternating −0.928, third harmonic 0.837∠124.78°, short clock 0.735∠42.99°, transient 0.231, echo 0.001; identical to the table in section 2. (2) Eigenvalues of Σ: −0.932, 0.925∠82.55°, 0.834∠124.63°, 0.694∠41.23°, 0.845, 0.676, 0.257; section 3.2 gives −0.93, 0.93∠82.6°, 0.83∠124.6°, 0.69∠41.2°, 0.85, 0.68, 0.26. (3) rms of the pieces, level to echo then pulse units: 5.0, 9.5, 12.6, 36.9, 10.7, 28.8, 87.1, 62.7, 5.6, 22.4; section 4.3 gives 5.0, 9.5, 12.6, 36.9, 10.8, 28.7, 87.1, 62.5, 5.6, 22.5. All logits 69.0 (report 69.0). (4) Answers changed: 12.5, 44.6, 7.4, 30.2, 8.1, 31.0, 49.8, 22.5, 14.3, 14.0 %; report 12.4, 44.5, 7.4, 30.4, 8.1, 31.1, 49.8, 22.5, 14.2, 14.0 %. Largest differences 0.2 logits and 0.2 points, on a different sample.

#### `mode_patterns.png`

*caption:* Right eigenvectors r (top) and dual vectors ℓ (bottom) of level, runner and clock on the 13 non-pulse units; clock as amplitude bars with phase dials. The level pattern is positive everywhere; ℓ_level sits on unit 2, ℓ_clock mostly on the comparators.

*checks:* From the weights, in the normalisation of hide/modes.py. (1) Level r relative to its entry on unit 2 (units 0, 1, 2, 3, 6, 7, 8, 10-15): 1.35, 0.51, 1, 0.19, 0.63, 0.79, 0.70, 1.06, 1.14, 1.58, 0.43, 1.23, 0.81; section 4.3 is identical except unit 12, 1.59 (the value is 1.5849). (2) ℓ on comparators 1, 6, 7, 8, written as the update-rule coefficients ℓ[k] g_k / w₄ of the table in section 5: clock 0.800∠−123.9°, 1.308∠159.3°, 1.377∠28.7°, 0.670∠−44.6° (report 0.80∠−124°, 1.31∠159°, 1.38∠29°, 0.67∠−45°); runner +1.409, −0.202, −5.385, +4.098 (report +1.41, −0.20, −5.39, +4.10); level +0.029, +0.069, −0.055, −0.043 (report +0.03, +0.07, −0.06, −0.04). The dials in panel f show these clock phases. (3) |U_o r_clock| / Σ_u |U[:,u] r_u| = 0.031 (section 4.3: 3%). (4) ℓ_level: 0.41 on unit 2, next largest 0.11; |ℓ_clock| is 0.77, 1.21, 1.06, 0.55 on the comparators and at most 0.22 on a wave unit (section 4.3: ℓ_level sits on unit 2, ℓ_clock mostly on the comparators). (5) The nine patterns add up to W[o, 4] exactly. (6) rms |ζ|/w₄ of the final states, quoted in the panels: 1.50, 8.60, 0.74 (verifier V5's note: 1.499, 8.593, 0.740).

#### `gate_patterns.png`

*caption:* Left: share of states in which each unit is on, by age of the maximum (positive maximum), with the resting pattern's share below. Right: cumulative share of steps covered by the most common gate patterns; the resting pattern covers 10%, ten patterns cover half.

*checks:* 2^20 sequences, seeds 7201-7204. (1) Resting pattern: 10.3% of all steps (report 10%), 29.0% at t = 9 (report 29%); by age of a positive maximum 0, 0, 0.006%, 29%, 26%, 4.2%, 2.6%, 54%, 64%, 61% (report: never at ages 1 and 2; 2% at ages 5 and 6, see notes). (2) 1384 distinct patterns (report: about 1400; arm A's sample 1377); 10 cover half (report 10), 74 cover 90% (report 75), 236 cover 99% (arm A 237). (3) Heatmap cells: unit 4 on at age 0 in 99.9% (section 5: fires on 99.9% of positive new maxima); units 5 and 9 on at age 1 in 100.0% and 99.8% (section 5.4: 100%, 99.7%); unit 0 off at age 2 in 99.85%, unit 11 off at age 5 in 91.9%, unit 10 off at age 6 in 95.0%. With section 4.2's conditioning (M > 0.3, split by runner-up age >= 1) I get 98.3-100%, 94.9-99.3% (73.6% when the runner-up is the current input) and 88.1-99.3%; report 98-100%, 95-99% (74%), 88-99%. (4) Computed and printed, not plotted: all four comparators on in 66.1% of steps and off in 21.6% (section 2: 66%, 22%); all four exactly zero in 93.7% of new maxima over a positive old maximum (report 94%).

### Checker

#### `modes_spectrum.png` (fixed)

*caption:* Eigenvalues of the resting map A (dots) and the shift matrix Σ (rings): Σ lacks level, runner, clock and echo. Bars: rms of each mode's piece of the logits; answers changed by deleting it.

*remark:* Recomputed with my own code on other seeds; everything agrees. Eigenvalues of A and Σ match sections 2 and 3.2 exactly. Piece rms 5.0, 9.5, 12.6, 36.9, 10.7, 28.8, 87.1, 62.7, 5.6, 22.4 against the report's 5.0, 9.5, 12.6, 36.9, 10.8, 28.7, 87.1, 62.5, 5.6, 22.5; all logits 69.0; answers changed within 0.2 points. Fixed: the subtitle said 'a dot inside a ring', but the short clock's dot sits on the ring's edge (distance 0.046). It now states the criterion behind the title: Σ has an eigenvalue within 0.06 (matched modes are 0.004-0.046 away, level, runner, clock and echo 0.13-0.26). The title adds the echo to the report's three absent modes; that is computed (Σ's smallest eigenvalue is 0.257). Left as is: the label 62.7 crosses the 'all logits' line behind a masking box, still readable.

#### `mode_patterns.png` (ok)

*caption:* Patterns r (top) and dual vectors ℓ (bottom) of level, runner and clock on the 13 non-pulse units. Level r is positive everywhere; level ℓ sits on unit 2, clock ℓ mostly on the comparators.

*remark:* Picture unchanged; I only added assertions in the script that guard the panel titles. My own eigen-decomposition (left vectors from Aᵀ) reproduces all 78 bar heights and all 26 dial angles. Agrees with the report: level r relative to unit 2 (unit 0: 1.35, unit 14: 1.23; unit 12 is 1.585, report 1.59); ℓ[k]·g_k/w₄ of the section 5 table (clock 0.80∠-124°, 1.31∠159°, 1.38∠29°, 0.67∠-45°; runner +1.41, -0.20, -5.385, +4.10); rms ζ = 1.50, 8.60, 0.74 w₄ (V5: 1.499, 8.593, 0.740). Phase convention checked on x = (1, 0, ...): arg z = 0.1° at t = 1, then 36°, 80°, 123°. Panel e's title (runner ℓ on unit 2 and comparators 7, 8) holds in this scaling of the units only, as the footnote and section 4.3 say.

#### `gate_patterns.png` (fixed)

*caption:* Left: how often each unit is on, by age of the maximum, with the resting pattern's share below. Right: cumulative share of steps covered by the most common gate patterns; ten cover half.

*remark:* Heatmap, age shares and the ten patterns reproduce with my own code (maker's seeds: identical, 1384 patterns; other seeds: within 1 point). Agrees with the report: resting pattern 10.3% of steps, 29% at t = 9; unit 4 on at age 0 in 99.9%; units 5, 9 at age 1 in 100%, 99.8%. Fixed: (1) title b said '74 cover 90%' against the report's 75. This sample's 74 cover 90.005%; 16 other samples of 2²⁰ all need 75, and a pooled 2²⁴ gives 89.98% for 74. Counts now carry their sampling range (10, 74-75, 236-237; ± 2 binomial standard errors, which matches the spread over those samples). (2) Title a: 'locked to' became 'follow', plus the computed statement on the resting pattern; the last row is labelled and explained; cell text colour switches at the luminance crossover. Disagreement with the report, plot not bent: section 4.3 says the resting pattern holds in 2% of states at ages 5 and 6; over states with a positive maximum it is 4.2% and 2.6% (2.5% and 2.2% only under V5's unstated condition S > 0.05). Also in 4.3, not plotted: 'no sequence has more than four consecutive resting steps' is not exact; 233 of 2²² sequences have five to seven.

## plot_errors_units.py

### Maker

**notes:** Files written (nothing else was touched; no __pycache__ anywhere):
- c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_errors_units.py (one script for the three figures, the name used in plots\index.md; `--redraw` reuses the cache; a full run takes about 30 s and prints the comparison values)
- c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_errors_units.npz
- c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\plots\errors_by_arrangement.png, accuracy_by_gaps.png, wave_unit_troughs.png

Disagreements with the report:
1. (2, 1, 9), (6, 5, 9), (7, 6, 9) in the 8–20% bin. The report says 0.23–0.25. I get 0.272, 0.245, 0.218 (±0.003, n ≈ 22,500 each); plain sampling gives 0.277, 0.246, 0.218. Reviewer N3's run behind the report had 0.254, 0.248, 0.230 with n ≈ 3,400, and R2's t3b had 0.269, 0.249, 0.219. The range should read 0.22–0.27. Each single value is within about 2 standard errors of N3's.
2. "Both differences above 0.2 M and a positive third". I get 99.976% (error rate 2.4e-4 ± 0.1e-4 on 1.88M such sequences); the report has 99.97% (2.9e-4 ± 0.25e-4). That is 2σ, so sampling noise, but the figure's footnote prints 99.98%.
Everything else agrees within sampling noise.

Things that differ from what the task text expected:
- wave_unit_troughs.png outlines seven cells, not three. The rule "exactly zero in more than 80% of states" also catches the four comparators at age 0 (97.8–99.4% each). Panel b's title says so. The 95% in that title is for this figure's selection; the report's 94% is for new maxima over a positive old maximum.
- In that figure the wave-unit rows are sorted by the age (1–9) at which the unit's median is lowest, computed from the data: 0, 14, 15, 12, 3, 11, 10, 13, 2. This shows the low point moving through units 0, 15, 12, 11, 10 at ages 2, 3, 4, 5, 6. Every cell carries its median, because each row has its own colour scale.
- accuracy_by_gaps.png gives the error share of each region in a separate bar panel (b), not written on the heatmap. The heatmap has a wider gap at 0.05 and the two bands are named on its axes with the same words as the bars. The fifth class (runner-up not positive) is not on the heatmap, which covers S > 0 only.
- The accuracy heatmap uses seven colour classes (boundaries at 50, 75, 90, 95, 99, 99.9%), not a continuous scale. A linear scale left the 90–99.9% range blank and a log scale made everything dark. Every cell prints its accuracy.
- Sample sizes exceed the 300k–1M guideline for two figures: 8M sorted draws for figure 1 (2.3M network runs; about 10,200 and 22,500 sequences per cell) and 2^23 sequences for figure 2 (the rarest cell holds 618). Figure 3 uses 500k.
- Figure 1 samples exactly from N(0,1) conditioned on the maximum being last: sorted draws, largest at position 9, the rest in random order. It uses T > 0.2 as e22 and the reviewers did, and this is stated in the figure.

Design: style(), SURF, INK, INK2, MUTED, AXIS, BLUE come from fig_mechanism. The heatmaps use the dataviz skill's blue sequential steps 100–700 (validator: monotone, single hue; the lightest step is below 2:1 on the surface, which the palette allows for heatmaps). All three PNGs were opened and corrected until no text overlapped or was clipped.

#### `errors_by_arrangement.png`

*caption:* Maximum last, third above 0.2: P(answer = third largest) by position of runner-up (rows) and third (columns), for (S − T)/S of 3–8% and 8–20%. Errors sit in a few arrangements; (0, 7, 9) still fails in 55% at 8–20%.

*checks:* Plotted value vs report section 7, bins 3–8% and 8–20%:
- (0, 7, 9): 0.63 and 0.55 (report 0.62, 0.55).
- (8, 0, 9): 0.70 and 0.23 (report 0.69, 0.22).
- (7, 8, 9): 0.71 and 0.14 (report 0.73, 0.14).
- (7, 0, 9): 0.27 and 0.21 (report 0.26, 0.21).
- (2, 1, 9), (6, 5, 9), (7, 6, 9) at 8–20%: 0.27, 0.25, 0.22 (report: 0.23–0.25). See notes.
- All 72 arrangements with the maximum last, 8–20%: 0.0398 (reviewer R2's t3b_cells: 0.0395).
The two bins that are not plotted also agree: (0, 7, 9) 0.64 / 0.37, (8, 0, 9) 0.87 / 0.005, (7, 8, 9) 0.96 / 0.000 (report 0.63 / 0.37, 0.86 / 0.005, 0.96 / 0.001).
All differences from the report are within the sampling error of the reviewers' runs (n ≈ 1,500–3,400 per cell; mine 10,200 and 22,500), except the range noted for (2, 1, 9).
Cross-check of the sampling shortcut: 20M plain N(0,1) sequences, keeping those with the maximum last, differ from the figure by at most 0.013 (3–8%) and 0.007 (8–20%) over the 72 cells.

#### `accuracy_by_gaps.png`

*caption:* (a) Accuracy over the relative differences (M − S⁺)/M and (S⁺ − T⁺)/M, positive runner-up only; (b) share of all errors per region; (c) accuracy by number of positive inputs. Errors concentrate below 0.05 M and where the runner-up is not positive.

*checks:* Plotted value vs report section 7 (2^23 sequences here, 2^21 in the report):
- Overall accuracy 0.9532 (report 0.9529 ± 0.0002).
- Share of all errors: runner-up close to third 53.9% (report 54.1%); maximum close to runner-up 15.7% (15.7%); both 4.9% (4.9%); runner-up not positive 17.9% (17.8%). The three 'close' classes together 74.5% (74.6%).
- Accuracy with 0, 1, 2 positive inputs: 0.116, 0.231, 0.978 (report 0.119, 0.231, 0.979).
- Both differences above 0.2 M and a positive third: 99.976% (report 99.97%); printed as 99.98% in the footnote. See notes.
Not in the report, computed here: accuracy 99.53% when both differences are at least 0.05 M (75.0% of inputs); 3, 4, ... 10 positive inputs give 0.969 falling to 0.942.
Cross-check with other seeds and separate code (4 × 2^20 sequences): error shares 17.9%, 53.9%, 15.6%, 5.0%, 7.6%.

#### `wave_unit_troughs.png`

*caption:* Median h[u]/M by age of the maximum (M > 0.3, runner-up at least one step old), each row scaled to its own maximum. Each wave unit has its low point at a fixed age; outlined cells are exactly zero in over 80% of states.

*checks:* Plotted value vs report section 4.2 (and 4.1 for unit 2):
- Exactly zero: unit 0 at age 2 in 99.8% of states (report 98–100%); unit 11 at age 5 in 96.1% (95–99%); unit 10 at age 6 in 95.5% (88–99%). By age of the runner-up my ranges are 98.2–100%, 94.9–99.5%, 88.1–99.2%, and unit 11 drops to 74% when the runner-up is the current input, as the report says.
- Unit 12 at age 4: median 0.24 M and exactly zero in 33% of states, so it is not marked (report: a trough that is not clipped; lab notebook 21–42% pooled over steps).
- Unit 2: 18.0, 17.8, 18.5, 18.2 M at ages 1–4 and 15.0 M at age 7 (report: 17.3 max(M_(t−1), 0) as a fit over all states, sagging for a maximum older than four steps; 10.5 M + 6.9 S⁺ at age 7). The lead's e20 medians are 17.2–18.8 at ages 1–4.
- All four comparators exactly zero when the maximum arrives: 95.2% under this figure's selection (report: 94% of new maxima; reviewer N3: 0.943 for a positive old maximum).
Cross-check with another seed and separate code (400k sequences): medians differ by at most 0.06 M and zero shares by at most 0.002 over the 130 cells.

### Checker

#### `errors_by_arrangement.png` (fixed)

*caption:* Maximum last, third above 0.2: P(answer = third largest) by position of runner-up (rows) and third (columns), for (S − T)/S of 3–8% and 8–20%. A few arrangements carry the errors.

*remark:* Data verified with my own plain sampling (24M sequences, the 2.4M with the maximum last kept): all 72 cells of both panels agree (largest difference 0.019, rms z 1.0). Plot vs report, 3–8% / 8–20%: (0,7,9) 0.63/0.55 (report 0.62/0.55); (8,0,9) 0.70/0.23 (0.69/0.22); (7,8,9) 0.71/0.14 (0.73/0.14); (7,0,9) 0.27/0.21 (0.26/0.21). Fixed, text only: the header now defines the binning variable (S − T)/S and the triple notation before the panel titles use it. Disagreement with the report, plot left as is: (2,1,9), (6,5,9), (7,6,9) at 8–20% are 0.27, 0.25, 0.22 (my check: 0.27, 0.25, 0.21); the report's range 0.23–0.25 is too narrow. For the index caption: the report's 0.21, 0.051, 0.017, 0.004 'with the maximum last' mean the maximum arriving after both candidates at any position (I get 0.203, 0.052, 0.017, 0.004); for position 9 alone the rates are 0.27, 0.10, 0.040, 0.010, which is what the panel titles show. Script (edited, full run 28 s, --redraw works, no __pycache__): c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_errors_units.py

#### `accuracy_by_gaps.png` (fixed)

*caption:* (a) Accuracy over the relative differences (M − S⁺)/M and (S⁺ − T⁺)/M, positive runner-up only; (b) share of all errors per region; (c) accuracy by number of positive inputs.

*remark:* Data verified with my own code on 6 × 2^20 fresh sequences: accuracy 0.9532 (plot 95.32%; report 0.9529 ± 0.0002); error shares 53.9, 15.7, 4.9, 17.9% (plot identical; report 54.1, 15.7, 4.9, 17.8%); accuracy with 0, 1, 2 positive inputs 0.118, 0.230, 0.978 (plot 0.12, 0.23, 0.98; report 0.119, 0.231, 0.979); heatmap orientation and cells agree within sampling noise. Disagreement with the report, plot left as is: the footnote gives 99.98% for both differences above 0.2 M with a positive third; the report says 99.97%. The script's count is right for its seeds (460 errors in 1,881,692; I reproduce it exactly), but those seeds are a 2.7σ low draw: 48 fresh seeds give an error rate of (2.78 ± 0.05) × 10⁻⁴, i.e. 99.972%, so the report's digit is the better one. Fixed, text only: the footnote now prints the error count, so the precision of that number is visible. Nothing overlaps or is clipped.

#### `wave_unit_troughs.png` (fixed)

*caption:* Median h[u]/M by age of the maximum (M > 0.3, runner-up at least one step old), each row scaled to its own maximum. Outlined cells are exactly zero in over 80% of states.

*remark:* Data verified with independent bookkeeping (600k sequences): medians within 0.06 M and zero shares within 0.002 in all 130 cells. Plot vs report: zeros 99.8, 96.1, 95.5% (report 98–100, 95–99, 88–99%); unit 12 at age 4 not outlined (zero in 33% pooled, 14.5% of final states; report 14%); unit 2 at 15.0 M at age 7 (report 10.5 M + 6.9 S⁺). Fixed: (1) the panel-a title claimed 'each has its low point at a fixed age'; in my check that holds only for units 0, 12, 11, 10 (the minima of 14, 3, 13, 2 are shallow and move with S/M and step), so it now states the three zeros and the two deep medians; (2) the footnote explains the low age-0 column (units still set by the previous maximum, 0.44 M in the median) and gives the report's 94% beside this selection's 95% (both computed; figure 0.35 in taller). Caveat: unit 15's 0.2 M at age 3 is a pooled median (0 at steps 4–6, 2.6 M at step 9); unit 12's dip holds at every step. The figure contradicts one word of section 4.2: of the 'other troughs that matter', unit 15 at age 7 and unit 13 at age 4 are peaks with positive readout weights (13.2 and 8.3 M at the step of the new maximum; U = +2.62, +2.09). Unrelated to the figures: I accidentally overwrote another checker's temporary helper, C:\Users\linda\AppData\Local\Temp\claude\0c13cd028550ec56\cd99c4bf-b605-4494-8f6c-0ec20b242da2\scratchpad\chk\crop.py (outside the project); its original content is lost and I left a notice in the file.

## plot_worked_example.py

### Maker

**notes:** Both figures are made and no plotted number disagrees with the report.

One script writes both: `c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_worked_example.py` (cache `plot_worked_example.npz` beside it, holding only the fitted phase law). A full run takes about 9 s and `--redraw` about 2 s. Run it with `python -B`; no `__pycache__` was created and nothing outside `hide/plot_worked_example.*` and the two PNGs was written. The file names match the entries already in `plots/index.md`.

Where I departed from the brief:
- **Centring in panels c and d.** Each column is shown minus the mean over the inputs before x_t, not minus the mean of the whole column. A low newest input sits 60 to 100 logits below the rest, so the whole-column mean shifts every older entry by up to 20 logits (step 4) and breaks the row-wise reading. The frames and the M/S marks do not depend on this choice. Three cells beyond the ±50 colour range have their value printed (−102, −86, −63).
- **Panel d, top entry.** The frame is the largest entry among the inputs before x_t, as in the report's table. The newest input's cell is shown but hatched, because over all entries the newest one would be the top of m_t in 7 of 10 columns.
- **Colour of unit 4.** θ₄ is drawn in the dark neutral INK2, not MUTED as in `mechanism_summary.png`. θ₄ runs right beside θ₇ here, and MUTED against AQUA fails the dataviz colour-blind check (distance 5.1 under deuteranopia); INK2 passes.
- **Pulses.** Panel b is three small strips on one scale (one per unit, neutral bars with values) instead of three colours, since no remaining palette hue is clearly distinct from the four comparator colours.

Additions beyond the brief, all computed by the script:
- M and S letters for the true largest and second largest input so far, event labels per step, and a hollow marker for each threshold that lies below the input (that comparator is off).
- A note in panel a: one step after each new max (steps 1, 4, 8), θ₁, θ₇, θ₄ are at the new max and θ₆, θ₈ at the old one.
- In the clock figure, two hollow points for steps 2 and 5 with x_t = −1, with arrows showing what the new runner-up did to z, and a table of radius, phase and φ(a) per step that also serves as the legend.
- The statements in the panel titles are asserted in `check()`.

Two small caveats on numbers:
- "Without the new runner-up" depends slightly on the low value substituted. At step 5 the phase is 37.6° to 39.7° for x₅ between 0 and −2 (38.7° for −1); at step 2 it is 35.9° to 36.3°.
- α(a) from my pooled per-age fit (not plotted) is 1.005, 1.018, 0.983, 0.944, 0.913, 0.883, 0.854, 0.822, 0.799, within 0.012 of section 4.1's per-cell values.

The step-1 column shows the report's first bullet: the true second largest is x₁ = −0.5 (S on position 1), but the largest entry of s₁ is position 0.

Not made, available on request: a third score panel for V_t = s_t − m_t (the comparators' part), and the value printed in every heatmap cell.

#### `worked_example_timeline.png`

*caption:* Section 5.5's sequence step by step. (a) Inputs, the five thresholds they meet, max and runner-up so far. (b) Pulses of units 4, 5, 9. (c) Scores s_t per input; (d) m_t. Framed top entry: the runner-up in c, the earlier maximum in d.

*checks:* Every value is computed from the weights by the script, which prints the full table of section 5.5; every cell agrees with the report to the printed digits.
- Thresholds (all 50 values match). Step 4: θ₄ 1.18, θ₁ 1.16, θ₆ 0.77, θ₇ 1.22, θ₈ 0.83, which is also the 'sum' row of the table in 4.1. Step 2: θ₁ = −0.01 and θ₈ = 0.24 lie below x₂ = 0.3, so comparators 1 and 8 are off. Step 7: θ₄ = 1.14.
- Comparators off per step: all, none, 1 and 8, all, none, 1, none, all, none, none (as in the table).
- Pulses: h[4] = 8.13, 4.40, 4.64 at steps 0, 3, 7; h[5] = 9.91, 4.84, 10.69, 4.82, 10.30 at steps 1, 3, 4, 6, 8; h[9] = 1.31, 2.02, 1.78 at steps 1, 4, 8 (as in the table).
- Tops: s_t gives 0, 0, 2, 0, 0, 5, 5, 3, 3, 3; m_t among the inputs before x_t gives 0, 0, 0, 3, 3, 3, 3, 7, 7 (as in the table). The network answers 3.
- Final logits by position: −39.9, −32.2, −0.8, 28.7, 11.2, 6.3, −0.5, 14.5, 11.1, −26.5, equal to hide/outputs/e28_worked_example.txt.
- Newest low input against the runner-up's entry at steps 4, 6, 8: −114, −100, −90 logits; section 5.1's 61.6·(x_t − S) gives −111, −99, −86.
- The statements in the panel titles and the in-plot note are asserted in the script (check()) and pass.

#### `worked_example_clock.png`

*caption:* Clock coordinate z/w_4 after each step of the section 5.5 sequence, on the phase law φ(a) (rays). One step after each new maximum it restarts near 0°, then turns; hollow points show steps 2 and 5 without the new runner-up.

*checks:* - |z|/w₄ and phase after steps 1 to 9: 0.80 at 0°, 0.58 at 25°, 0.31 at −8°, 0.61 at 9°, 0.44 at 31°, 0.42 at 75°, 0.11 at 47°, 0.47 at 13°, 0.50 at 39°. This is identical to the 'clock after' column of the table in 5.5; the same numbers are printed in the table beside the plot.
- Hollow points (x_t replaced by −1 at the two new-runner-up steps): 0.81 at 36° for step 2 and 0.56 at 39° for step 5. The report says 36° and 39°; reviewer note rev_N2 gives 0.81@36° and 0.56@39°.
- Phase law (rays), fitted in the script on 500k N(0,1) sequences as the phase of the (M − S⁺) coefficient per age of the max: 0.3°, 36.3°, 79.3°, 123.1°, 165.9°, 208.6°, 251.0°, 294.0°, 337.8°. Section 4.1 gives 0, 36, 79, 123, 166, 209, 251, 294, 338. Largest difference 0.4°; three seeds agree within 0.1°.
- One step after the second and third maximum the radius is 0.61 and 0.47, against an excess d = h[4]/w₄ of 0.43 and 0.46, as stated in 5.5.

### Checker

#### `worked_example_timeline.png` (fixed)

*caption:* The sequence of section 5.5 step by step: inputs against the thresholds they meet, pulses of units 4, 5, 9, scores per input. Framed top entry: runner-up in s_t, earlier maximum in m_t.

*remark:* All plotted values are right; I fixed one title claim and three legibility faults. Every plotted array (thresholds, gates, pulses, s_t and m_t by position, frames, M/S marks) was recomputed from the weight file alone and agrees to 1e-14. Against the table in 5.5: step-4 thresholds 1.18, 1.16, 0.77, 1.22, 0.83; h_3[4] = 4.40, h_4[5] = 10.69, h_4[9] = 2.02; tops of s_t 0,0,2,0,0,5,5,3,3,3 and of m_t 0,0,0,3,3,3,3,7,7. All equal.

Changes, all in c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_worked_example.py:
1. Panel a's title said θ_1, θ_6, θ_8 all swing between runner-up and max. In this sequence θ_8 stays between −0.12 and 0.41 of the way from runner-up to max, because the max is never more than 4 steps old (θ_1: −0.18 to 0.89, θ_6: −0.15 to 1.26). New title: 'θ_4 and θ_7 sit at the max of the earlier inputs, θ_8 near the runner-up; θ_1 and θ_6 swing between the two', plus a one-line note pointing to section 4.1. 'Max so far' became 'max of the earlier inputs', since θ(t) sits at M_(t−1).
2. The input markers covered the threshold markers at both new-runner-up steps (the hollow θ_8 at step 2). Inputs are now drawn 0.17 of a step right of their step, and the legend says so.
3. The second legend was drawn twice and looked bold.
4. Gridlines ran through the event labels, the notes and the pulse values.
Also corrected: an off-by-one in the 'old max' behind the in-plot note (no effect on this sequence), and the note in c now says 'of this sequence'. All title statements are asserted in check().

Kept from the maker: columns of c and d are centred on the mean over the earlier inputs, not on the whole column as the brief asked. It is stated on the colour bar, and the frames and M/S marks do not depend on it.

#### `worked_example_clock.png` (fixed)

*caption:* Clock coordinate z/w_4 after each step of section 5.5's sequence, with the phase law φ(a) as rays. It restarts near 0° one step after a new maximum; hollow points omit the new runner-up.

*remark:* Only the title changed; every plotted value is right. The title now reads 'One step after a new max the clock restarts near 0°, then turns; a new runner-up shortens it and holds it back'. At the step of a new max itself the clock is at −8° and 47° with a small radius, so 'after each new max' was imprecise. The title statements are asserted in check().

Recomputed independently from the weight file:
- Radius and phase after steps 1–9 equal the 'clock after' column of 5.5: 0.80 at 0°, 0.58 at 25°, 0.31 at −8°, 0.61 at 9°, 0.44 at 31°, 0.42 at 75°, 0.11 at 47°, 0.47 at 13°, 0.50 at 39°.
- Hollow points: 0.81 at 36° and 0.56 at 39°, as in 5.5 and rev_N2.
- Rays: 0.3, 36.3, 79.3, 123.1, 165.9, 208.6, 251.0, 294.0, 337.8°, against 0, 36, 79, 123, 166, 209, 251, 294, 338 in 4.1. My own fit with another seed is within 0.04°.

One caveat: the hollow points move by about 1° with the low value substituted for the runner-up (35.9–37.0° at step 2 and 37.6–39.7° at step 5, for values from 0 to −2). The figure states the value used, −1.

## plot_update_rule.py

### Maker

**notes:** One script makes all three figures: hide\plot_update_rule.py (about 7 s locally; `--redraw` reuses the 16 kB cache hide\plot_update_rule.npz). I wrote only that script, its cache and the three PNGs in plots\. No .md file, nothing in reverse-engineered-networks, no __pycache__.

Disagreement with the report (weights only, so not sampling noise): two entries of the section 5 coefficient table are rounded twice. "Comparator 7 off" is −0.0548 for the level (report −0.06; correct to two decimals is −0.05) and −5.3848 for the runner (report −5.39; correct is −5.38). hide\rev2_R\rewrite_checks.py printed −0.055 and −5.385, and these were rounded again. The same two numbers are in lab_notebook.md ("Update rule for the mode coordinates"). The figures show −0.055 and −5.38. Every other table entry reproduces to the printed digits. I did not edit report.md or the notebook.

Things I added beyond the brief:
- Figure 1 rescales no arrow. The input (0.011) and unit 5 (0.074) are shown in panel b, a ×9 zoom of the box at the origin of panel a, together with the sum of the four comparator arrows (0.019).
- Panel c draws the four comparator arrows tip to tail (gap 0.019).
- Panel d draws the exact terms of step 2 of the impulse x = (1, 0, …, 0), with the numbers listed beside it: 0.918 ∠ 42.8° (λz₁) + 0.089 ∠ −137.9° (unit 5) + 0.134 ∠ −4.4° (unit 9) + 0.045 ∠ 15.7° (unit 0 clipped) + 0.010 ∠ 142.7° (unit 14 clipped) = 0.964 ∠ 36.4°.
- Panel a has an outer dial 1–9: the phase of z on the impulse at t = 1..9.
- Figure 3 has a grey layer under the blue one: the prediction from the first three terms only, so the reader sees what the two gate terms account for.

Two details of the impulse that the report does not state (consistent with it):
- z₁ = 0.966, not 1. The zero input at step 1 lies above θ₆ = −0.042 and θ₈ = −0.032, so comparators 6 and 8 are off and their term is 0.036 ∠ 173°.
- At step 2 wave unit 14 is clipped as well as unit 0. The report's first-step formula lists unit 0 only and gives 1.00 e^{i36.0°} per unit of unit 4's pulse.
- Of the 6.4° by which the first step falls short of 42.75°, unit 9's pulse gives 6.1° and the clips 0.4°; unit 5's pulse only shortens z (0.918 to 0.829). So the title of panel d names unit 9.

Not made, could be a fourth figure: the rms size of each term on data. The values are already in the cache (rms_terms), in units of w₄, in the order free evolution, input, pulses, comparators off, clips: level 1.065, 0.004, 0.419, 0.008, 0.008; runner 5.44, 0.10, 1.19, 1.03, 0.29; clock 0.646, 0.011, 0.424, 0.232, 0.032.

Design: colour means kind of term in all three figures (pulses blue, comparators orange, input aqua, clipped wave units yellow, free evolution grey). The four hues pass the dataviz palette validator; aqua and yellow are below 3:1 contrast on the surface, so every such mark has a direct label. All three PNGs were opened and checked for overlaps and clipped text.

For the project CLAUDE.md (I did not touch it): mechanistic-report\plots\ holds the figures and hide\plot_update_rule.py makes the three update_rule_*.png.

#### `update_rule_clock.png`

*caption:* Clock row of the update rule in the complex plane, in units of w₄, nothing rescaled. (a) One arrow per term; arc: free rotation λ. (b) Zoom on the small terms. (c) Comparator arrows cancel tip to tail. (d) Impulse: first step is 36.3°.

*checks:* All from the weights (script value vs report section 5 table). λ_clock 0.9509 ∠ 42.75° vs 0.951 e^{i42.75°}. Input 0.0106 ∠ −51.5° vs 0.011 e^{−i52°}. Unit 5: 0.0744 ∠ −137.9° vs 0.074 e^{−i138°}. Unit 9: 0.9122 ∠ −4.4° vs 0.91 e^{−i4°}. Unit 4: 1 (gauge). Comparators 1, 6, 7, 8 off: 0.8001 ∠ −123.9°, 1.3077 ∠ 159.2°, 1.3767 ∠ 28.7°, 0.6699 ∠ −44.6° vs 0.80 e^{−i124°}, 1.31 e^{i159°}, 1.38 e^{i29°}, 0.67 e^{−i45°}. Their sum 0.0193 ∠ −35.5° vs 0.02 (lab notebook 0.02 ∠ −35°). Panel d, impulse x = (1, 0, …, 0), step 2 (section 5.4): z₂/z₁ = 0.9984 ∠ 36.31° vs 0.998 e^{i36.3°}; clip of unit 0: 0.0446 ∠ 15.7° vs 0.045 e^{i15.7°}; unit-5 and unit-9 terms 0.0894 ∠ −137.9° and 0.1341 ∠ −4.4° vs 0.088 e^{−i137.9°} and 0.133 e^{−i4.4°} (the report's are c₅·W[5,4] = 0.0883 and c₉·W[9,4] = 0.1330; on the impulse the two pulses are 1.2% and 0.8% larger). Outer dial (phase of z on the impulse at t = 1..9): 0.1, 36.4, 79.6, 123.4, 166.2, 209.1, 251.8, 294.6, 337.3° vs the report's 0, 36, 79, 123, 166, 209, 251, 294, 338° (all within 0.8°).

#### `update_rule_level_runner.png`

*caption:* Level and runner rows of the update rule as signed bars (units of w₄, separate scales). Unit 4 alone writes the level (others at most 0.13). Comparators 7, 8, 1 and unit 5 write the runner coordinate; the four comparator terms sum to −0.08.

*checks:* All from the weights (script value vs report section 5 table). Level: λ 1.0099 vs 1.0099; input −0.0044 vs −0.004; unit 5 −0.0724 vs −0.072; unit 9 −0.1263 vs −0.126; comparators 1, 6, 7, 8 off +0.0290, +0.0695, −0.0548, −0.0433 vs +0.03, +0.07, −0.06, −0.04; sum +0.0004 vs 0.00. Runner: λ 0.9735 vs 0.9735; input +0.1010 vs +0.101; unit 5 +1.7232 vs +1.72; unit 9 −0.4428 vs −0.44; comparators +1.4092, −0.2024, −5.3848, +4.0985 vs +1.41, −0.20, −5.39, +4.10; sum −0.0795 vs −0.08. Two entries differ in the last printed digit (comparator 7 off: −0.0548 vs −0.06, −5.3848 vs −5.39), see notes.

#### `update_rule_check.png`

*caption:* Check on 2,000,000 one-step updates (200k N(0,1) sequences): coordinate predicted from h(t−1) and x_t against the actual one. Blue, all five terms: exactly on the diagonal, errors below 6 × 10⁻¹⁴. Grey, first three terms only: off by up to 15.

*checks:* Report: the rule holds exactly. Script, seed 20261120, in units of w₄: largest |predicted − actual| is 4.9 × 10⁻¹⁵ (level, values up to 5.1), 5.3 × 10⁻¹⁴ (runner, up to 28.8), 1.7 × 10⁻¹⁴ (clock, complex modulus; Re 1.2 × 10⁻¹⁴, Im 1.3 × 10⁻¹⁴), which is float rounding; all 2,000,000 five-term points fall in diagonal bins. A separate loop-based re-derivation (scratch, not saved) gave 4.5 × 10⁻¹⁵, 5.2 × 10⁻¹⁴, 1.7 × 10⁻¹⁴. Gate statistics of the same sample vs section 2: all four comparators on in 66.1% of updates (report 66%; 50.0% at t = 0 and 81.9% at t = 9 vs 50%, 82%), all four off 21.6% (22%), one to three on 12.3% (12%); resting pattern, 13 units on and pulses off, 10.3% and 29.1% at t = 9 (report 10%, 29%; scratch check). New numbers, not in the report: the first three terms alone are exact on 22.6% of updates (all 13 non-pulse units on) and otherwise off by up to 0.16 (level), 15.2 (runner), 2.4 and 2.8 (Re z, Im z).

### Checker

#### `update_rule_clock.png` (fixed)

*caption:* Update rule, clock row, in the complex plane. Unit 4 writes 1; the four comparator arrows are as large but sum to 0.02; unit 9 holds the first step after a maximum to 36.3°.

*remark:* Fixed in c:\Users\linda\VS Code Workspace\AI interpred models\1-algzoo-second-argmax-432\mechanistic-report\hide\plot_update_rule.py and regenerated; layout and wording only, no plotted number changed. (1) Panel b: the input arrow was drawn broken (its white ring painted over its own shaft); it is now one clean arrow. (2) Panel a: dial number 1 was struck through by the real axis and "|c| = 1" sat on a grid crossing; the labels now hide the lines under them and "|c| = 1" is moved. (3) Panel c: the note now gives the report's condition for the cancellation (all four off by the same xₜ − θₖ). Recomputed from the weights with my own code; all agree with section 5: λ = 0.9509 ∠ 42.75° (report 0.951 e^{i42.75°}); unit 9: 0.912 ∠ −4.4° (0.91 e^{−i4°}); unit 5: 0.074 ∠ −137.9° (0.074 e^{−i138°}); comparators 1, 6, 7, 8 off: 0.800 ∠ −123.9°, 1.308 ∠ 159.2°, 1.377 ∠ 28.7°, 0.670 ∠ −44.6° (0.80, 1.31, 1.38, 0.67 at −124°, 159°, 29°, −45°); their sum 0.019 ∠ −35.5° (0.02); impulse z₂/z₁ = 0.998 ∠ 36.31° (0.998 e^{i36.3°}); dial phases within 0.8° of the report's 0, 36, 79, …, 338°. Panel d's title holds: of the 6.4° by which the first step falls short of 42.75°, unit 9's pulse gives 6.1°, the clips 0.4°, unit 5 none. Three things in panel d go beyond the report without contradicting it: the terms are those of the actual impulse, so unit 5 and unit 9 read 0.089 and 0.134 (report, per unit of unit 4's pulse: 0.088, 0.133); wave unit 14 is clipped as well as unit 0 (0.010 ∠ 142.7°); and z₁ = 0.966, not 1, because comparators 6 and 8 are off at step 1.

#### `update_rule_level_runner.png` (ok)

*caption:* Level and runner rows of the update rule as signed bars (units of w₄, separate scales). Unit 4 alone writes the level; comparators 7, 8, 1 and unit 5 write the runner coordinate.

*remark:* The figure is right, but two entries of the table in report.md section 5 (and of lab_notebook.md) are not. "Comparator 7 off" is −0.0548 for the level (report −0.06; correctly rounded −0.05) and −5.3848 for the runner (report −5.39; correctly −5.38). The report rounded twice: hide\rev2_R\rewrite_checks.py prints −0.055 and −5.385, which I re-ran. The figure shows −0.055 and −5.38; I did not edit report.md. Every other value, recomputed from the weights with my own code, agrees with the report: λ 1.0099 and 0.9735; input −0.0044 and +0.1010 (report −0.004, +0.101); unit 5: −0.0724 and +1.7232 (−0.072, +1.72); unit 9: −0.1263 and −0.4428 (−0.126, −0.44); comparators 1, 6, 8 off: +0.0290, +0.0695, −0.0433 and +1.4092, −0.2024, +4.0985 (+0.03, +0.07, −0.04; +1.41, −0.20, +4.10); sums +0.0004 and −0.0795 (0.00, −0.08). Layout is clean; the PNG is pixel-identical to the maker's.

#### `update_rule_check.png` (ok)

*caption:* Check on 2,000,000 one-step updates (200,000 N(0,1) sequences): predicted against actual coordinate. All five terms (blue) are exact to 6 × 10⁻¹⁴; the first three alone (grey) miss by up to 15 (units of w₄).

*remark:* Unchanged, pixel-identical to the maker's. The prediction uses only h(t−1), xₜ and the weights, the actual value is ℓ·h(t)[o], so the check is not circular; axes are actual (x) against predicted (y). The script reproduces its cache bit for bit (5.6 s for a full run). My own loop-based code with another seed gives largest five-term errors of 4.0 × 10⁻¹⁵ (level), 5.9 × 10⁻¹⁴ (runner) and 1.6 × 10⁻¹⁴ (clock), against 4.9 × 10⁻¹⁵, 5.3 × 10⁻¹⁴ and 1.7 × 10⁻¹⁴ behind the figure: float rounding in both, as the report's "exactly" requires. Gate statistics of the same sample agree with section 2: all four comparators on in 66.1% of steps (report 66%), 50.1% at t = 0 and 81.9% at t = 9 (50%, 82%), all four off 21.6% (22%), one to three on 12.3% (12%), resting pattern 10.3% (10%). Two things have no counterpart in the report: the grey layer (first three terms only, largest errors 0.16, 15.2, 2.4, 2.8) and the 23% of updates with no gate term (I also get 22.6%).
