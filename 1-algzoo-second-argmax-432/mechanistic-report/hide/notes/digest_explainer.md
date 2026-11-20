# Digest of `explainer.html` for the second-argmax RNN $M_{16,10}$

Sub-agent note for the mechanistic report, written 2026-11-19.

## 0. Sources, method and summary

**Files compared (both read-only).**

- Explainer: `c:/Users/linda/VS Code Workspace/AI interpred models/reverse-engineered-networks/1-algzoo-second-argmax-432/explainer.html`. It has 761,893 characters, of which 733,318 are five base64 PNGs.
- Paper source: `c:/Users/linda/VS Code Workspace/AI interpred models/reverse-engineered-networks/1-algzoo-second-argmax-432/paper.typ`.

**Method.**

1. A script decoded the five embedded PNGs and stripped the tags. I read the full text and viewed every image.
2. I compared the SHA-256 of each PNG with `figures/*.png` and diffed the inline SVG against `figures/algorithm.svg`.
3. I compared the explainer with `paper.typ` by reading both in full, by a diff of all numeric tokens, and by matching each explainer sentence to its closest sentence in `paper.typ`.
4. I re-ran the worked example on the real weights and in the coordinates of `code/exact_rewrite.py`. Exact figure values come from `code/analysis.json`, `code/readout_stats.json` and `code/figures.py`.

**Files I wrote.**

- This digest: `c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/notes/digest_explainer.md`.
- The check behind Section 5 (about 1 s, NumPy only): `c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/notes/check_explainer_example.py`, with its output in `c:/Users/linda/VS Code Workspace/AI interpred models/1-algzoo-second-argmax-432/mechanistic-report/hide/notes/check_explainer_example_output.txt`.
- Temporary extraction files in the session scratchpad:
  - `C:\Users\linda\AppData\Local\Temp\claude\0c13cd028550ec56\cd99c4bf-b605-4494-8f6c-0ec20b242da2\scratchpad\explainer_digest\scripts\extract.py`
  - `C:\Users\linda\AppData\Local\Temp\claude\0c13cd028550ec56\cd99c4bf-b605-4494-8f6c-0ec20b242da2\scratchpad\explainer_digest\out\explainer_stripped.html` (the page without the base64 data, 29 KB)
  - `C:\Users\linda\AppData\Local\Temp\claude\0c13cd028550ec56\cd99c4bf-b605-4494-8f6c-0ec20b242da2\scratchpad\explainer_digest\out\explainer_text.txt`

**Summary.**

- The explainer is a condensed version of `paper.typ`: the same claims, the same numbers and the same six figures. The five PNGs are byte-identical to `figures/*.png`, and the inline SVG has the same body as `figures/algorithm.svg`.
- Its one substantive addition is a step-by-step table of the worked example (Section 2.1), with the remark that $10\max(x - M, 0)$ gives 4.0 for unit 4 at step 6 where the network gives 5.7.
- I found no numerical disagreement between the two documents. All differences are omissions or wording (Section 4).
- The page is static. It has no JavaScript, sliders or widgets. Its interactive elements are eight collapsible sections, two hover tooltips and one link.
- `paper.typ` is the more complete source. Section 4.3 lists what the explainer drops.
- Beyond the request, I checked the worked example against the weights (Section 5). The table is correct. The check also gives the thresholds the network used and four leads on the clock circle.

## 1. Outline of the explainer

1. **Title and subtitle.** "How a 432-parameter RNN finds the second-largest number". The subtitle names the audience (readers outside the team) and the prerequisite (RNN, ReLU), and points to `paper.pdf` and `code/`.
2. **"Short answer." box**, two paragraphs.
   - The core algorithm: the network approximately keeps the largest value so far and the runner-up, plus a rotating clock whose angle tracks the age of the largest value. Four comparators test each input against thresholds that swing with the clock. At the end the comparators move votes from the largest input's position to the runner-up's.
   - The negative result: the idealised account solves the task but reproduces only 5% of the network's mistakes. The full account reproduces 96%, but it is not shorter than the network and most of its coefficients are uninterpreted.
3. **"The core algorithm in one picture".** A block diagram (figure E1), three stat tiles (agreement and mistakes reproduced for the plain rule, the idealised account and the full account) and a note that defines agreement.
4. **"Details".** Eight collapsible sections, all closed on load. Titles and grey hints are verbatim.
   1. "The task, the network, and how we test an account" (hint "definitions"). Task, network and recurrence, parameter count, three tests, and why agreement alone is a weak test.
   2. "The algorithm, step by step" (hint "memory, comparisons, the idealised account"). Unit groups, memory, comparisons, the idealised account as six lines of pseudo-code, how it was fitted, and its blind spot (a negative runner-up).
   3. "Walk through one input" (hint "the real network, step by step"). Figure E2, the step table (explainer only), a note on unit 4, figure E3, "How the vote works here." and "Across inputs."
   4. "Evidence" (hint "the clock, and both accounts tested against the network"). Figure E4, the accounts table (paper Table 1), the size comparison, and the modes table (paper Table 2).
   5. "What the idealised account leaves out" (hint "removal experiments and the network's mistakes"). Figure E5 with five bullets, figure E6, the near-tie statistics, and the example "The network remembers more than the top three."
   6. "Two efforts with overlapping results". The two routes and the unit-role table (paper Table 3).
   7. "Limits and open questions". Five bullets.
   8. "Data, and what is in this folder". The samples and the folder contents.

## 2. In the explainer but not in `paper.typ` (verbatim)

### 2.1 The step-by-step table of the worked example (the main addition)

It sits in "Walk through one input", between figures E2 and E3. `paper.typ` has no such table, and its text gives only three of the ten inputs (1.8 at position 6, 1.4 at position 1, −1.0 at position 5).

| step | input | max M | runner-up S | age of M | unit 4 | comparators 1 / 6 / 7 / 8 | what happens |
|---|---|---|---|---|---|---|---|
| 0 | +0.3 | 0.3 | 0.0 | 0 | 3.0 | 0 / 0 / 0 / 0 | new maximum (0.3): record unit fires, comparators silent |
| 1 | +1.4 | 1.4 | 0.3 | 0 | 11.2 | 0 / 0 / 0 / 0 | new maximum (1.4): record unit fires, comparators silent |
| 2 | -0.5 | 1.4 | 0.3 | 1 | 0.0 | 18 / 8 / 25 / 10 | below the runner-up: 4 of 4 comparators fire |
| 3 | +0.2 | 1.4 | 0.3 | 2 | 0.0 | 1 / 9 / 16 / 6 | below the runner-up: 4 of 4 comparators fire |
| 4 | +0.9 | 1.4 | 0.9 | 3 | 0.0 | 0 / 6 / 7 / 0 | new runner-up (0.9): 2 of 4 comparators fire |
| 5 | -1.0 | 1.4 | 0.9 | 4 | 0.0 | 24 / 28 / 31 / 23 | below the runner-up: 4 of 4 comparators fire |
| 6 | +1.8 | 1.8 | 1.4 | 0 | 5.7 | 0 / 0 / 0 / 0 | new maximum (1.8): record unit fires, comparators silent; 1.4 becomes the runner-up |
| 7 | +0.1 | 1.8 | 1.4 | 1 | 0.0 | 17 / 12 / 23 / 15 | below the runner-up: 4 of 4 comparators fire |
| 8 | -0.3 | 1.8 | 1.4 | 2 | 0.0 | 16 / 20 / 28 / 21 | below the runner-up: 4 of 4 comparators fire |
| 9 | +0.4 | 1.8 | 1.4 | 3 | 0.0 | 10 / 15 / 19 / 10 | below the runner-up: 4 of 4 comparators fire |

Note under the table, verbatim:

> "M, S and age are computed from the inputs; unit values are the real network's. Unit 4 is only roughly 10·max(x − M, 0): at step 6 that formula gives 4.0, and the network gives 5.7."

What the table adds to `paper.typ`:

- The full input, $x = (0.3,\ 1.4,\ -0.5,\ 0.2,\ 0.9,\ -1.0,\ 1.8,\ 0.1,\ -0.3,\ 0.4)$.
- Unit 4's activation at the three records: 3.0, 11.2 and 5.7.
- All four comparator activations at every step.
- At the new runner-up (step 4) only two comparators fire, units 6 and 7.
- The conventions. $M$ and $S$ include the current input and start at 0, so $S = 0.0$ at step 0 is the floor, not an input. The age is 0 at the step where the maximum arrives.
- A number for "not exactly $M$": $10\,(1.8 - 1.4) = 4.0$ against 5.7. Section 5.2 gives the threshold the network used at that step, 1.240.

Every unit value in the table matches the real weights after rounding (Section 5.1).

### 2.2 Sentences that `paper.typ` lacks or words differently

1. Unit 4, in "The algorithm, step by step": "Unit 4 fires when the input beats the maximum, roughly 10·max(x − M, 0). Its real threshold is close to, but not exactly, M." The paper says only that the threshold "is not exactly $M$".
2. Single-unit test: "run the real network, but overwrite one unit at every step with the account's value for it. If the account describes that unit correctly, the network's answers should survive." The second sentence, the rationale, is not in the paper.
3. Limits: "Some unit roles are vague. Units 3, 10, 11 and 15 are described only as parts of the wave group." This is a fifth limitation bullet. The paper states the fact in its Introduction, not under Limitations.
4. Parameter count: "w has 16 weights, W has 256 and the readout U has 160: 432 in all." The paper gives the shapes $16 \times 16$ and $10 \times 16$ instead.
5. Caption of figure E2: "(A) The inputs. (B) The real network's units at each step. Unit 4 fires when a new maximum arrives (positions 0, 1, 6), and the comparators are silent then." Same content as the paper's caption, shorter.
6. Caption of figure E1: "The same 16 ReLU units run at every step; their activity plays these roles. Unit numbers are the network's own."
7. Note under the stat tiles: "Agreement means the account, put in place of the network's hidden state, gives the network's answer."
8. Subtitle: "It assumes you know what an RNN and a ReLU are."
9. The idealised account as pseudo-code. The content equals the paper's prose and equations. Only the notation is new: "z_M, z_S", "[record last step]" for the paper's $r$, and "(both start at 0)".

```
M' = max(M, x)          S' = max(S, min(M, x))          (both start at 0)
clocks z_M, z_S: multiply by e^(0.74i) (42.4°) each step; reset to 1 when the value is replaced;
                 on a record the old maximum's clock passes to the runner-up
comparator j:  g_j · max(θ_j − x, 0),  θ_j = S + (M − S)·(A_j + B_j·Re z_M + C_j·Im z_M + D_j·[record last step])
unit 4: 10·max(x − M, 0)    unit 5: 12·max(x_(t−1) − S, 0)    unit 9: 1.5·M·[record last step]
wave units: ReLU of a half-integer mix of 12 features (levels, z_M, and z_S with its 2nd and 3rd powers)
```

In LaTeX: $M' = \max(M, x)$, $S' = \max(S, \min(M, x))$, $z \leftarrow z\,e^{0.74 i}$, $h_j = g_j \max(\theta_j - x, 0)$ with $\theta_j = S + (M - S)(A_j + B_j\,\mathrm{Re}\,z_M + C_j\,\mathrm{Im}\,z_M + D_j r)$, $h_4 = 10 \max(x - M, 0)$, $h_5 = 12 \max(x_{t-1} - S, 0)$, $h_9 = 1.5\,M\,r$.

### 2.3 Interactive elements (none in the paper)

- Eight `<details>` sections that open on click. None is open on load, so a reader who does not click sees only the short answer, figure E1 and the stat tiles.
- Two hover tooltips (`<abbr>`) in the short answer:
  - on "clock": "A two-dimensional part of the hidden state that rotates by about 42° each step. A new maximum restarts it, so its angle tracks how many steps ago the maximum arrived."
  - on "comparator": "Units of the form g·max(θ − x, 0): they fire when the input falls below a threshold θ set by memory."
- One link, to ARC's AlgZoo post (`https://www.alignment.org/blog/algzoo-uninterpreted-models-with-fewer-than-1-500-parameters/`).
- Horizontal scrolling for the diagram and the tables on narrow screens.
- There is no JavaScript. Nothing is computed in the page.

### 2.4 Presentation elements with no new content

- Three stat tiles: 'Rule "answer the true 2nd-largest"', "95.3%", "agreement with the network; 0% of its mistakes". "Idealised account (171 constants)", "93.2%", "agreement with the network; 5% of its mistakes". "Full account (334 constants)", "99.6%", "agreement with the network; 96% of its mistakes". These are paper Table 1, rounded.
- A folder listing in the last section (`paper.pdf`, `code/`, `figures/`, `original-papers/`).

### 2.5 Figures

No figure is explainer-only: all six are the paper's. Several numbers are printed only inside the figures and appear in the text of neither document. Section 3 lists them, marked "figure only".

## 3. The explainer's figures

| id | where in the explainer | same file as | paper figure | alt text |
|---|---|---|---|---|
| E1 | "The core algorithm in one picture" (inline SVG) | `figures/algorithm.svg` (same body) | Fig. 1 | "Diagram of the algorithm" |
| E2 | "Walk through one input" | `figures/example_trace.png` (same SHA-256) | Fig. 2 | "Inputs and the 16 hidden units at each step" |
| E3 | "Walk through one input" | `figures/example_vote.png` (same SHA-256) | Fig. 3 | "Contribution of comparators and other units to each output logit" |
| E4 | "Evidence" | `figures/clock_phase.png` (same SHA-256) | Fig. 4 | "Clock angle against age of the maximum" |
| E5 | "What the idealised account leaves out" | `figures/ablations.png` (same SHA-256) | Fig. 5 | "Agreement and mistakes reproduced after one change to the full account" |
| E6 | "What the idealised account leaves out" | `figures/accuracy_by_gap.png` (same SHA-256) | Fig. 6 | "Network accuracy by smallest gap among the top three" |

The explainer also has four tables: the step table (Section 2.1, explainer only) and the accounts, modes and unit-role tables (paper Tables 1 to 3).

### E1. Block diagram of the algorithm

- No axes. Header: "Every step runs the same 16 ReLU units: h_t = ReLU(w·x_t + W·h_(t−1))".
- Left to right: "input x, one per step" → blue box "1. Compare x with memory" → orange box "2. Update memory" → grey box "3. After step 9: linear readout". A feedback arrow from box 2 to box 1 is labelled "thresholds for the next step come from memory".
- Box 1 holds three cards:
  - "record unit (4)": "fires when x beats the maximum M, roughly 10·max(x − M, 0)".
  - "comparators (1, 6, 7, 8)": "fire when x is below a threshold θ near S and M that swings with the clock: g·max(θ − x, 0)".
  - "one-step pulses (5, 9)": "5: after a record or new runner-up", "9: after a record".
- Box 2: "two levels" ("M: largest value so far", "S: runner-up so far"), "a clock" ("turns ~42° per step; a record restarts it, so its angle ≈ age of M") and "history waves" ("ages of earlier candidates, including the runner-up").
- Box 3: "ages → positions; comparators push the max down and the runner-up up".
- The reader should see one loop per step (compare, update memory, memory sets the next thresholds) and a single linear readout at the end.

### E2. One input and the 16 units (two panels)

- Panel A, "One input sequence". Bars, x-axis "position (step)" 0 to 9, y-axis "input value" (−1.3 to 2.4). Position 1 is orange and labelled "2nd largest (answer)". Position 6 is blue and labelled "largest".
- Panel B, "The network's 16 hidden units at each step". Heatmap, x-axis "position (step)", one row per unit in the order 1, 6, 7, 8 (labelled "comparators"), 4, 5, 9 ("pulses"), 0, 2, 3, 10 to 15 ("wave units"). Colour is "activation", white at 0 and darkest at the largest value, 33.2 (unit 2 at step 7).
- What the caption points to: unit 4 is lit at steps 0, 1 and 6 (3.0, 11.2, 5.7) and the comparator rows are blank there. The paper's caption adds that the comparators fire most strongly at step 5 (23.6, 27.5, 30.6, 23.3 for units 1, 6, 7, 8).
- Also visible, not mentioned in either text (values from `code/analysis.json`):
  - Unit 5 is lit at steps 1, 2, 5, 7 (3.3, 16.0, 7.1, 14.8): the step after each record (0, 1, 6) and after the new runner-up (4).
  - Unit 9 is faint at steps 1, 2, 7 (0.13, 2.33, 2.74): the step after each record.
  - Unit 2 is 5.3 at step 1, 24 to 26 at steps 2 to 6 and 32 to 33 at steps 7 to 9. It follows the maximum with one step of delay. The docstring of `code/exact_rewrite.py` gives unit 2 as about 17.3 times the maximum of the earlier inputs ($17.3 \times 0.3 = 5.2$, $17.3 \times 1.4 = 24.2$, $17.3 \times 1.8 = 31.1$).
  - Units 12, 11, 10 fade out in that order after the record at step 1: unit 12 reaches 0 at step 5, unit 11 at step 6, and unit 10 is down to 4.3 at step 6.

### E3. The final vote

- Title "C. How the final state votes for each output position". Grouped bars, x-axis "output position" 0 to 9, y-axis "contribution to logit" (−95 to 75). Grey is "other 12 units (waves and pulses)", blue is "comparator units 1, 6, 7, 8", orange is "total logit". Two annotations: "answer (position 1)" and "largest input: other units +47, comparators −35".
- Exact bar heights (`code/analysis.json`; reproduced by my check). The text of both documents quotes only positions 1, 2, 6 and 9; the rest are figure only.

| output position | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| other 12 units | 5.42 | −2.13 | −45.80 | −66.51 | −32.10 | 3.18 | 47.20 | 21.68 | 2.27 | 50.95 |
| comparators | 9.21 | 38.88 | 57.12 | 45.87 | 8.10 | −25.40 | −34.77 | −28.79 | −13.41 | −81.87 |
| total logit | 14.63 | 36.75 | 11.32 | −20.64 | −24.00 | −22.22 | 12.43 | −7.11 | −11.14 | −30.92 |

- The reader should see that the orange bar is highest at position 1 (the answer), that at position 6 (the largest input) grey is +47 and blue is −35, that the comparators' largest vote (+57 at position 2) is cancelled by grey (−46), and that position 9 has the extreme bars of both groups.
- Also visible: over positions 0 to 8 each group's votes form one smooth wave, with the two waves roughly opposite in sign. The winning margin is 36.75 against 14.63 (position 0) and 12.43 (position 6).

### E4. Clock angle against the age of the maximum

- Title "The clock's angle tracks the age of the maximum". x-axis "age of the current maximum (steps since it arrived)", 1 to 9. y-axis "clock angle (degrees, unwrapped)", about 60 to 395. Blue line with dots is the "median angle", the light band is the "middle 50% of steps", and the dashed line is "straight line, 42.0° per step".
- What is plotted (`code/analysis.py`): the angle $\mathrm{atan2}(\text{clock\_s}, \text{clock\_c})$ of `memory[8:10]` in `code/exact_rewrite.py`, after each step, over 4,000 inputs and only for steps whose running maximum is positive. Age 0 is not plotted.
- The reader should see a straight line with a narrow band: the angle encodes the age.
- Values behind the plot (`code/analysis.json`). The per-age numbers are in neither text.

| age | 0 (not plotted) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| median angle (°) | 181.0 | 62.8 | 95.9 | 138.9 | 181.8 | 225.6 | 268.4 | 310.4 | 350.5 | 395.0 |
| lower quartile (°) | 32.9 | 62.0 | 86.9 | 127.1 | 172.1 | 218.5 | 260.5 | 302.1 | 339.6 | 386.4 |
| upper quartile (°) | 317.0 | 65.6 | 98.8 | 142.8 | 185.9 | 229.1 | 271.7 | 314.0 | 355.4 | 399.7 |
| median radius | 0.041 | 0.610 | 0.631 | 0.588 | 0.552 | 0.549 | 0.558 | 0.578 | 0.562 | 0.546 |
| steps | 8924 | 6945 | 5433 | 4293 | 3348 | 2596 | 1915 | 1332 | 860 | 423 |

- Successive differences of the median: 33.1°, 43.0°, 42.9°, 43.8°, 42.8°, 42.0°, 40.1°, 44.5°. The first step is short. The band is skewed: the lower quartile is 1° to 12° below the median and the upper quartile 3° to 5° above.

### E5. One change to the full account at a time

- Title "One change to the full account at a time, no refitting". Two panels of horizontal bars that share 19 rows. Left x-axis "agreement with the network (%)" with a dotted line at 98. Right x-axis "network mistakes reproduced (%)" with a dotted line at 80.
- Colours: dark grey is no change, green is "angles and harmonic decay", blue is "idealise levels or clock", orange is "remove an extra mode", purple is "remove calibration", light grey is "control: remove a part both accounts have".
- Data: 19,776 inputs, the full account's development set, numbers from `code/accounts_working_notes.md` (hard-coded in `code/figures.py`).

| change (label verbatim) | agreement % | mistakes % | in the text? |
|---|---|---|---|
| no change | 99.7 | 96.7 | figure only |
| harmonics tied at exactly 2× and 3× of 41.6° | 99.7 | 96.6 | both: "costs nothing" |
| main clock angle 42.75° → 42.40° | 99.4 | 93.3 | figure only |
| all three angles → the idealised 42.40° family | 98.3 | 84.4 | both |
| harmonics tied to 2× and 3× of 42.75° | 97.7 | 81.2 | paper only (81%) |
| harmonic decay tied to the main clock's | 98.3 | 82.4 | figure only |
| echo mode removed | 99.6 | 96.0 | figure only |
| main clock never decays (radius → 1) | 92.2 | 64.0 | both, as ranges |
| both levels held exactly (factor → 1) | 88.8 | 58.8 | both, as ranges |
| alternating mode removed | 91.3 | 68.7 | both, agreement only |
| short transient removed | 89.5 | 58.9 | both, agreement only |
| short-lived clock removed | 50.9 | 36.6 | both, agreement only |
| comparators read only levels and main clock | 36.0 | 22.8 | both, agreement only |
| comparator gains → 10.5, 11, 13, 12 | 89.4 | 65.4 | figure only |
| wave-unit input weights zeroed (not unit 14) | 97.4 | 77.9 | figure only |
| all nine wave-unit input weights zeroed | 89.5 | 54.6 | figure only |
| clipping not written back into memory | 60.1 | 40.5 | both, agreement only |
| unit 9 pulse removed (control) | 67.8 | 41.8 | figure only |
| unit 5 pulse removed (control) | 21.3 | 29.8 | figure only |

- The reader should see that tying the angles costs little agreement but many mistakes, and that removing the short-lived clock, the comparators' extra inputs, the clipping write-back or unit 5 destroys agreement.
- A figure-only point worth keeping: rounding the comparator gains from (10.56, 11, 13.17, 12.31) to (10.5, 11, 13, 12) alone lowers agreement to 89.4%.

### E6. Accuracy by gap

- Title "Errors are near-ties". Bars, x-axis "smallest gap among the top three values (max − 2nd, 2nd − 3rd)" in seven bins, y-axis "network accuracy (%)". 20,000 inputs.

| smallest gap | 0–0.01 | 0.01–0.025 | 0.025–0.05 | 0.05–0.1 | 0.1–0.2 | 0.2–0.4 | ≥0.4 |
|---|---|---|---|---|---|---|---|
| accuracy % (bar label) | 61 | 80 | 91 | 97 | 98 | 99 | 100 |
| inputs | 809 | 1198 | 1811 | 3150 | 4648 | 5241 | 3143 |

- The reader should see accuracy rise from 61% for near-ties to 100% for well-separated inputs. The bar values are figure only. The text gives 77% of mistakes on the 19% of inputs with a gap below 0.05, and 63% of mistakes naming the third-largest.

## 4. Disagreements between the explainer and `paper.typ`

### 4.1 Numbers

None found. Every number in the explainer's text that also occurs in `paper.typ` has the same value. The explainer's summary rounds (95.3%, 93.2%, 99.6%, 5%, 96%), consistent with its own table and the paper's (95.29%, 93.18%, 99.64%, 5.2%, 96.1%).

### 4.2 Wording that shifts the meaning slightly

1. **Fitting loss.** Explainer: "0.02 × the squared difference from the network's hidden states". Paper: "0.02 times the mean squared difference". The working notes say "0.02 times mean squared hidden-state error", so the paper is right.
2. **Unit 4's threshold.** Explainer: "close to, but not exactly, M". Paper: "not exactly $M$". In the explainer's own example the threshold is 0.16 below $M = 1.4$ at step 6 (Section 5.2).
3. **Tests.** The explainer defines three tests and never defines KL, although its table has a "KL (nats)" column. The paper defines a fourth, "KL divergence from the network's output distribution to the program's, in nats".
4. **Limitations.** Five bullets against four (Section 2.2, item 3).
5. **History waves.** Explainer text: they "carry information about earlier candidates". Paper text adds "including the runner-up". Figure E1, in both, says "ages of earlier candidates, including the runner-up".
6. **Idealised accuracy.** The paper says the idealised account "solves the task more often than the network (96.5% against 95.3%)". The explainer's prose says only that it "solves the task"; its table has 96.46% and 95.29%.
7. **Labels.** Role-table header "this explainer" against "this paper". Modes-table row "unit 4 input weight" against "record unit input weight".

### 4.3 In `paper.typ` but not in the explainer

1. The KL test definition and its direction (4.2, item 3).
2. The 19,776 inputs "were held out from fitting the idealised account". The explainer says only that they were used to choose the full account's rounding.
3. Caption of Fig. 2: "On other steps the comparators fire, most strongly at position 5 (input −1.0)."
4. Clock: "at every age the middle half of steps lies within about 12° of the median", "steps with a positive maximum", "The dashed line is a straight-line fit", and "so later inputs keep pushing it back out".
5. "each unit replaced alone keeps at least 99.5% of the network's answers."
6. "The three pulses are kept separate, and every constant is rounded to a fixed grid."
7. Table 2: the idealised harmonics' angles "(84.8°, 127.2°)", "(factor 1)", and the caveat that the runner-up form is "not a coefficient of `full_account.py`".
8. "The two extra rotations have their own base frequency, about 41.6°", and "tying the two rotations to multiples of the main clock's 42.75° keeps 81% of mistakes."
9. "Dotted lines mark 98% and 80%."
10. On the controls: "Their large cost shows that these pulses are essential to both accounts, not that the idealised account lacks them."
11. On error causation: "for example by adding parts to the idealised account and checking which mistakes appear."
12. On the top-three example: "The changed input was briefly the runner-up, then dropped to fourth, and it still changes the answer."
13. Limitations: thresholds "in the idealised form", "any one of units 1, 6, 7 or 8", and "We know where the mistakes are and which parts the full account needs, but not which parts produce which mistakes."
14. The working papers' numbers: "170 constants, 96.6% accuracy and 6% of mistakes for the idealised account".
15. The Introduction's framing: a network for this task "must keep more than a running maximum", and the two questions that the two accounts separate.

## 5. Cross-checks beyond the four requested items

Script: `check_explainer_example.py` in this folder. It reads the weights and `code/exact_rewrite.py` and writes nothing outside this folder.

### 5.1 The step table matches the real network

Running $h_t = \mathrm{ReLU}(W_{ih} x_t + W_{hh} h_{t-1})$ on the example gives unit 4 = 3.047, 11.209 and 5.691 at steps 0, 1 and 6, and comparator values that round to every entry of the table (for example 18.29, 8.21, 25.34, 9.65 at step 2). The logits and the vote split equal `code/analysis.json`. The exact rewrite reproduces the hidden state within 0.0033 at every step.

### 5.2 The thresholds the network used in the worked example

Because the network has no biases and $W_{ih}[4] > 0 > W_{ih}[j]$ for $j \in \{1, 6, 7, 8\}$, these are exact identities, not fits:

$$h_t[4] = w_4\,\mathrm{ReLU}\big(x_t - \theta^{(4)}_t\big), \qquad \theta^{(4)}_t = -\frac{W_{hh}[4,\cdot]\,h_{t-1}}{w_4}, \qquad w_4 = W_{ih}[4] = 10.157,$$

$$h_t[j] = g_j\,\mathrm{ReLU}\big(\theta^{(j)}_t - x_t\big), \qquad \theta^{(j)}_t = \frac{W_{hh}[j,\cdot]\,h_{t-1}}{g_j}, \qquad g_j = -W_{ih}[j] = (10.565,\ 10.996,\ 13.169,\ 12.312).$$

$M_{t-1}$ and $S_{t-1}$ are the maximum and runner-up of $x_0..x_{t-1}$, floored at 0. At $t = 0$ all thresholds are 0.

| $t$ | $x_t$ | $M_{t-1}$ | $S_{t-1}$ | $\theta^{(4)}_t$ | $\theta^{(1)}_t$ | $\theta^{(6)}_t$ | $\theta^{(7)}_t$ | $\theta^{(8)}_t$ |
|---|---|---|---|---|---|---|---|---|
| 1 | +1.4 | 0.3 | 0.0 | 0.296 | 0.252 | −0.013 | 0.305 | −0.009 |
| 2 | −0.5 | 1.4 | 0.3 | 1.379 | 1.232 | 0.247 | 1.425 | 0.284 |
| 3 | +0.2 | 1.4 | 0.3 | 1.367 | 0.261 | 1.027 | 1.418 | 0.661 |
| 4 | +0.9 | 1.4 | 0.3 | 1.351 | 0.581 | 1.457 | 1.414 | 0.382 |
| 5 | −1.0 | 1.4 | 0.9 | 1.296 | 1.236 | 1.502 | 1.327 | 0.893 |
| 6 | +1.8 | 1.4 | 0.9 | 1.240 | 1.590 | 1.470 | 1.221 | 1.060 |
| 7 | +0.1 | 1.8 | 1.4 | 1.789 | 1.688 | 1.237 | 1.814 | 1.278 |
| 8 | −0.3 | 1.8 | 1.4 | 1.768 | 1.176 | 1.541 | 1.815 | 1.429 |
| 9 | +0.4 | 1.8 | 1.4 | 1.784 | 1.306 | 1.802 | 1.809 | 1.229 |

What I read from it:

1. Unit 4's threshold sits below the maximum and sags as the maximum ages: $M_{t-1} - \theta^{(4)}_t = 0.004,\ 0.021,\ 0.033,\ 0.049,\ 0.104,\ 0.160$ for $t = 1..6$, then $0.011,\ 0.032,\ 0.016$ after the new record. The explainer's 5.7 is $10.157\,(1.8 - 1.240) = 5.69$.
2. At step 6 unit 4 would have fired for any $x_6 > 1.240$. That includes $1.240 < x_6 < 1.4$, which is not a record.
3. $\theta^{(7)}$ is within 0.03 of $M_{t-1}$ at $t = 1..4$ and $t = 7..9$. It sags to $M - 0.07$ and $M - 0.18$ at $t = 5, 6$, when the maximum is 4 and 5 steps old.
4. $\theta^{(1)}$, $\theta^{(6)}$ and $\theta^{(8)}$ swing between about $S - 0.2$ and $M + 0.2$. At step 6, $\theta^{(1)} = 1.59$ and $\theta^{(6)} = 1.47$ are above the old maximum 1.4, so a narrow record $1.4 < x_6 < 1.47$ would have fired unit 4 together with units 1 and 6. "Comparators silent at a record" is therefore typical, not a rule. The docstring of `code/exact_rewrite.py` gives 88% for records at $t = 9$.

### 5.3 Imprecisions shared by the explainer and the paper

1. **"12 features".** The listed ones (two levels, $z_M$, three powers of $z_S$) count to 10. `code/idealised_account.py` has two more: $M$ times a flag for a record at this step, and $S$ times a flag for a new runner-up at this step. The clock features are scaled by the levels ($M\,\mathrm{Re}\,z_M$, $M\,\mathrm{Im}\,z_M$, $S\,\mathrm{Re}\,z_S^k$, $S\,\mathrm{Im}\,z_S^k$), and unit 14 gets an extra $-1.25\,x$ before its ReLU.
2. **Unit 5's $S$.** In the code $h_t[5] = 12 \max(x_{t-1} - S_{t-2}, 0)$: $S$ is the runner-up before $x_{t-1}$ arrived.
3. **"0.974·S − 0.016·M".** This is the one-step free flow of the runner-up coordinate, $R \mapsto 0.974\,R - 0.016\,L$ with $L \approx M$ and $R \approx S$ (working notes: ".97363 R − .01615 L"; `code/exact_rewrite.py` has `RUNNER_POLE, RUNNER_LEAK = 0.97349, -0.0159`). It is not the value that the coordinate stores.
4. **"Position 9 … gets the largest votes of both groups (+51 and −82)".** +51 is the other units' largest positive vote. Their largest vote in size is −66.5 at position 3.
5. **Legend of E3.** It says "other 12 units". `code/figures.py` plots the 11 units other than unit 4 and the comparators. Unit 4 adds exactly 0 here because $h_9[4] = 0$, so no number changes.
6. **Baseline of E5.** Its "no change" row (99.7%, 96.7%) is the full account on its own development set. The accounts table has 99.64% and 96.1% on 131,072 fresh inputs.
7. **"About 42° per step".** This is the straight-line fit over ages 1 to 9 (42.0°). The step from age 1 to age 2 is 33.1°, and the other seven are 40.1° to 44.5°.
8. **"Four samples".** The near-tie statistics (seed 2 in `code/analysis.py`) and the readout statistics (seed 3 in `code/readout_stats.py`) use two different 20,000-input samples. The clock's 4,000 inputs are the first 4,000 rows of the seed-2 sample. The 16,076 inputs are the seed-3 rows whose maximum and runner-up both lie in positions 0 to 8.
9. **Numbers computed but not reported.** `code/analysis.json`: 21.9% of mistakes name the largest input. `code/readout_stats.json`: the comparators' largest vote among positions 0 to 8 is at the runner-up's position on only 12.7% of inputs (the other units' on 7.7%); accuracy is 96.0% when the runner-up is the last input and 93.1% when the maximum is the last input.

### 5.4 Leads on the clock circle

These rest on the constants of `code/idealised_account.py` and `code/exact_rewrite.py` and on the one worked example. I ran no statistics.

**(a) What the explainer's formula already implies.** With $z_M = e^{i\phi}$, the idealised thresholds are

$$\theta_j = S + (M - S)\big(A_j + B_j \cos\phi + C_j \sin\phi + D_j r\big).$$

As the clock turns, the four thresholds trace an ellipse with centre $S + (M - S) A_j$ and amplitude $(M - S)\sqrt{B_j^2 + C_j^2}$. Both centre and size are set by the stored $M$ and $S$. The coefficients are in neither document. In `code/idealised_account.py` (`ALPHA`), $(A, B, C, D)$ is $(0.57, -0.22, -0.18, 0.27)$ for unit 1, $(0.75, -0.04, 0.03, -0.74)$ for unit 6, $(0.76, 0.07, -0.01, -0.04)$ for unit 7 and $(0.68, 0.04, -0.32, -0.47)$ for unit 8.

**(b) An offset proportional to the maximum.** In `code/exact_rewrite.py` (`SYNTHESIS`), each of the 13 non-pulse units reads the `level` coordinate with a positive weight, from 3.12 (unit 3) to 16.35 (unit 12). Each reads the clock with amplitude $\sqrt{c^2 + s^2}$ between 1.85 (unit 13) and 19.0 (unit 12). So in unit space the clock's circle rides on an offset proportional to the level, which is about $M$. The `runner` column adds a second offset, positive for units 1, 2, 3, 6, 7, 8, 13 and negative for units 0, 10, 11, 12, 14, 15.

**(c) The radius is set by the record increment, and the restart takes three steps.** In the exact rewrite, new clock = rotated old clock + pulse kick + write-back (input drive and ReLU clipping). In the example:

| $t$ | event | rotated old clock | pulse kick | write-back | new clock |
|---|---|---|---|---|---|
| 1 | record | 0.001 | 0.300 at 62.7° | 0.244 at −103.0° | 0.089 at 19.0° |
| 2 | step after record | 0.084 at 61.7° | 1.098 at 61.8° | 0.005 | 1.179 at 62.0° |
| 3 | | 1.121 at 104.8° | 0.154 at 24.8° | 0.059 at 76.5° | 1.213 at 96.2° |
| 4 | new runner-up | 1.154 at 139.0° | 0 | 0.481 at −15.0° | 0.751 at 122.7° |
| 5 | | 0.714 at 165.4° | 0 | 0.033 at −83.5° | 0.703 at 167.9° |
| 6 | record | 0.669 at −149.3° | 0.052 at −75.2° | 0.528 at 61.3° | 0.316 at 166.8° |
| 7 | step after record | 0.301 at −150.4° | 0.560 at 62.7° | 0.001 | 0.350 at 90.6° |
| 8 | | 0.333 at 133.3° | 0.189 at 33.7° | 0.051 at 81.7° | 0.403 at 99.1° |
| 9 | | 0.383 at 141.9° | 0 | 0.004 | 0.380 at 141.4° |

- At a record step the write-back shrinks the clock: from 0.300 to 0.089 at step 1 and from 0.685 to 0.316 at step 6. The docstring says "A record zeroes it"; here it removes 70% and 54%.
- On the next step the record pulse $p = h[4]/10.157 = \max(x - \theta^{(4)}, 0)$ adds $1.0005\,p$ at $62.70°$ (`PULSE_DRIVE`). That direction is the median clock angle at age 1 in figure E4 (62.8°).
- One step later the pulses of units 5 and 9 add a smaller kick at 25° to 34°. This fits the short first step of 33.1° in figure E4.
- So the clock radius follows the record increment $p$, not $M$: 1.18 to 1.21 after $p = 1.10$ (steps 2, 3) and 0.35 to 0.40 after $p = 0.56$ (steps 7 to 9, where the kick lands on an uncancelled remainder).

**(d) A new runner-up also moves the clock.** At step 4 ($x = 0.9$, units 1 and 8 silent) the write-back takes the clock from 1.154 at 139.0° to 0.751 at 122.7°. At low inputs the write-back is small (0.001 to 0.06), as the docstring's "A low input leaves no trace" says. The clock is therefore not a pure function of the age of the maximum.
