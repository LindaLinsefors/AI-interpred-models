# Two accounts of a second-maximum circuit

## What the circuit is doing

The target is a 16-unit bias-free ReLU RNN reading ten independent standard-normal scalars: `h_t = ReLU(W_ih x_t + W_hh h_(t-1))`, starting at zero. Only `h_9` reaches the ten-class readout `W_out h_9`; the label is the zero-based position of the second-largest input. Its 432 weights are 16 input, 256 recurrent, and 160 readout weights, and its task accuracy is about 95.3%.

A large input has two jobs: it changes the values to beat, and it writes its **age** into memory. The network approximately keeps a maximum level and a runner-up level. A record replaces the maximum and makes the old maximum a candidate runner-up; an intermediate input can replace the runner-up without replacing the maximum. Low inputs mostly cancel out of persistent memory. The age code is a rotating, clipped wave, not a one-hot position register. Additional waves carry candidate history. At the end, age `a` means position `9-a`.

Four large negative-input units compare the current input with different thresholds between or around those levels. Their clock-dependent clipping helps subtract the largest candidate's position while retaining the second-largest candidate's evidence. Unit 4 supplies a record increment; units 5 and 9 supply delayed contender/reset corrections. The final linear readout, not an explicit sort operation, chooses the answer.

These are two different strengths of account. **Account 1 makes the order-statistic algorithm literal and uses a small, task-calibrated harmonic emission into the 16 units. It is not a high-fidelity causal replacement of those units. Account 2 is the high-fidelity recurrence.** Its dense event/write/read calibration is retained and measured, rather than silently identified with the ideal algorithm.

The earlier exact rewrite identified the levels, clipped clock and candidate register; the second write-up used 3,600 age-conditioned gains; the third found no small recursion meeting the strict causal bar. This update does **not** overturn that negative result: the small account below is judged as an ideal task algorithm, while the faithful account retains dense calibration. In particular, its high task accuracy is not a new claim of a small strict-bar explanation.

## Account 1: two exact levels, reset clocks, approximate neural emission

Let `M,S` be the preceding *positive* maximum and runner-up, initially zero. The ideal update is

\[
M'=\max(M,x),\qquad S'=\max(S,\min(M,x)).
\]

Here comparisons with `M` and `S` perform the rank update directly. The clock-dependent comparator units below are emitted observations, not the cause of that update. This is an intentional algorithmic abstraction: Account 2 instead updates its memory through the units’ clipping and dense writeback.

Each value has a unit-modulus complex clock. Every step multiplies both clocks by `exp(0.74i)`; a new value's clock resets to 1. On a record, the rotated old maximum clock moves to the runner-up clock. Thus the angle is **42.399° per step**, with tied harmonics at **84.798° and 127.197°**. Unlike the network's clock, these clocks never damp or partially reset. Only the values, clocks, a one-step reset flag, and the preceding contender increment are carried; the prefix is not rescanned.

Before that update, units `j=1,6,7,8` use

\[
h_j=g_j[\theta_j-x]_+,\quad
\theta_j=S+(M-S)\alpha_j,\quad
\alpha_j=A_j+B_j\Re z_M+C_j\Im z_M+D_jr.
\]

Here `r` marks a record on the preceding step. The reset term matters: a smooth sinusoid alone misses the immediate post-record kick. “Mix” does **not** mean convex mix: a threshold may slightly overshoot either level.

| unit | gain | A | B (cosine) | C (sine) | D (reset) |
|---|---:|---:|---:|---:|---:|
|1|10.5|0.57|−0.22|−0.18|0.27|
|6|11|0.75|−0.04|0.03|−0.74|
|7|13|0.76|0.07|−0.01|−0.04|
|8|12|0.68|0.04|−0.32|−0.47|

The ideal pulses are `h4=10[x-M]+`, `h5=12 e`, and `h9=1.5 M r`, where `e` is the preceding step's `[x-S]+`. In particular the ideal coincidence pulse is only a coarse reset spike, not the network's narrow coincidence test.

After updating, the remaining nine units read **12 named features**: the two levels; the maximum's first harmonic and reset spike; and the runner-up's first, second and third harmonics and reset spike. Unit 14 additionally receives `−1.25x`. A half-integer linear emission followed by ReLU supplies these units. It is a spectral response table over named quantities, **not a table indexed by age, age pairs, input bins, or data rows**. The unchanged network readout consumes all 16 emitted units. This emission is deliberately fitted through that readout, so its success should not be mistaken for independent evidence that each real unit literally computes its ideal formula.

### Complete runnable program

```python
"""An ideal positive top-two recursion with reset clocks, ideal pulses and a task-calibrated harmonic emission. Not a high-fidelity neural recurrence."""
import math
import numpy as np
PREDICTS = ('algzoo-m16-10/h',)
READS = ()
ROT = complex(math.cos(.74), math.sin(.74))
COMPARATORS = [1, 6, 7, 8]
WAVES = [0, 2, 3, 10, 11, 12, 13, 14, 15]
GAIN = np.array([10.5, 11, 13, 12])
ALPHA = np.array([
    [0.57, -0.22, -0.18, 0.27],  # unit 1
    [0.75, -0.04, 0.03, -0.74],  # unit 6
    [0.76, 0.07, -0.01, -0.04],  # unit 7
    [0.68, 0.04, -0.32, -0.47],  # unit 8
])
EMIT = np.array([
    [10.5, -5.5, 0.5, -3.5, -2.0, -2.0, -2.0, 3.0, 0.0, 1.5, 3.0, 1.0],  # unit 0
    [12.5, 7.5, -0.5, 1.0, -6.5, 0.5, 1.5, -0.5, 1.5, -2.0, 1.5, -1.0],  # unit 2
    [3.0, 1.5, 1.0, 0.0, -3.0, 0.5, -0.5, 1.5, -3.0, 5.5, -0.5, -8.0],  # unit 3
    [10.5, -5.5, 0.5, 4.5, -4.0, -0.5, 6.5, 2.5, -0.5, -0.5, -2.5, 3.0],  # unit 10
    [10.0, -6.0, 3.5, 3.0, -6.5, 4.0, 3.0, -2.0, -3.0, 1.5, 1.5, 1.0],  # unit 11
    [13.0, -4.0, 5.0, -1.5, -7.5, 6.0, -1.5, -1.0, 2.0, 2.5, -2.0, -3.5],  # unit 12
    [5.5, 1.5, -0.5, 0.0, -2.0, -0.5, -1.0, 3.5, 0.5, 0.5, 0.0, -3.0],  # unit 13
    [10.5, -1.5, 0.0, -3.5, -2.0, -2.5, -3.0, -2.5, 0.0, -0.5, -2.5, 8.5],  # unit 14
    [8.0, -1.5, 3.0, -3.0, -6.0, 3.0, -3.5, 4.0, -4.5, -0.5, 0.0, -5.5],  # unit 15
])

def step(carry, prefix, upstream):
    if carry is None:
        carry = (0., 0., 1., 1., 1., 0.)
    M, S, zM, zS, reset, contender = carry
    x = float(prefix.values[prefix.position])
    phase = np.array([1., zM.real, zM.imag, reset])
    threshold = S + (M-S)*(ALPHA @ phase)
    h = np.zeros(16)
    h[COMPARATORS] = GAIN * np.maximum(threshold-x, 0.)
    h[4] = 10*max(x-M, 0.)
    h[5] = 12*contender
    h[9] = 1.5*M*reset
    contender = max(x-S, 0.)
    zM, zS = zM*ROT, zS*ROT
    reset = runner_reset = 0.
    if x > M:
        M, S, zM, zS, reset = x, M, 1., zM, 1.
    elif x > S:
        S, zS, runner_reset = x, 1., 1.
    harmonics = (zS, zS**2, zS**3)
    features = [M, S, M*zM.real, M*zM.imag, M*reset]
    for z in harmonics:
        features.extend((S*z.real, S*z.imag))
    features.append(S*runner_reset)
    h[WAVES] = EMIT @ features
    h[14] -= 1.25*x
    h = np.maximum(h, 0.)
    return (M, S, zM, zS, reset, contender), {PREDICTS[0]: h.tolist()}
```

### Size, fitting, and measurements

The complete block contains **171 numeric-literal occurrences**, counting every coefficient, zero, repeated literal, unit index, dimension and exponent. Specifically: 108 emission entries, 16 threshold coefficients, 4 comparator gains, two written copies of the rotation angle, 3 pulse gains, the unit-14 input gain, and 37 structural/index/initialisation literals. This deliberately conservative source-level count is **below 432**. No constants are hidden in files or a learned library. The fixed external network readout is used as the contract specifies; its weights are not embedded in this program. If its 160 weights are nevertheless charged to this account, the total is **331**, still below 432. Official gzip size: **867 bytes**. The fenced block was extracted, parsed and imported, then checked against `ideal.py` for exact equality of all 16 outputs at 1,000 steps on 100 independent normal rows (NumPy seed **9041**).

The rotation and feature family came from the earlier modal/clock findings. Least-squares state fits initialise the emission and threshold coefficients; the final coefficients were optimised on **training rows `[0,32768)`**, with seed **371**, Adam step size **0.01**, minibatches **2048**, **4001 updates**, final-step task cross-entropy plus **0.02 times mean squared hidden-state error**. No test or eval values were fitted. Development comparisons used subsets of `[131072,180224)`; the final emission was rounded to half-integers and threshold coefficients to hundredths. These threshold coefficients are consequently **task-calibrated ideal coefficients**, not a claim of a numerically exact fit to the network's comparator thresholds.

The fixed final program was then measured on **19,776 training-distribution rows `[180224,200000)`**, not used in fitting or choosing its rounding. Classification uses the real, unchanged `W_out`; the local reference forward is fp32 and its argmax agreed with the exported GPU target on all 19,776 rows (maximum absolute logit difference 0.000393). “Agreement” means equal argmax; “error reproduction” means the **same wrong answer on a row the network gets wrong**, divided by the network's error count.

| measure | Account 1 |
|---|---:|
|task accuracy|**96.511%** (19,086/19,776)|
|network's task accuracy on these rows|95.292% (18,845/19,776)|
|agreement with network|**93.249%** (18,441/19,776)|
|network errors reproduced|**6.015%** (56/931)|
|state variance explained|0.80510|
|pooled Pearson r|0.90526|
|official 1,024-row eval: variance explained / r|**0.80613356 / 0.90530217**|

For the single-unit test, predict the ideal trajectory independently, then run the real RNN while replacing **only unit i at every step** by that trajectory's unit i. The other 15 units remain recurrently computed by the real network. Agreement percentages, units 0–15:

|unit|0|1|2|3|4|5|6|7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|agreement|63.6|35.2|67.7|90.9|95.5|74.1|39.4|33.6|

|unit|8|9|10|11|12|13|14|15|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|agreement|48.5|97.5|70.7|67.5|68.7|83.3|71.6|82.5|

The poor single-unit results, especially for comparator units, are important: the ideal emitter and its task-calibrated thresholds compensate one another. This is a compact task account, **not** a claim that the network's calibrated unit formulas have been individually recovered.

### What it gets wrong

It forgets displaced contenders completely, uses exact nonleaking levels, ties the harmonics, ignores almost all small input-drive terms, and replaces clipped/damped phase history by exact reset clocks. It also floors both order statistics at zero. On the final measurement rows, accuracy was **97.453%** when there were at least two positive inputs (19,553 rows), but only **13.527%** with exactly one positive input (207 rows), and **18.75%** with none (16 rows). The latter two groups lack the negative runner-up's location in this ideal state.

It corrects **701** of the network's **931** errors, but introduces **460** errors on rows the network answers correctly. With runner-up minus third-largest below **0.05** (2,397 rows), network/ideal accuracy is **77.764% / 90.822%**; with both top gaps at least **0.05** (16,082 rows), it is **98.445% / 97.326%**. Thus the ideal formula is better near some ties but worse on some well-separated cases. Of its 690 errors, 178 select the maximum, 129 select the third-largest, and 383 select ranks 4–10; the corresponding network counts are 198, 572, and 161. Its remaining errors are not simply the network's near-tie errors. The exact rank recursion is easy; a small smooth code into this particular neural readout is imperfect.

A concrete history witness (a constructed example, not a frequency estimate) is
`[1.2, 0.1, 0, 1.23, 1.26, 0, 0, 0, 0, 0]`
versus the same row with the second entry changed to `1.1`. Both have the same final top three, at the same positions; the changed entry was an earlier runner-up but is finally fourth in both. The fp32 network answers **3 versus 0** (zero-based; the true answer is 3 in both), with final hidden-state maximum difference **2.606772** and class-3 minus class-0 margins **+2.746696 versus −2.713192**. Account 1 emits exactly the same final state for both and answers 3. We found this witness by enumerating 1,512 constructed pairs (467 network-answer flips); this search is separate from the random-row performance measurements. Thus even a perfect top-three summary cannot, in general, reproduce this circuit's answer. The deterministic search used record positions `2≤i<j≤9`, gaps `{.001,.003,.01,.03,.1,.3}`, changed-entry pairs `{(0,1.19),(.1,1.1),(.5,1)}`, and backgrounds `{−.5,0,.5}`, with `x0=1.2`, `xi=1.2+gap`, `xj=1.2+2gap`; no random seed was used.

## Account 2: the circuit, with its calibration retained

The submitted file is **`fullsafe.py`**. It is a deterministic, precision-rounded modal rewrite, **not** a refitted task model. It separates ordinary units `O=(0,1,2,3,6,7,8,10,11,12,13,14,15)` from pulses `p=(h4,h5,h9)`. Its 13-dimensional carry `m` obeys

\[
f_t=\Lambda m_{t-1}+D p_{t-1},\quad
u_t=[Bf_t+d x_t]_+,\quad
p_t=[C m_{t-1}+F p_{t-1}+e x_t]_+,\quad
m_t=B^{-1}u_t.
\]

The emitted state places `u` and `p` back in their original unit slots. In the file, `B=SYNTHESIS`, `D=PULSE_DRIVE`, `C=PULSE_READ`, `F=PULSE_FEEDBACK`, `d=INPUT`, `e=PULSE_INPUT`; `POLES` specifies the real/rotation blocks of `Λ`. `ANALYSIS` is computed as the inverse of `B`, not another fitted table. Before quantisation this is a change of coordinates of the target recurrence. After quantisation it is an approximation tested below.

The basis comes from the eigensystem of the 13 ordinary units' recurrent submatrix. A pair's phase is chosen so the record kick lies along its positive first coordinate; column scaling makes each real column, or jointly each pair, have maximum absolute entry one. This fixes the numerical gauge used below. Coefficients are rounded to grids **B: 1/1024; D: 1/256; C: 1/1024; F: 1/256; d,e: 1/64; real and imaginary pole components: 1/2048**. Thus individual rounding errors are bounded by half these grid steps. No coefficient fitting or random seed is involved in this submitted rewrite.

It stores **285 coefficient slots** (169+39+39+9+13+3+13), or **334 numeric-literal occurrences including all structural constants**. Its official compressed size is **1,300 bytes**. The difference from the ideal account is not primarily expensive free dynamics: the four dense read/write/event blocks contain **256 of those 285 coefficient slots**.

### Independent strict-bar check

The precision was selected using the already available rows `[180224,200000)`; those are therefore **development/ablation rows for Account 2**, although they were unused validation rows for Account 1. After freezing `fullsafe.py`, we fetched a fresh **32,768 rows `[200000,232768)`**. Neither program was changed afterwards. The local fp32 reference and exported GPU target agreed in argmax on all these rows (maximum absolute logit difference **0.000328**). Both accounts are compared here on exactly these fresh rows.

| measure | Account 1 | Account 2 | strict bar for Account 2 |
|---|---:|---:|---:|
|task accuracy|96.51794% (31,627/32,768)|95.36743% (31,250/32,768)|not the fidelity criterion|
|agreement with network|93.29834% (30,572/32,768)|**99.609375% (32,640/32,768)**|≥98%|
|same network errors reproduced|4.53939% (68/1,498)|**96.06142% (1,439/1,498)**|≥80%|
|minimum single-unit agreement|33.19092%|**99.54529%** (unit 7)|≥95%|
|state variance explained|0.80592725|**0.99999327**|reported, not the causal bar|
|pooled Pearson r|0.90560420|**0.99999702**|reported|

The reference network is correct on **31,270/32,768 = 95.42847%**. The same single-unit replacement procedure as above gives Account 2:

|unit|0|1|2|3|4|5|6|7|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|agreement %|99.7711|99.6796|99.8260|99.8077|99.9146|99.8230|99.6735|99.5453|

|unit|8|9|10|11|12|13|14|15|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|agreement %|99.7192|99.8901|99.6887|99.7772|99.7681|99.8077|99.7742|99.7955|

On the official **1,024-row eval**, `eval(fullsafe.py)` reports **variance explained 0.9999933917, Pearson r 0.9999970875**, compressed size 1,300 bytes. The classification/intervention tables above are explicitly our independent training-distribution measurements; the official tool returns state fits, not those classification statistics.

### Which weights carry the ideal quantities?

The mode indices below are exactly the file's indices; an entry in column `k` of `B` tells how ordinary units read mode `k`, row `k` of `D` tells how the three pulses write it, and column `k` of `C` tells how it controls future pulses.

| quantity | full program coordinates / constants | value in submitted program | ideal account |
|---|---|---|---|
|maximum level|`m[0]`, first real pole|1.009765625|perfectly held level, pole 1|
|runner mixture|`m[1]`, second real pole|0.9736328125|perfectly held runner-up, pole 1|
|record clock|`m[2:4]`, first rotation block|radius 0.95090614, angle 42.752627°|radius 1, 42.398877°|
|second harmonic|`m[4:6]`|radius 0.93712621, angle 83.267490°|radius 1, 84.797754°|
|alternating history|`m[6]`|pole −0.92822265625|absent|
|third harmonic|`m[7:9]`|radius 0.83707814, angle 124.783932°|radius 1, 127.196631°|
|short-lived clock/history|`m[9:11]`|radius 0.73483927, angle 42.980534°|absent as an independent mode|
|short transient|`m[11]`|pole 0.23095703125|reset flags only|
|input echo|`m[12]`|pole 0.0009765625|no persistent echo|

The second and third harmonics are nearly a **41.6°** family (83.267/2 = 41.634°, 124.784/3 = 41.595°), distinct from the main clock's 42.75° (and the short-lived clock's 42.98°) and from Account 1's 0.74 rad = 42.40°. Tying them at exactly 2× and 3× 41.6° has negligible cost on these rows (table below: 99.65% agreement, 96.6% errors kept, versus baseline 99.65% and 96.7%), and moving the main clock alone to the ideal 42.40° costs 0.27 agreement points. Angular simplification is relatively cheap here; uncompensated changes to damping and history modes are much more damaging. This is not a claim that the precise damping constants are necessary under every possible recalibration.

These level coordinates have arbitrary linear scales, not numerical values equal to the order statistics. A descriptive regression on **4,000 rows `[0,4000)`**, all **40,000 steps**, gives

\[
m_0\simeq19.7955M-1.22693S-0.06739x\quad(\mathrm{VE}=0.99533),
\]
\[
m_1\simeq3.81090M+8.52406S+0.81503x\quad(\mathrm{VE}=0.91548).
\]

Here `m` is measured after the current step and `M,S` immediately before it: the fresh record is still partly in the current pulse. Thus **L=m0/19.8** is approximately `M−.062S`; **R=m1/8.52−.447L** is approximately `S+.1x`. In these semantic coordinates the free runner flow is approximately **`.97363 R − .01615 L`**, versus the ideal `R`. This recovers the earlier write-up's runner leak rather than assigning a literal maximum or runner-up to a raw eigenvector coordinate.

The pulse correspondence is also concrete. Full input drives into `(h4,h5,h9)` are **`(10.15625, −.28125, −.234375)`**, versus the ideal `(10,0,0)` plus its explicit delayed-event formulas. The full pulse self-feedback diagonal is **`(−.984375, −.421875, −.554688)`**. The full contender and coincidence read all memory modes through `C`; neither is literally the ideal `12e` or `1.5Mr`. In `D`, the main clock's record kick is **`(1.6640625,0)`**, the second harmonic's is **`(.97265625,0)`**, and the third's is **`(1.29296875,0)`**, per unit of raw `h4`. The short-lived clock receives record and contender kicks **`(2.203125,0)`** and **`(−1.09765625,−.734375)`** respectively. This is where old contender evidence can continue rotating after a contender has ceased to be a top order statistic.

The ideal harmonics are powers of one current runner-up phase. **The full harmonic pairs are independent states**: their amplitudes and phases can superpose multiple event histories. Resetting a literal rank clock is therefore an abstraction of the full `D` kicks plus ordinary-unit clipping and `B^-1` writeback, not an entrywise approximation to a single matrix.

### Threshold map, including the numerical disagreement with the ideal emitter

In the full recurrence the ordinary comparator threshold is exactly

\[
\theta_j=(B_j f)/|d_j|\quad(j=1,6,7,8).
\]

The record threshold is `−(C0 m + F0 p)/e0`. Here `a` is the maximum’s age at the end of the preceding step, so `a=0` immediately after a record, exactly matching the ideal program’s pre-update clock. Regressing these pre-clipping thresholds on the *same* simple clock features used by Account 1, on the same 40,000 steps just specified, gives

\[
\theta_j\simeq c_S S+(M-S)(A+B_c\cos(.74a)+B_s\sin(.74a)+D_r\mathbf1_{a=0}).
\]

|unit|cS|A|Bc|Bs|Dr|R²|
|---|---:|---:|---:|---:|---:|---:|
|1|.96836|.67529|−.68288|−.31575|.85141|.99342|
|6|.96741|.74086|−.33181|.31559|−.45556|.99518|
|7|.98818|.79925|.13043|.19515|.09420|.99852|
|8|1.01319|.71078|.13694|−.71802|−.87921|.99305|
|4, record|.97259|.82021|.09253|.13823|.07732|.99897|

Full comparator gains are **10.5625, 11, 13.171875, 12.3125**, close to the ideal 10.5, 11, 13, 12. But the threshold-wave coefficients are **not all close**: e.g. unit 1's cosine coefficient is −.683, not the ideal −.22, and unit 8's sine coefficient is −.718, not −.32. Those are task-adapter compensations in Account 1, not rounding errors. Returning to the real network requires joint threshold and emission calibration. The high R² says the clock/level language is descriptive; it does not license independently replacing comparator units with these regressions or with the task-trained ideal coefficients.

We checked the causal importance of that small descriptive residual separately. Using the **exact network** (not the rounded recurrence), we refitted the same five-feature threshold form on **rows `[0,20000)`**, all **200,000 steps**, by deterministic least squares. Threshold R² values for units 1,6,7,8,4 were **.9935, .9951, .9985, .9931, .9990**. We then replaced only the indicated threshold-generated unit values at every step, with the exact input gains, leaving the other units recurrently computed by the real network. On the 19,776 development rows:

| fitted threshold substituted | agreement % | task accuracy % | errors kept % |
|---|---:|---:|---:|
|unit 1|81.40|80.48|52.0|
|unit 6|88.42|87.27|54.9|
|unit 7|82.74|81.18|57.7|
|unit 8|77.69|76.87|50.7|
|unit 4, record|99.21|95.40|90.0|
|all four negative-input comparators|68.27|68.06|41.2|

The record comparison is nearly ideal; the other comparators are not independently replaceable despite their excellent threshold fits. This measures the importance of residual calibration/history, without claiming that all residual error belongs to a single mode or only occurs near ties.

`EMIT` in Account 1 and `B` here also have different coordinates and timing, so there is no honest entrywise equation `B=EMIT+small noise`. The correspondence is between their roles: reading levels and age waves into units. The additional full columns, pulse reads/writes, and clipped writeback supply the corrections in the next table. This is a map of the implementation onto the ideal algorithm, not a claim that the ideal program is obtained by simply deleting a handful of coefficients.

### Named corrections, and how much they matter

Every row below changes the indicated part of **the frozen submitted full program**, with the other constants held fixed and **no refit**. All use the **19,776 development/ablation rows `[180224,200000)`**, with **931 network errors**. “Errors kept” is the same wrong-answer reproduction fraction used for the bar. Ablations are not additive and do not uniquely assign a semantic role to every individual matrix entry.

| correction group | idealisation/removal | agreement % | task accuracy % | errors kept % |
|---|---|---:|---:|---:|
|full baseline|none|99.65|95.23|96.7|
|level calibration|both level poles →1|88.79|87.39|58.8|
|record-age damping|main clock radius →1|92.24|90.43|64.0|
|frequency calibration|main clock angle alone 42.75° →42.40° (Account 1)|99.38|95.23|93.3|
|frequency calibration|harmonics tied at exactly 2×, 3× 41.6° (83.2°, 124.8°)|99.65|95.24|96.6|
|frequency calibration|harmonics tied to 2×, 3× the main clock's 42.75°|97.70|94.47|81.2|
|frequency calibration|all three angles →Account 1's tied angles|98.28|94.81|84.4|
|harmonic envelope calibration|r2→r1², r3→r1³|98.26|95.01|82.4|
|contender-history corrections|alternating mode's pole →0|91.32|89.19|68.7|
|contender-history corrections|short-lived clock's block →0|50.92|50.43|36.6|
|recent-event correction|short transient's pole →0|89.48|87.86|58.9|
|echo persistence|echo pole →0|99.61|95.24|96.0|
|threshold-history calibration|comparators read only the two level modes and main clock|36.01|36.18|22.8|
|threshold gain calibration|gains →10.5,11,13,12, keeping immediate thresholds fixed|89.39|87.43|65.4|
|low-input tail|zero wave-unit input drives except unit 14|97.36|94.43|77.9|
|low-input tail, broad wave too|zero all nine wave-unit input drives|89.46|88.18|54.6|
|nonlinear reset/history writing|do not write wave-unit clipping back into memory|60.12|59.20|40.5|
|coincidence correction|set unit 9's pulse to zero every step|67.81|67.00|41.8|
|contender event writing|set unit 5's pulse to zero every step|21.33|20.58|29.8|

Operational details matter. Dropping a mode here sets its free-flow pole/block to zero; it can still receive an immediate pulse or clipping write. For the comparator-read ablation, masked `B` rows are used **only to read**; the original inverse writeback is retained. For rounded comparator gains, the corresponding read rows are rescaled along with the input gain so that the immediate threshold remains unchanged, while the emitted gain and subsequent feedback change. For the clipping-write ablation the emitted waves remain nonnegative, but their unclipped preactivations, rather than clipped values, are written into memory. The wave-unit set is `{0,2,3,10,11,12,13,14,15}`. These definitions avoid treating a changed coordinate system as if it were a removed causal pathway.

The first three groups correspond to the ideal's exact levels and perfect tied clocks. The history groups are what the ideal's literal top-two state discards. The threshold/gain and low-input groups calibrate small logit margins, particularly near ties; this is consistent with, but not proved solely by, the aggregate ablation scores. The full program also preserves the constructed history witness's **3 versus 0** answers, with class-3 minus class-0 margins **+2.45215 versus −3.34993**; the ideal erases that distinction.

For the history witness, an exact-modal decomposition of the final **5.46** difference between the two class-3 minus class-0 margins locates the lost evidence: the short-lived clock contributes **+6.46**, the second harmonic **+0.90**, third harmonic **+0.64**, main clock **−1.11**, alternating mode **−1.28**, transient **−0.11**, level **−0.007**, and runner mode **−0.03**; pulse contributions are zero (rounding accounts for the displayed sum). This is a descriptive decomposition in the stated gauge, not a mode ablation. It links the displaced runner-up specifically to history waves missing from the ideal state, rather than to a changed final maximum or runner-up value.

A useful distinction is **removal without compensation versus necessity in every possible implementation**. In a separate experiment, both level poles were fixed at exactly 1 and all remaining dense constants refitted on rows `[0,100000)`: on the common 19,776 development rows this recovered **98.473% agreement, 86.144% errors reproduced, and 97.492% minimum single-unit agreement**. Fixing levels to 1, the main clock radius to 1, all three angles to the ideal values, and the echo pole to 0, with the same refitting protocol, reached only **96.486%, 72.932%, and 94.782%**. These float refits used seed **0**, **1,500 Adam steps**, batch **4,096**, learning rate **0.0001**, with state-error and joint/single-unit output-KL losses; they are **not** the submitted program. Thus the level detuning can be compensated by dense recalibration. Failure of the more extensive idealisation in this search is not a proof of impossibility.

Two further refits (trained on rows `[0,100000)`, scored on rows `[100000,120000)`, seed 0, Adam learning rate 0.0001 with cosine decay to 0.1×, batch 4,096, same losses) test whether the dense blocks can be made cheap. (a) Levels at 1, echo at 0, and all angles tied to one fitted fundamental (θ = 42.625°; second harmonic 2θ, third 3θ, short-lived clock θ), radii free, 3,000 float steps: **98.08% agreement, 82.7% errors kept, 97.1% minimum unit**, so the bar is met in floating point. But rounding it in stages to coarse grids (B and C 1/128; D, F, d, e 1/32; poles 1/1024 and 1/16°), refitting the still-unrounded constants after each stage, gives **96.67% / 72.2% / 95.9%**, below the bar. The stages, in order, were: poles (200 steps); F, e, d (200); D (200); C (300); then the columns of B in nine groups — level, runner, main clock, second harmonic, alternating, third harmonic, short-lived clock, transient and echo — 300 steps each. (b) The exact poles, rounded and refitted in the same stages to the same coarse grids (after a 300-step float warm-up): **97.81% / 77.9% / 97.6%**, also below the bar. For comparison, plain rounding without refits of the exact rewrite in the same gauge to B 1/256, D 1/64, C 1/256, F, d, e 1/64, pole entries 1/512 gives 98.23% / 83.4% / 98.4% on those rows (barely passing), and the submitted grid gives 99.6% / 96.8% / 99.6% there. So in our searches the 1/128 grids failed even with staged refitting, and only grids of about 1/256 or finer on the read map passed; this does not prove that every lower-precision representation must fail.

Account 2 is not a compression of the network: its 285 coefficient slots exceed the network's own 272 input and recurrent weights (the modal gauge adds 13 poles). What it adds is coordinates in which the levels, clocks, harmonics, history modes and pulses are separate constants that can be compared with Account 1 and ablated one group at a time.

## What remains unexplained

The two-level/age-code account explains the computational organisation, not why training chose these particular event-write and unit-read coefficients. The ideal emission is a fitted interface to the fixed readout, not a derivation from a minimal sorting circuit. Account 2 preserves causal fidelity but retains dense calibration. Its correction groups are interventions on named parts of the recurrence, not an assertion that all their entries have a unique semantic role; modal gauges mix meanings, and several groups can affect the same near-tie decision. Neither a good state fit alone nor good task accuracy alone establishes a faithful mechanistic explanation.