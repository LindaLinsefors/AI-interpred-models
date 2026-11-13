"""How algzoo-m16-10 finds the position of the second-largest of ten inputs: its exact
recurrence, rewritten in 13 named memory quantities plus 3 pulses.

THE ALGORITHM IN ONE PARAGRAPH
The network keeps (i) the running maximum, as a `level`; (ii) the runner-up, as a
second, slowly leaking level; (iii) a `clock`, a damped 2-d rotation that every
new record clips and re-kicks, so its phase reads roughly as the age of the
maximum; and (iv) a register of eight candidate `score` contrasts that shifts one
place per step. Each input x is compared with thresholds read off this memory:
unit 4 fires when x beats the maximum (a record), unit 7 measures how far x falls
below the maximum, and units 1, 6, 8 how far it falls below three thresholds that
swing with the clock. A low input leaves no
trace: the four below-threshold units share one input direction, and its effect on
the next step cancels (leftovers within +-0.5 of their 10-13 input weights). An input
that clips some of them, one near the top, writes evidence into the memory, tagged by
its age. The RNN is the same at every step, so it tracks ages, not indices; at the
last step age a is position 9 - a, and the fixed readout turns each register entry
into the logit contrast of two adjacent positions.

THE NAMED QUANTITIES (the carry after step t; fits on 4,000 training rows, with
M+ and S+ the positive parts of the max and runner-up of x_0..x_{t-1})
  level     ~ 1.00 M+ - 0.06 S+ (VE .994): the max of the earlier inputs. Pole 1.0099.
            The current step's record pulse holds the fresh increment.
  runner    ~ 1.00 S+ + 0.10 x_t (VE .80). Pole .9735, minus a .0159 * level leak.
  clock     turns 42.75 degrees per step at radius .951 (period 8.42 steps). A record
            zeroes it and the record pulse restarts it: median phase 63, 95, 139, 193,
            226, 269, 310, 352, 34 degrees at ages 1..9 of the max; amplitude ~ .57.
  echo      the current input as clipped by the units: ~ x_t for ordinary inputs (VE
            .88), capped for records. Pole .001: gone by the next step.
  score[k]  ~ logit_k - logit_{k+1} if the readout read this state now (r = .91-.98,
            slope 1.0-1.3 for k = 0..6; score[7], the refill slot, r ~ .5). The free
            flow moves score[k+1] into score[k]; TAIL refills score[7].
  pulses    record = unit 4 / 10.157 = max(x - theta, 0), theta ~ .85 level + .18 runner
            + .99 previous record (+ clock ripple), i.e. the running max: on for
            records. contender = unit 5 / 12: on the step after a record or a new
            runner-up. coincidence = unit 9 / 1.5: a narrow correction the step after
            a record. A low previous input shuts both off (its leftover drive into
            units 5 and 9 is -18 and -84 per unit of drawdown).

HOW EACH UNIT IS DERIVED. The 13 ordinary units are max(SYNTHESIS . flow + w x, 0),
and the named quantities are exactly their linear re-coding (memory = ANALYSIS .
units). So this program is the network's recurrence, organised by what it computes:
  7         13.17 max(theta7 - x, 0), theta7 ~ the running max (sagging toward the
            runner-up as the clock turns): the drawdown below the max; off for records.
  1, 6, 8   10.56, 11.00, 12.31 max(theta - x, 0), thresholds that swing with the
            clock around the max. Minus unit 7 they are clipped waves
            min(theta7 - theta, theta7 - x): memory-made when x is low, x-made when x
            lands between the thresholds.
  2         ~ 17.3 x the max of the earlier inputs (positive part), always on.
  10,11,12  record-age waves (12 -> 11 -> 10 delay chain); at t = 9 unit 10 is ~0
            exactly when the max is 6 steps old, unit 11 at 5, unit 12 near 0 at 4.
  14        a weak comparator (gain 1.32) with a far threshold: in effect a broad
            record-age wave.
  0,3,13,15 clipped candidate and clock waves (0 and 15 really need their ReLU).
  4, 5, 9   the record, contender and coincidence pulses.

HOW THE READOUT ANSWERS (logits = W_out h_9)
  class 9 (the newest input) ~ +45.9 echo - 41.5 runner - 50.7 record (level only
     +.93, clock -6.1 cos + 1.6 sin): x_9's logit grows with x_9 against the
     runner-up level and is knocked down if x_9 is a new record.
  class 8: the contender pulse adds +40.8 when x_8 was near the top.
  older classes: memory alone (every unit but 1, 4, 6, 7, 8) votes for the current
     max's position, top among classes 0-8 in 88-99% of rows. At the last step the
     comparators 1, 6, 7, 8, whose thresholds come from that same memory, put their
     most negative vote on that position (76-86% of non-record rows), and their
     clipped waves bring the runner-up's evidence. If x_9 is a new record they are
     silent (88% of those rows) and memory's vote for the old max, now the
     runner-up, stands.
  The network is right on 95.2% of rows. Its errors name the third-largest (63%) or
  the largest (21%), mostly when the top three lie within 0.05 of each other.

HOW MUCH IS MESS (mean KL to the network's output; this program ~1e-6, from its
5 significant digits). The story above reads a finely tuned circuit, and none of its
calibration can be dropped:
  pure shift register (TAIL = 0): KL 6.0, agreement 59%.
  no clipping feedback into the score register: KL 16; into the clock: KL 253.
  no coincidence pulse: KL 1.8.
  unit 2 replaced by its ideal 17.31 M+: KL .98. The +-0.4 drawdown leftovers into
  units 10-13 zeroed: KL .34. Hidden weights under .02 pruned: KL 1.1. Constants at
  3 significant digits: KL .06 (4 digits: 1e-4).
  Harmless: units 2, 12, 13 never need their ReLU (KL .002).
  Noise of 0.3 per unit on the last state alone costs KL .26. No summary by top-two
  statistics comes close: the best per-position fit of the final logits from x_j, the
  max, runner-up and third gives KL 1.8.
"""
import numpy as np

PREDICTS = ('algzoo-m16-10/h',)
READS = ()


def step(carry, prefix, upstream):
    x = float(prefix.values[prefix.position])
    if carry is None:
        carry = (np.zeros(13), np.zeros(3))
    memory, pulses = carry
    score, clock = memory[:8], memory[8:10]
    echo, runner, level = memory[10], memory[11], memory[12]

    # 1. Free flow: what the memory becomes if nothing new happened.
    flow = np.r_[
        score[1:], TAIL @ score,       # candidate scores shift one place toward class 0
        ROTATE @ clock,                # record-age clock turns 42.75 degrees
        ECHO_POLE * echo,              # the previous input is forgotten
        RUNNER_POLE * runner + RUNNER_LEAK * level,  # runner-up level, slow leak
        LEVEL_POLE * level,            # running-max level, kept
    ] + PULSE_DRIVE @ pulses           # last step's record/contender pulses kick it

    # 2. Ordinary units: each reads its own mixture of the flowed memory (its
    # baseline) and compares the input with it.  For the comparators 1, 6, 7, 8
    # (input weight -|w|) this is |w| * max(threshold - x, 0) with threshold =
    # baseline / |w|: unit 7's threshold is about the running max, so unit 7 is
    # the drawdown of x below the max; 1, 6, 8 have thresholds that swing with
    # the clock.  The other nine have small input weights (|w| <= 1.32): they are
    # clipped memory waves.
    baseline = SYNTHESIS @ flow
    units = np.maximum(baseline + INPUT_DRIVE * x, 0.0)

    # 3. Pulse units read the previous step's memory.  record = max(x - θ, 0)
    # with θ ≈ .85 level + .18 runner + .99 previous record (≈ the running
    # max): x beats the max.  contender fires the step after a record or a new
    # runner-up; coincidence is a narrow one-step correction.
    pulses = np.maximum(PULSE_READ @ memory + PULSE_FEEDBACK @ pulses + PULSE_INPUT * x, 0.0)

    # 4. Write back.  Only the input drive and the clipping of units at zero
    # move memory off its free flow (equivalently memory = ANALYSIS @ units).
    memory = flow + ANALYSIS @ (units - baseline)

    state = np.zeros(16)
    state[ORDINARY] = units
    state[PULSE_UNITS] = PULSE_SCALE * pulses
    return (memory, pulses), {PREDICTS[0]: state.tolist()}

# ---------------------------------------------------------------- constants
ORDINARY = [0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15]
PULSE_UNITS = [4, 5, 9]
PULSE_SCALE = np.array([10.157, 12, 1.5])  # unit = scale * pulse

# Free flow of memory (poles of the network's ordinary dynamics).
ROTATE = np.array([[0.69828, -0.64543], [0.64543, 0.69828]])  # 42.75 deg/step, radius .9509
ECHO_POLE = 0.0009887
RUNNER_POLE, RUNNER_LEAK = 0.97349, -0.0159
LEVEL_POLE = 1.0099
# score7's refill from the register (the shift's one feedback tap).
TAIL = np.array([0.071304, -0.29446, 0.0036672, -0.22344, -0.10721, -0.46267, -0.66761, -0.35818])

# Input weight of each ordinary unit (comparators 1,6,7,8: large and negative).
INPUT_DRIVE = np.array([0.069195, -10.565, 0.014607, -0.12265, -10.996, -13.169, -12.312, 0.05985, 0.148, 0.29801, -0.047013, -1.3207, -0.11911])

# Each ordinary unit's reading of the flowed memory; columns:
# score0..score7, clock_c, clock_s, echo, runner, level.
SYNTHESIS = np.array([
    [0.025126, -0.015886, -0.069666, -0.01258, 0.014682, 0.026488, 0.096851, 0.015629, -1.0542, -13.173, 0.35791, -8.7441, 13.021],  # unit  0: clipped candidate/reset wave
    [0.055508, -0.003885, 0.018948, -0.0029429, 0.056933, 0.089533, 0.046447, 0.15122, -1.4011, -7.8195, -8.1167, 2.8532, 7.6145],  # unit  1: deficit comparator, gain 10.56
    [0.035173, 0.0054585, 0.027134, 0.024335, 0.037982, 0.04832, 0.036646, 0.06046, -3.7241, 2.4203, 0.20604, 4.8035, 14.591],  # unit  2: delayed max, ~17.3*level
    [-0.092694, 0.26264, -0.028841, 0.047135, 0.14215, -0.0068892, 0.14891, 0.033719, 4.0112, 0.19017, -0.58427, 1.8409, 3.1201],  # unit  3: older-candidate wave
    [-0.02278, 0.0030779, -0.027747, -0.022538, -0.07426, -0.091501, -0.061693, -0.12232, -4.5688, -0.38227, -9.0082, 2.3279, 8.8852],  # unit  6: deficit comparator, gain 11.00
    [0.00058885, 0.0026949, 0.0095153, 0.0030649, -0.003263, -0.0043865, -0.011009, -0.0031709, -1.0706, 3.0502, -10.568, 2.6884, 11.013],  # unit  7: drawdown below max, gain 13.17
    [-0.09989, -0.04608, -0.092998, -0.066094, -0.14333, -0.171, -0.10569, -0.2331, 8.8479, -3.1583, -10.566, 2.2638, 9.7818],  # unit  8: deficit comparator, gain 12.31
    [-0.025183, -0.00056803, 0.025136, -0.15854, -0.13366, -0.12854, -0.082813, -0.057375, -5.9629, 14.647, 0.035369, -6.0026, 10.582],  # unit 10: record-age wave (<- 11)
    [-0.070856, -0.017346, -0.044768, 0.0026473, -0.19069, -0.18585, -0.11683, -0.15949, 7.5656, 11.27, -0.21187, -6.7807, 11.231],  # unit 11: record-age wave (<- 12)
    [-0.081929, 0.06132, -0.039437, -0.018066, -0.0097315, -0.16139, -0.12731, -0.24061, 18.995, 0, -0.27712, -7.8906, 16.354],  # unit 12: record-age wave
    [0.042458, 0.15219, 0.023766, 0.060313, 0.031959, 0.18212, 0.043239, 0.048488, -1.7164, 0.67754, 0.089594, 1.7944, 6.1473],  # unit 13: runner-up/clock wave
    [-0.12667, -0.093402, -0.10236, -0.10188, -0.037129, 0.0053207, 0.038129, 0.084956, -2.3591, -15.093, -1.2646, -6.137, 12.653],  # unit 14: broad record-age wave, gain 1.32
    [-0.05467, -0.015761, 0.16467, 0.035386, -0.059464, 0.030188, 0.057773, -0.17405, 12.647, -3.8523, -0.50466, -1.6769, 9.3948],  # unit 15: older-candidate wave
])
# The named quantities are this linear re-coding of the 13 ordinary units.
ANALYSIS = np.linalg.inv(SYNTHESIS)

# Pulses read the previous memory (same columns), the previous pulses and x.
PULSE_READ = np.array([
    [0.00013131, -0.00045318, -0.0006618, -0.00060119, -0.00068882, -0.00067459, -0.00037607, -0.00018361, -0.056722, -0.1498, 0.0018489, -0.18372, -0.84763],  # record
    [0.0024335, -0.0025152, 0.0026094, -9.0941e-05, 0.00020769, 0.0029272, 0.0013415, 0.0062337, -0.058032, -0.0077254, 1.2393, -1.0319, -0.032626],  # contender
    [0.047765, 0.0087689, 0.02504, 0.026704, 0.089206, 0.109, 0.094446, 0.14724, -2.2248, -6.1177, 45.519, -12.252, -45.186],  # coincidence
])
PULSE_FEEDBACK = np.array([
    [-0.98561, 0.0020027, 0.012315],  # record
    [1.0047, -0.42111, 0.03717],  # contender
    [0.98733, 0.34288, -0.55285],  # coincidence
])
PULSE_INPUT = np.array([1, -0.023442, -0.15171])
# How last step's pulses (record, contender, coincidence) kick each memory quantity.
PULSE_DRIVE = np.array([
    [-6.0198, 0.050837, -8.1037],  # score0
    [-23.578, 8.4615, -2.1885],  # score1
    [-10.168, 12.591, -2.5691],  # score2
    [2.4162, 2.2716, 0.93772],  # score3
    [7.1331, 0.48768, 4.4533],  # score4
    [-1.6661, -7.86, 5.2507],  # score5
    [6.6355, -65.819, -5.7584],  # score6
    [56.287, 28.344, 1.2535],  # score7
    [0.45897, 0.0225, 0.07078],  # clock_c
    [0.88905, -0.085028, 0.11471],  # clock_s
    [-0.081428, 0.064745, 0.0033719],  # echo
    [-0.26098, 0.42939, -0.0041027],  # runner
    [1.0368, -0.088647, -0.019332],  # level
])
