"""An ideal positive top-two recursion with reset clocks, 
ideal pulses and a task-calibrated harmonic emission. 
Not a high-fidelity neural recurrence."""

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
