"""Precision-rounded modal recurrence: levels, clock, candidate-history modes and three event pulses. Dense read/write calibration is retained."""
import numpy as np
PREDICTS = ('algzoo-m16-10/h',)
READS = ()
ORDINARY = [0,1,2,3,6,7,8,10,11,12,13,14,15]
PULSE_UNITS = [4,5,9]
GROUPS = [(0,), (1,), (2,3), (4,5), (6,), (7,8), (9,10), (11,), (12,)]
SYNTHESIS = np.array([[871, -1024, -739, -310, -290, 10, -157, -220, 226, 253, -232, 117, -35], 
                      [329, 334, -461, -142, -74, -86, 8, -9, -27, 2, -191, 369, 786], 
                      [646, 563, 27, 268, -13, -21, -73, -81, -4, -83, -37, 253, -20], 
                      [120, 216, 122, -211, -228, 1024, 952, 1014, 1024, -418, 365, -215, 57], 
                      [407, 273, -148, 236, -65, 32, -43, -55, 29, -85, -33, -159, 873], 
                      [509, 315, 135, 143, 29, -2, 3, 6, -24, -50, 16, 12, 1024], 
                      [455, 265, 76, -565, 71, -17, 9, 13, -18, 308, 82, -772, 1024], 
                      [683, -703, 624, 729, 151, 561, -421, 276, -262, -202, -320, -174, -3], 
                      [734, -794, 818, -94, 612, -6, 397, -147, -6, -66, 53, -515, 21], 
                      [1024, -924, 529, -1024, -359, -80, -32, 369, 259, -189, 302, -464, 27], 
                      [277, 210, -11, 111, -778, 732, 1024, 118, 178, -511, -50, 527, -9], 
                      [793, -719, -879, -293, 547, 86, 464, 451, -112, 1024, 106, -1024, 123], 
                      [524, -196, 144, -789, 543, 1023, -664, -173, -636, -171, 548, -325, 49]], dtype=float) / 1024
PULSE_DRIVE = np.array([[517, -37, -65], [42, 73, -19], [426, -23, 387], [0, -21, -30], [249, 19, -76], [0, -216, -100], [68, 75, 101], [331, 185, 201], [0, -248, -163], [564, -281, 211], [0, -188, 228], [659, -113, 142], [22, -15, -6]], dtype=float) / 256
PULSE_READ = np.array([[-403, -219, -98, -11, -8, -24, -13, -16, 3, 15, -33, -1, -2], [259, -1450, -24, 35, 186, -54, -128, -105, -162, 16, -84, 158, -1441], [-3090, -2152, -588, -75, -69, 7, -26, -20, 80, 42, -103, 513, -6616]], dtype=float) / 1024
PULSE_FEEDBACK = np.array([[-252, 0, 21], [304, -108, 76], [37, 11, -142]], dtype=float) / 256
INPUT = np.array([4, -676, 1, -8, -704, -843, -788, 4, 9, 19, -3, -85, -8], dtype=float) / 64
PULSE_INPUT = np.array([650, -18, -15], dtype=float) / 64
POLES = np.array([2068, 1994, 1430, 1322, 225, 1906, -1901, -978, 1408, 1101, 1026, 473, 2], dtype=float) / 2048
ANALYSIS = np.linalg.inv(SYNTHESIS)

def step(carry, prefix, upstream):
    if carry is None:
        carry = (np.zeros(13), np.zeros(3))
    memory, pulses = carry
    x = float(prefix.values[prefix.position])
    flow = np.empty(13)
    k = 0
    for group in GROUPS:
        i = group[0]
        a = POLES[k]
        k += 1
        if len(group) == 1:
            flow[i] = a * memory[i]
        else:
            j = group[1]
            b = POLES[k]
            k += 1
            flow[i] = a*memory[i] - b*memory[j]
            flow[j] = b*memory[i] + a*memory[j]
    flow += PULSE_DRIVE @ pulses
    units = np.maximum(SYNTHESIS @ flow + INPUT*x, 0.)
    pulses = np.maximum(PULSE_READ @ memory + PULSE_FEEDBACK @ pulses + PULSE_INPUT*x, 0.)
    memory = ANALYSIS @ units
    h = np.empty(16)
    h[ORDINARY] = units
    h[PULSE_UNITS] = pulses
    return (memory,pulses), {PREDICTS[0]:h.tolist()}
