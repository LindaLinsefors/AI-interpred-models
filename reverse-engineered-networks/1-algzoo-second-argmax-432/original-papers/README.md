# Original working papers

These are the two internal working papers that this folder's explainer and short paper are built on. They are drafts, written for our own team, and they use some team-specific vocabulary (for example "VPD", the weight-decomposition method, and "explain2", the program-writing agent system). The short paper in the parent folder states the same results without that vocabulary.

- `decomposition-route_second-argmax-paper.pdf` (20 pages, version of 18 October 2026): "How ARC's 432-parameter RNN finds the second-largest input: comparators, a recorder and a rotating tape". The weight-decomposition route. It assigns all 16 units to roles and analyses near-tie errors in detail.
- `agent-program-route_second-argmax-paper-v3.pdf` (17 pages, draft of 28 October 2026): "How ARC's 432-parameter RNN finds the second-largest input, and why its mistakes have resisted a compact account so far". The program-writing route. It contains the idealised and full accounts and the search for a compact faithful account.

## Why their numbers differ slightly from the short paper's

The short paper and explainer report our own run of `code/verify.py` on 131,072 fresh inputs (seed 1). The working papers used other samples and conventions, so some numbers differ:

- The idealised account is "170 numbers" in the program-route paper and 171 numeric literals in our count; the difference is counting convention.
- Its accuracy (96.6% vs our 96.5%) and mistakes kept (6% vs our 5.2%) come from the authors' own test samples (131,072, 32,768 or 19,776 inputs in different runs).
- "Its comparators put in place of the network's keep 28% of its answers" appears to mean replacing the comparators as a group (an earlier measurement of the same kind gave 31%); our 33–49% replaces one comparator at a time.
- The full account's 95.6% of mistakes kept is on other inputs; ours is 96.1%.
- The decomposition-route paper groups the units differently (five comparators including unit 4, a recorder, two timers, eight memory units); the short paper's Table 3 maps its roles onto ours.
