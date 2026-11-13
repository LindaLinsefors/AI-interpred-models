# Code for the tiny AlgZoo networks

Each script holds a network's trained weights, runs the network, and prints the numbers quoted in
`../explainer.html` and `../paper.pdf`. Needs Python 3 and numpy (matplotlib for the figures). Seeded, so the
output is reproducible; each takes a few seconds.

| File | What it does |
|---|---|
| `median_2_10.py` | median_2_10 seed 0: network vs. the one-line median-update rule, regime shares, intervention test |
| `second_argmax_3_3.py` | 2nd_argmax_3_3 seed 1: network vs. the 12-line region rule, region table, tie-breaking, float32, aliasing, the knife-edge margin and weight sensitivity |
| `argmedian_16_5_checks.py` | argmedian_16_5 seed 0: accuracy and the partial facts listed in section 3 (stored inputs, comparator neurons) |
| `make_figures.py` | redraws `../figures/*.png` and `../figures/*.svg` (the SVGs are vector copies of the same figures) |
| `explainer_src.html`, `build_explainer.py` | source of `../explainer.html`; the build inlines the PNG figures |

The paper's source is `../paper.typ` (Typst) with references in `../refs.yml`. To rebuild it:
`python -c "import typst; typst.compile('paper.typ', output='paper.pdf')"` from the parent folder (`pip install typst`).
