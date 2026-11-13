"""Inlines the figures into explainer_src.html and writes ../explainer.html (one self-contained file)."""
import base64, os
here = os.path.dirname(os.path.abspath(__file__))
fig = lambda n: base64.b64encode(open(os.path.join(here, "..", "figures", n), "rb").read()).decode()
html = open(os.path.join(here, "explainer_src.html")).read()
html = html.replace("{{MEDIAN_FIG}}", fig("median_mechanism.png")).replace("{{ARGMAX_FIG}}", fig("second_argmax_map.png"))
open(os.path.join(here, "..", "explainer.html"), "w").write(html)
print("wrote explainer.html,", len(html) // 1024, "KB")
