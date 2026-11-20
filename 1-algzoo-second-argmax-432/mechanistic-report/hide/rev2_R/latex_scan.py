"""Reviewer R: read-only scan of report.md for LaTeX that may not render in a Markdown preview."""
import io
import re
from pathlib import Path

REPORT = Path(__file__).resolve().parent.parent.parent / "report.md"
lines = io.open(REPORT, encoding="utf-8").read().split("\n")
D = chr(36)                      # the dollar sign
inline = re.compile(re.escape(D) + "[^" + re.escape(D) + "]*" + re.escape(D))

for i, l in enumerate(lines, 1):
    s = l.replace(D + D, "")
    if s.count(D) % 2:
        print("ODD number of dollars, line", i, "|", l[:100])
    maths = inline.findall(s)
    if l.lstrip().startswith("|"):
        for m in maths:
            if "|" in m:
                print("PIPE inside math in a table row, line", i, m)
    for m in maths:
        if "−" in m:
            print("unicode minus inside math, line", i, m[:80])
        if "*" in m:
            print("asterisk inside math, line", i, m[:80])
        if "_" in m and l.count("_") - sum(x.count("_") for x in maths) > 0:
            print("underscores inside and outside math, line", i)

text = "\n".join(lines)
print("commands:", sorted(set(re.findall(r"\\[A-Za-z]+", text))))
print("display-math lines (line number, characters):")
for i, l in enumerate(lines, 1):
    if l.strip().startswith(D + D):
        print("  ", i, len(l))
print("symbol counts in whole file:")
for pat in [r"\bz_5\b", r"m_\{\\text", r"\bh_9\b", r"\bh_5\b", r"age \$g\$", r"step \$s\$", r"\$s \+ ?1\$", r"t_r", r"relative gap", r"wipe", r"vote", r"bump", r"kick", r"supply", r"currency", r"centred", r"slot", r"place", r"easy inputs"]:
    hits = [i for i, l in enumerate(lines, 1) if re.search(pat, l)]
    print("  ", pat, len(hits), hits[:14])
