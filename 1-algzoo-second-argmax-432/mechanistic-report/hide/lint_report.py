"""Lint ../report.md, or another report in that folder (it does not modify the report).

Checks: balanced dollar signs per line, display equations alone on their line, no pipe inside math in a table row.
    python -B hide/lint_report.py                    lints report.md
    python -B hide/lint_report.py report_v2.md       lints report_v2.md
Further arguments are words to locate:  python -B hide/lint_report.py report_v2.md "some phrase"
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
name = sys.argv.pop(1) if len(sys.argv) > 1 and sys.argv[1].endswith(".md") else "report.md"
REPORT = Path(__file__).resolve().parent.parent / name
txt = REPORT.read_text(encoding="utf-8")
L = txt.split("\n")
print(len(txt), "chars,", len(L), "lines,", len(txt.split()), "words")
print("lines with an odd number of dollar signs:", [i + 1 for i, l in enumerate(L) if l.count("$") % 2])
dd = "$" + "$"
print("display equations not alone on their line:",
      [i + 1 for i, l in enumerate(L) if dd in l and not (l.startswith(dd) and l.rstrip().endswith(dd))])
print("table rows with a pipe inside math:",                      # odd-numbered pieces of a split on $ are inside math
      [i + 1 for i, l in enumerate(L) if l.startswith("|") and any("|" in s for s in l.split("$")[1::2])])
print("section headings:", sum(l.startswith("#") for l in L))
for word in sys.argv[1:]:
    print(repr(word), [i + 1 for i, l in enumerate(L) if word in l])
