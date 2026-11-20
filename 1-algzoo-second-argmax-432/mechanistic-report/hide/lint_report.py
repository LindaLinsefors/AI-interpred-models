"""Lint ../report.md (it does not modify the report).

Checks: balanced dollar signs per line, display equations alone on their line, no pipe inside math in a table row.
Extra arguments are words to locate:  python -B assemble_report.py "some phrase"
(The name is historical: the report was first assembled from seven part files, which no longer exist.)
"""
import sys
from pathlib import Path

sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
REPORT = Path(__file__).resolve().parent.parent / "report.md"
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
