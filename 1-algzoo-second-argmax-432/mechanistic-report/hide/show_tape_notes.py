"""Print chosen sections of the tape notes.  Usage: python -B show_tape_notes.py <file stem> <heading prefix> [...more prefixes]"""
import sys, pathlib
p = pathlib.Path(__file__).resolve().parent / "notes" / (sys.argv[1] + ".md")
want = sys.argv[2:]
show = False
for line in p.read_text(encoding="utf-8").split("\n"):
    if line.startswith("## "):
        show = any(line[3:].startswith(w) for w in want)
    if show:
        print(line)
