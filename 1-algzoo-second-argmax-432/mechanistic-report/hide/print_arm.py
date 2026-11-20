"""Print an arm result compactly: summary, claim ids + statements, contradictions, surprises, open questions."""
import sys, pathlib
sys.stdout.reconfigure(encoding="utf-8")
t = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
maxlen = int(sys.argv[2]) if len(sys.argv) > 2 else 1200
head, rest = t.split("## Claims", 1)
print(head[:7000])
claims, tail = rest.split("## Contradictions with the lead", 1)
for block in claims.split("### ")[1:]:
    lines = block.strip().split("\n")
    print("###", lines[0])
    body = "\n".join(lines[1:]).strip().split("\n- numbers:")[0].strip()
    print(body[:maxlen])
    print()
print("## Contradictions with the lead" + tail.split("## Files")[0][:12000])
