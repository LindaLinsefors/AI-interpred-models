"""Dump the results of the workflow "tape-what-is-stored" (arms t1_*, verifiers tv_*) from its journal into
notes/t1_<key>.md and notes/tv_<key>.md.   Usage: python -B hide/dump_tape_results.py <journal.jsonl>"""
import json, sys, pathlib

J = pathlib.Path(sys.argv[1])
NOTES = pathlib.Path(__file__).resolve().parent / "notes"
labels, done = {}, []
for line in open(J, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "started":
        labels[d["agentId"]] = d.get("label", "")
    if d.get("type") != "result" or not isinstance(d.get("result"), dict):
        continue
    role, key = (labels.get(d["agentId"], "unknown:unknown").split(":", 1) + ["unknown"])[:2]
    r = d["result"]
    if role == "arm":
        out = [f"# Tape investigation, arm {key}\n", "## Answer\n", r.get("answer", ""), f"\n## Claims ({len(r.get('claims', []))})\n"]
        for i, c in enumerate(r.get("claims", []), 1):
            out += [f"### {i}. [{c.get('strength')}] {c.get('claim')}\n", "- evidence: " + str(c.get("evidence")),
                    "- script: " + str(c.get("script")), "- caveats: " + str(c.get("caveats")), ""]
        for h, k in (("Did not work", "failed"), ("Open", "open"), ("Files", "files")):
            out += [f"## {h}\n"] + ["- " + str(v) for v in r.get(k, [])] + [""]
        name = f"t1_{key}.md"
    else:
        out = [f"# Tape investigation, verifier of arm {key}\n", "## Summary\n", r.get("summary", ""), "\n## Corrected answer\n",
               r.get("corrected_answer", ""), f"\n## Verdicts ({len(r.get('verdicts', []))})\n"]
        for v in r.get("verdicts", []):
            out += [f"### [{v.get('verdict')}] {v.get('claim')}\n", "- my numbers: " + str(v.get("my_numbers")), "- note: " + str(v.get("note")), ""]
        out += [f"## Attacks ({len(r.get('attacks', []))})\n"]
        for a in r.get("attacks", []):
            out += ["### " + str(a.get("test")) + "\n", "- result: " + str(a.get("result")), "- consequence: " + str(a.get("consequence")), ""]
        out += ["## Files\n"] + ["- " + str(v) for v in r.get("files", [])]
        name = f"tv_{key}.md"
    (NOTES / name).write_text("\n".join(out), encoding="utf-8")
    done.append(name)
print("wrote", sorted(done))
