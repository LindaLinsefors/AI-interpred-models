"""Dump the report reviewers' structured results from the review workflow journal into notes/rev_<name>.md."""
import json, sys, pathlib
J = pathlib.Path(sys.argv[1])
OUTDIR = pathlib.Path(__file__).resolve().parent / "notes"
labels = {}
n = 0
for line in open(J, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "started":
        labels[d.get("agentId")] = d.get("label", "").replace("review-", "")
        labels[d.get("key")] = labels[d.get("agentId")]
    if d.get("type") != "result":
        continue
    r = d.get("result")
    if not isinstance(r, dict) or "issues" not in r:
        continue
    key = labels.get(d.get("agentId")) or labels.get(d.get("key")) or f"unknown{n}"
    n += 1
    order = {"major": 0, "moderate": 1, "minor": 2}
    issues = sorted(r["issues"], key=lambda i: order.get(str(i.get("severity", "")).lower(), 3))
    out = [f"# Report review {key}\n", "## Summary\n", r["summary"], f"\n## Issues ({len(issues)})\n"]
    for i in issues:
        out += [f"### [{i.get('severity')}] section {i.get('section')}\n", "- quote: " + str(i.get("quote")),
                "- problem: " + str(i.get("problem")), "- suggested fix: " + str(i.get("suggested_fix")), ""]
    out += ["## Files\n"] + ["- " + str(f) for f in r.get("files", [])]
    (OUTDIR / f"rev_{key}.md").write_text("\n".join(out), encoding="utf-8")
    counts = {}
    for i in issues:
        counts[i.get("severity")] = counts.get(i.get("severity"), 0) + 1
    print("wrote", f"rev_{key}.md", counts)
print("reviewers done:", n)
