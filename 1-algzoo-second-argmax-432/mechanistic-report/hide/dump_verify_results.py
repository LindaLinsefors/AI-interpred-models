"""Dump the verifiers' structured results from the verification workflow journal into notes/v_<bundle>.md."""
import json, sys, pathlib
J = pathlib.Path(sys.argv[1])
OUTDIR = pathlib.Path(__file__).resolve().parent / "notes"
keys = ["V1_tape", "V2_scores", "V3_interventions", "V4_core", "V5_modes", "V6_deposits", "V7_units", "V8_codereview"]
n = 0
for line in open(J, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") != "result":
        continue
    r = d.get("result")
    if not isinstance(r, dict) or "verdicts" not in r:
        continue
    blob = " ".join(r.get("files", [])) + r.get("summary", "")
    key = next((k for k in keys if "v_" + k in blob), None)
    if key is None:
        ids = " ".join(v["claim_id"] for v in r["verdicts"])
        key = {"T": "V1_tape", "S": "V2_scores", "I": "V3_interventions", "K": "V4_core", "M": "V5_modes", "D": "V6_deposits", "O": "V7_units"}.get(ids[:1], f"unknown{n}")
    n += 1
    out = [f"# Verification bundle {key}\n", "## Summary\n", r["summary"], "\n## Verdicts\n"]
    for v in r["verdicts"]:
        out += [f"### {v['claim_id']}: {v['verdict']}\n", "- my numbers: " + v["my_numbers"], "- comment: " + v["comment"], ""]
    out += ["## Objections tested\n"]
    for o in r["objections"]:
        out += [f"- **{o['objection']}** (affects conclusion: {o['affects_conclusion']})", "  - test: " + o["test_done"], "  - result: " + o["result"]]
    out += ["\n## Bugs or issues\n"] + ["- " + b for b in r["bugs_or_issues"]]
    (OUTDIR / f"v_{key}.md").write_text("\n".join(out), encoding="utf-8")
    counts = {}
    for v in r["verdicts"]:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    print("wrote", f"v_{key}.md", counts)
print("verifiers done:", n)
