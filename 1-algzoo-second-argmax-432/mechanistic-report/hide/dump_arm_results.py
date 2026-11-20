"""Dump the structured results of the round-1 arms from the workflow journal into notes/r1_<arm>.md."""
import json, sys, pathlib
J = pathlib.Path(sys.argv[1])
OUTDIR = pathlib.Path(__file__).resolve().parent / "notes"
seen = 0
for line in open(J, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") != "result":
        continue
    r = d.get("result")
    if not isinstance(r, dict) or "claims" not in r:
        continue
    files = " ".join(r.get("files", []))
    arm = None
    for key in ("A_census", "B_max_circuit", "C_clock", "D_geometry", "E_logit_sources", "F_record_event", "H_runnerup_events", "I_errors"):
        if "r1_" + key in files or "r1_" + key in r.get("summary", ""):
            arm = key
    if arm is None:
        arm = f"unknown{seen}"
    seen += 1
    out = [f"# Round-1 arm {arm}: structured result\n", "## Summary\n", r["summary"], "\n## Claims\n"]
    for c in r["claims"]:
        out.append(f"### {c['id']} [{c['evidence_type']}, confidence {c['confidence']}]\n")
        out.append(c["statement"] + "\n")
        out.append("- numbers: " + c["numbers"])
        out.append("- how checked: " + c["how_checked"])
        if c.get("caveats"):
            out.append("- caveats: " + c["caveats"])
        out.append("")
    out += ["## Contradictions with the lead\n", r["contradictions_with_lead"], "\n## Surprises\n", r["surprises"],
            "\n## Not done\n", r["not_done"], "\n## Open questions\n"] + ["- " + q for q in r["open_questions"]]
    out += ["\n## Files\n"] + ["- " + f for f in r.get("files", [])]
    p = OUTDIR / f"r1_{arm}.md"
    p.write_text("\n".join(out), encoding="utf-8")
    print("wrote", p.name, len("\n".join(out)), "chars,", len(r["claims"]), "claims")
