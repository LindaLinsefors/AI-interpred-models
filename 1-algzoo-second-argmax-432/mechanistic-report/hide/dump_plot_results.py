"""Dump the figure makers' and checkers' structured results from the plots workflow journal into
notes/plots_check_results.md.   Usage: python -B hide/dump_plot_results.py <journal.jsonl>"""
import json, sys, pathlib

J = pathlib.Path(sys.argv[1])
OUT = pathlib.Path(__file__).resolve().parent / "notes" / "plots_check_results.md"
labels, res = {}, {}
for line in open(J, encoding="utf-8"):
    try:
        d = json.loads(line)
    except Exception:
        continue
    if d.get("type") == "started":
        labels[d["agentId"]] = d.get("label", "")
    if d.get("type") == "result" and isinstance(d.get("result"), dict):
        role, topic = labels.get(d["agentId"], "unknown:unknown").split(":", 1)
        res.setdefault(topic, {})[role] = d["result"]

name = lambda p: str(p).replace("\\", "/").split("/")[-1]
out = ["# Extra figures: results of the makers and of the independent checkers (2026-11-20)\n",
       "One section per plot script. Each figure has the maker's own checks and the checker's verdict "
       "(`ok` = nothing changed, `fixed` = the checker edited the script and redrew, `problem` = an open fault). "
       "Dumped from the workflow journal by `hide/dump_plot_results.py`; the text is the agents', unedited.\n"]
for topic, r in res.items():
    out.append(f"## plot_{topic}.py\n")
    for role, title in (("make", "Maker"), ("check", "Checker")):
        if role not in r:
            continue
        out.append(f"### {title}\n")
        for k, v in r[role].items():
            if k != "figures" and isinstance(v, str):
                out += [f"**{k}:** {v}", ""]
        for f in r[role].get("figures", []):
            out.append(f"#### `{name(f.get('png'))}`" + (f" ({f['status']})" if "status" in f else "") + "\n")
            for k in ("caption", "checks", "remark"):
                if f.get(k):
                    out += [f"*{k}:* {f[k]}", ""]
OUT.write_text("\n".join(out), encoding="utf-8")
print("wrote", OUT.name, {t: sorted(r) for t, r in res.items()})
