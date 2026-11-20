"""Compact markdown tables from d3_intervene.txt.   python -B d3_summary.py"""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
lines = (HERE / "d3_intervene.txt").read_text().splitlines()
t0, block = None, None
rows = {}
for ln in lines:
    m = re.match(r"t0 = (\d)", ln)
    if m:
        t0 = int(m.group(1))
    m = re.match(r"\s+--- scale (\w+); variant (\w+); continuation (\w+)", ln)
    if m:
        block = (t0,) + m.groups()
        rows[block] = []
    m = re.match(r"\s+(\d\.\d\d) \|", ln)
    if m and block:
        rows[block].append([p.strip() for p in ln.split("|")])


def med(cell):                      # '0.61[ 0.27, 0.82]  +21[  -14,  +38]' -> ('0.61', '0.27-0.82', '+21', '-14..+38')
    a = re.findall(r"([-+]?\d+\.?\d*)\[\s*([-+]?\d+\.?\d*),\s*([-+]?\d+\.?\d*)\]", cell)
    return a[0], a[1]


for which in ("level", "clock"):
    for variant in ("pure", "relu"):
        for cont in ("real", "negreal", "zero", "neg"):
            for t0 in (3, 5, 7):
                key = (t0, which, variant, cont)
                if key not in rows:
                    continue
                print(f"\n**scale {which}, variant {variant}, continuation {cont}, t0 = {t0}**\n")
                scored = cont in ("real", "negreal")
                print("| f | negative entries in h'(t0) (of 13) | newly OFF units at k=1 | mean per step k>=2 | unit 4 fires extra / missed | clock ratio k=1: median [IQR] | phase k=1 (deg): median [IQR] | clock ratio last step | phase last step | level'/level k=1, last |"
                      + (" accuracy all | accuracy on K |" if scored else ""))
                print("|---|---|---|---|---|---|---|---|---|---|" + ("---|---|" if scored else ""))
                for r in rows[key]:
                    c1, c2, c3 = r[5].split(";")
                    (a1, p1), (a3, p3) = med(c1), med(c3)
                    off1, offm = r[2].split()
                    line = (f"| {r[0]} | {r[1]} | {off1} | {offm} | {r[4].replace(' ', ' / ')} | {a1[0]} [{a1[1]}, {a1[2]}] | {p1[0]} [{p1[1]}, {p1[2]}] |"
                            f" {a3[0]} [{a3[1]}, {a3[2]}] | {p3[0]} [{p3[1]}, {p3[2]}] | {r[6].replace('  ', ', ')} |")
                    if scored:
                        acc = r[7].split()
                        line += f" {acc[0]} | {acc[2]} |"
                    print(line)
