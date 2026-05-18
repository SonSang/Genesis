"""Parse J4 N=2 GRAD dump into a per-stage non-zero fields table."""

import re
import subprocess


def main():
    out = subprocess.run(
        ["python", "-u", "notes/diag_j4_n2_substep_dump.py"],
        capture_output=True, text=True, cwd="/home/sanghyun/Documents/Genesis",
    )
    lines = [ln for ln in out.stdout.splitlines() if "GRAD" in ln]

    ansi = re.compile(r"\x1b\[[0-9;]*m")
    stages = []
    for line in lines:
        line = ansi.sub("", line)
        # [GRAD f=X tag] field: max=v norm=v | field: ...
        m = re.match(r".*\[GRAD ([^\]]+)\] (.*)", line)
        if not m:
            continue
        tag = m.group(1)
        body = m.group(2)
        parts = [p.strip() for p in body.split("|")]
        nonzero = {}
        for p in parts:
            m2 = re.match(r"([\w.\[\]]+): max=([\d.eE+-]+) norm=", p)
            if not m2:
                continue
            name = m2.group(1)
            val = float(m2.group(2))
            if val > 0:
                nonzero[name] = val
        stages.append((tag, nonzero))

    # Print per-stage non-zero summary
    print("J4 N=2 backward stage dump — non-zero |.grad|_max")
    print("=" * 110)
    for tag, nz in stages:
        print(f"\n[{tag}]")
        for name, val in sorted(nz.items(), key=lambda kv: -kv[1]):
            print(f"  {name:<42}  {val:.3e}")


if __name__ == "__main__":
    main()
