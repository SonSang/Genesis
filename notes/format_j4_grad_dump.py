"""Re-format `notes/diag_j4_grad_dump.txt` so each dump stage prints with
one field per line. Nonzero entries shown first; zeros collapsed at the
end as a count + name list for compactness."""

import re


def fmt(path_in="notes/diag_j4_grad_dump.txt", path_out="notes/diag_j4_grad_dump_multi.txt"):
    with open(path_in) as f:
        text = f.read()
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    line_re = re.compile(r"\[GRAD ([^\]]+)\] (.+)")
    field_re = re.compile(r"([\w_.]+): max=([0-9.e+\-]+) norm=([0-9.e+\-]+)")

    out_lines = []
    for line in text.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        tag = m.group(1).strip()
        fields = [(fm.group(1), float(fm.group(2)), float(fm.group(3))) for fm in field_re.finditer(m.group(2))]
        nonzero = [(n, mx, nm) for n, mx, nm in fields if mx != 0.0]
        zero = [n for n, mx, nm in fields if mx == 0.0]

        out_lines.append("")
        out_lines.append(f"=== [GRAD {tag}] ===")
        if nonzero:
            nonzero.sort(key=lambda x: -x[1])
            name_w = max(len(n) for n, _, _ in nonzero)
            for n, mx, nm in nonzero:
                out_lines.append(f"  {n:<{name_w}}  max={mx:.3e}  norm={nm:.3e}")
        if zero:
            out_lines.append(f"  -- {len(zero)} fields with max=0 --")
            out_lines.append(f"     {', '.join(zero)}")

    with open(path_out, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"wrote {path_out} ({sum(1 for _ in out_lines)} lines)")


if __name__ == "__main__":
    fmt()
