"""Re-format `notes/diag_j4_n1_grad_dump_full.txt`:
- collapse zero fields to a single line per stage
- keep full per-element arrays for nonzero fields
"""

import re


def main():
    with open("notes/diag_j4_n1_grad_dump_full.txt") as f:
        raw = f.read()
    raw = re.sub(r"\x1b\[[0-9;]*m", "", raw)

    stage_header = re.compile(r"^===== \[(.+)\] =====$")
    field_header = re.compile(r"^  ([\w_.]+)\s+shape=(\[[^\]]+\])\s+max\|\.\|=([0-9.e+\-]+)$")

    lines = raw.splitlines()
    out = []
    i = 0
    while i < len(lines):
        m = stage_header.match(lines[i])
        if m:
            out.append("")
            out.append(f"===== {m.group(1)} =====")
            i += 1
            stage_zero = []
            stage_nonzero = []
            while i < len(lines) and not stage_header.match(lines[i]):
                fm = field_header.match(lines[i])
                if fm:
                    name = fm.group(1)
                    shape = fm.group(2)
                    mx = float(fm.group(3))
                    # Collect any continuation lines until next field/stage header
                    arr_lines = []
                    j = i + 1
                    while j < len(lines):
                        if stage_header.match(lines[j]):
                            break
                        if field_header.match(lines[j]):
                            break
                        arr_lines.append(lines[j])
                        j += 1
                    if mx == 0.0:
                        stage_zero.append(name)
                    else:
                        stage_nonzero.append((name, shape, mx, arr_lines))
                    i = j
                else:
                    i += 1
            # Print nonzero in descending magnitude
            stage_nonzero.sort(key=lambda x: -x[2])
            for name, shape, mx, arr_lines in stage_nonzero:
                out.append(f"  {name}  shape={shape}  max|.|={mx:.3e}")
                for ln in arr_lines:
                    if ln.strip():
                        out.append(ln)
            if stage_zero:
                out.append(f"  -- {len(stage_zero)} fields with max|.|=0 --")
                out.append(f"     {', '.join(stage_zero)}")
        else:
            i += 1

    with open("notes/diag_j4_n1_grad_dump_full_clean.txt", "w") as f:
        f.write("\n".join(out) + "\n")
    print(f"wrote notes/diag_j4_n1_grad_dump_full_clean.txt ({len(out)} lines)")


if __name__ == "__main__":
    main()
