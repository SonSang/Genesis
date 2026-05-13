"""Parse `_debug_grad_dump` output from notes/diag_j4_grad_dump.txt and
print a per-field stage timeline. Goal: spot fields whose `.grad`
magnitude collapses (zero or near-zero) between adjacent backward
stages, suggesting silent-drop locations."""

import re
from collections import defaultdict


def parse(path="notes/diag_j4_grad_dump.txt"):
    with open(path) as f:
        text = f.read()

    # Strip ANSI color codes
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)

    # Each dump line: [GRAD <tag>] field: max=X norm=Y | ...
    line_re = re.compile(r"\[GRAD ([^\]]+)\] (.+)")
    field_re = re.compile(r"([\w_.]+): max=([0-9.e+\-]+) norm=([0-9.e+\-]+)")

    rows = []  # list of (stage_idx_within_substep, substep_idx, tag, dict[field -> max])
    substep_idx = -1
    last_stage_idx = -1
    for line in text.splitlines():
        m = line_re.search(line)
        if not m:
            continue
        tag = m.group(1).strip()
        # New substep starts at "entry"
        if tag.endswith("entry"):
            substep_idx += 1
        body = m.group(2)
        fields = {fm.group(1): float(fm.group(2)) for fm in field_re.finditer(body)}
        rows.append((substep_idx, tag, fields))

    return rows


def field_timeline(rows, field):
    print(f"\n=== {field} (max-abs of .grad across stages) ===")
    for substep_idx, tag, fields in rows:
        v = fields.get(field, None)
        if v is None:
            continue
        print(f"  s{substep_idx} {tag:<45}  {v:.3e}")


def collapse_table(rows):
    """For each field, show the magnitude per stage. Highlight stages where
    a field is nonzero before but zero after (or drops by > 1e6×)."""
    # All fields seen
    all_fields = sorted({f for _, _, fl in rows for f in fl})
    # Print compact column matrix
    stages = [(s, t) for s, t, _ in rows]
    print("\n=== Stage-by-stage magnitude per field ===")
    print(f"{'field':<48}", end="")
    for s, t in stages:
        col = f"s{s}:{t[:8]}"
        print(f"{col:>11}", end="")
    print()
    for field in all_fields:
        print(f"{field:<48}", end="")
        for s, t, fl in rows:
            v = fl.get(field, None)
            if v is None or v == 0.0:
                print(f"{'-':>11}", end="")
            else:
                exp = f"{v:.1e}"
                print(f"{exp:>11}", end="")
        print()


def find_silent_drops(rows):
    """Heuristic: between adjacent stages within the same substep, find any
    field whose .grad goes from >1e-12 to 0. Those are silent-drop candidates."""
    print("\n=== Silent-drop candidates (nonzero -> 0 transition) ===")
    for i in range(1, len(rows)):
        s_prev, t_prev, f_prev = rows[i - 1]
        s_curr, t_curr, f_curr = rows[i]
        if s_prev != s_curr:
            continue
        for field in f_prev:
            v_prev = f_prev.get(field, 0.0)
            v_curr = f_curr.get(field, 0.0)
            if v_prev > 1e-12 and v_curr == 0.0:
                print(f"  s{s_curr} {t_prev:<45}  ->  {t_curr:<45}  {field}: {v_prev:.3e} -> 0")


if __name__ == "__main__":
    rows = parse()
    print(f"Parsed {len(rows)} stage dumps")
    collapse_table(rows)
    find_silent_drops(rows)
    # Useful per-field timelines:
    for f in [
        "links_state.cd_vel",
        "links_state.cd_ang",
        "dofs_state.cdof_vel",
        "dofs_state.cdof_ang",
        "dofs_state.cdofd_vel",
        "dofs_state.cdofd_ang",
        "links_state.cfrc_vel",
        "links_state.cfrc_ang",
        "links_state.cinr_inertial",
        "links_state.crb_inertial",
        "rigid_global_info.mass_mat",
        "rigid_global_info.mass_mat_L",
        "dofs_state.acc_smooth",
        "dofs_state.acc_smooth_bw",
        "dofs_state.force",
        "dofs_state.vel",
        "rigid_global_info.qpos",
    ]:
        field_timeline(rows, f)
