"""Parse J4 N=2 verbose dump (GENESIS_DEBUG_GRAD=2) into a stage-by-
stage table tracking the root_wx chain.

root_wx (= ctrl_force index 3 = chassis local x torque):
  - ctrl_force.grad[3]  = our ana[t=0][3]
  - traced back through:
    qpos.grad[3..6]  (chassis quat: qw, qx, qy, qz)
    dofs_state.vel.grad[3]  (chassis angular velocity wx)
    dofs_state.acc.grad[3]
    dofs_state.force.grad[3]
"""

import re


def main():
    raw = open("notes/diag_j4_n2_substep_dump_verbose.txt").read()
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    raw = ansi.sub("", raw)
    lines = raw.splitlines()

    # collect (stage_tag, {field: [floats...]})
    stages = []
    cur_tag = None
    cur_fields = {}
    for ln in lines:
        m = re.search(r"\[GRAD ([^\]]+)\]", ln)
        if m:
            if cur_tag is not None:
                stages.append((cur_tag, cur_fields))
            cur_tag = m.group(1)
            cur_fields = {}
            continue
        m = re.search(r"  (\S+)\.grad = \['([^']+)'(?:, '([^']+)')*\]", ln)
        if not m:
            # try a more permissive match
            m2 = re.match(r".*\s(\S+)\.grad = \[(.+)\]\s*$", ln)
            if not m2:
                continue
            field = m2.group(1)
            vals = re.findall(r"-?[\d.]+e[+-]\d+", m2.group(2))
            cur_fields[field] = [float(v) for v in vals]
            continue
        field = m.group(1)
        # re-extract all floats from the line
        vals = re.findall(r"-?\d\.\d+e[+-]\d+", ln)
        cur_fields[field] = [float(v) for v in vals]
    if cur_tag is not None:
        stages.append((cur_tag, cur_fields))

    out_lines = []
    out_lines.append("J4 N=2 seed=1000 — root_wx chain trace per stage")
    out_lines.append("=" * 110)
    out_lines.append(
        "Indices: qpos[3..6] = chassis quat (qw, qx, qy, qz)  |  vel.grad[3..5] = chassis ang vel  |"
    )
    out_lines.append(
        "         vel.grad[6] = arm angle  |  force/acc.grad[3] = chassis wx force/acc"
    )
    out_lines.append("")

    for tag, fields in stages:
        # only print if anything is non-zero
        relevant = ["rigid_global_info.qpos", "dofs_state.vel", "dofs_state.acc",
                    "dofs_state.force", "dofs_state.acc_smooth", "dofs_state.acc_smooth_bw",
                    "dofs_state.qf_bias", "links_state.cd_vel", "links_state.cd_ang",
                    "links_state.pos", "links_state.quat"]
        has_nonzero = False
        for f in relevant:
            v = fields.get(f, [])
            if any(abs(x) > 1e-15 for x in v):
                has_nonzero = True
                break
        if not has_nonzero:
            continue
        out_lines.append(f"\n[{tag}]")
        for f in relevant:
            v = fields.get(f)
            if v is None or not any(abs(x) > 1e-15 for x in v):
                continue
            # pretty-print as scientific
            vs = ", ".join(f"{x:>10.2e}" for x in v)
            out_lines.append(f"  {f:<32}  [{vs}]")

    text = "\n".join(out_lines)
    with open("notes/diag_j4_n2_substep_dump_parsed_wx.txt", "w") as fh:
        fh.write(text + "\n")
    print(text)
    print(f"\nwrote notes/diag_j4_n2_substep_dump_parsed_wx.txt ({len(out_lines)} lines)")


if __name__ == "__main__":
    main()
