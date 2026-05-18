"""Parse a GENESIS_DEBUG_GRAD=2 dump into a compact per-stage table.

Usage:
  python parse_dump.py [path_to_dump.txt]
Default: notes/j4_n2_dump_current.txt
"""

import os
import re
import sys


def strip_ansi(s):
    return re.sub(r"\x1b\[[0-9;]*m", "", s)


KEYS = [
    ("qpos", "rigid_global_info.qpos.grad"),
    ("vel", "dofs_state.vel.grad"),
    ("vel_next", "dofs_state.vel_next.grad"),
    ("acc", "dofs_state.acc.grad"),
    ("acc_smooth", "dofs_state.acc_smooth.grad"),
    ("acc_smooth_bw", "dofs_state.acc_smooth_bw.grad"),
    ("force", "dofs_state.force.grad"),
    ("qf_bias", "dofs_state.qf_bias.grad"),
    ("ctrl_force", "dofs_state.ctrl_force.grad"),
    ("links_pos", "links_state.pos.grad"),
    ("links_quat", "links_state.quat.grad"),
    ("cd_vel", "links_state.cd_vel.grad"),
    ("cd_ang", "links_state.cd_ang.grad"),
    ("cd_vel_bw", "links_state.cd_vel_bw.grad"),
    ("cd_ang_bw", "links_state.cd_ang_bw.grad"),
    ("cdofd_ang", "dofs_state.cdofd_ang.grad"),
    ("cdofd_vel", "dofs_state.cdofd_vel.grad"),
    ("cdof_ang", "dofs_state.cdof_ang.grad"),
    ("cdof_vel", "dofs_state.cdof_vel.grad"),
    ("cinr_pos", "links_state.cinr_pos.grad"),
]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    default_path = os.path.join(here, "j4_n2_dump_current.txt")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    with open(path) as f:
        lines = [strip_ansi(line.rstrip()) for line in f]

    stages = []
    cur = None
    for line in lines:
        m = re.search(r"\[GRAD ([^\]]+)\]", line)
        if m:
            if cur is not None:
                stages.append(cur)
            cur = {"_tag": m.group(1), "_": {}}
            continue
        for short, full in KEYS:
            m = re.search(rf"{re.escape(full)} = \[([^\]]*)\]", line)
            if m:
                vals = re.findall(r"'(-?[0-9.eE+-]+)'", m.group(1))
                cur["_"][short] = [float(v) for v in vals]
    if cur is not None:
        stages.append(cur)

    # Print compact: per-stage, each tracked field
    for idx, s in enumerate(stages):
        print(f"\n[{idx:2d}] {s['_tag']}")
        for short, _ in KEYS:
            v = s["_"].get(short)
            if v is None:
                continue
            mx = max(abs(x) for x in v) if v else 0.0
            if mx < 1e-10:
                continue  # skip all-zero
            print(f"   {short:14s} max={mx:.3e}  vec={['%.3e' % x for x in v]}")


if __name__ == "__main__":
    main()
