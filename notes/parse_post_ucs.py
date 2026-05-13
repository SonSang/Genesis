"""Parse the two `post-update_cartesian_space.grad` dump lines and
print, side-by-side, every field whose .grad is non-zero at each
substep. Validates the user's hypothesis:
    After post-UCS.grad, ONLY `rigid_global_info.qpos.grad` and
    `dofs_state.vel.grad` should be non-zero. Any other non-zero
    field is a cross-substep leak source.
"""

import re

with open("/tmp/post_ucs_dump.txt") as f:
    lines = [ln for ln in f.read().splitlines() if "post-update_cartesian_space" in ln]

assert len(lines) == 2, f"expected 2 dump lines, got {len(lines)}"


# Strip ANSI + "[GENESIS] [ts] [INFO] [GRAD f=0 after post-update_cartesian_space.grad] " prefix
def parse(line):
    ansi = re.compile(r"\x1b\[[0-9;]*m")
    line = ansi.sub("", line)
    # find " | " separated parts
    body = line.split("]", 4)[-1]  # after the trailing ] of "[GRAD ... post-...]"
    parts = [p.strip() for p in body.split("|")]
    fields = []
    for p in parts:
        m = re.match(r"([\w.\[\]]+): max=([\d.eE+-]+) norm=([\d.eE+-]+)", p)
        if not m:
            continue
        fields.append((m.group(1), float(m.group(2)), float(m.group(3))))
    return fields


substeps = [parse(line) for line in lines]
# substeps[0] = first dump emitted in time = t=N-1 (first backward) = t=1 for N=2
# substeps[1] = second dump = t=0 (second backward)
labels = ["t=1 (first BW)", "t=0 (second BW)"]

print("Per-substep post-UCS.grad dump comparison (max|.grad|)")
print("=" * 90)
print(f"{'field':<46} {'t=1 (first BW)':>18} {'t=0 (second BW)':>18}")
print("-" * 90)

# build a unified field list
field_names = [f[0] for f in substeps[0]]
for name in field_names:
    f0 = next((f for f in substeps[0] if f[0] == name), None)
    f1 = next((f for f in substeps[1] if f[0] == name), None)
    m0 = f0[1] if f0 else 0
    m1 = f1[1] if f1 else 0
    expected_zero = name not in ("rigid_global_info.qpos", "dofs_state.vel")
    flag = ""
    if expected_zero and (m0 > 0 or m1 > 0):
        flag = "  <-- LEAK (should be 0)"
    print(f"{name:<46} {m0:>18.3e} {m1:>18.3e}{flag}")
