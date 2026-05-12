"""Case 1 — Kernel-side `.grad` writes are inconsistent.

When a `@qd.kernel` writes to a field's `.grad` buffer (e.g. to zero it
between iterations), behavior is INCONSISTENT:

  * Isolated case (this script): kernel write DOES persist correctly.
  * Genesis context (real solver, many kernel forwards + `.grad` calls
    interleaved): kernel write SILENTLY no-ops. Only Python-side
    `qd_to_torch(field.grad, copy=False).zero_()` actually clears the
    buffer.

We have not yet isolated the exact trigger pattern in a minimal
reproducer. The Genesis fix that consistently works:

    # In `RigidSolver.substep_pre_coupling_grad`:
    @qd.kernel(fastcache=True)
    def kernel_zero_acc_smooth_bw_grad(dofs_state: ...):
        for i, i_d, i_b in qd.ndrange(2, ..., ...):
            dofs_state.acc_smooth_bw.grad[i, i_d, i_b] = 0.0   # NO-OP

    # vs. (this works)
    qd_to_torch(self.dofs_state.acc_smooth_bw.grad, copy=False).zero_()

When the `kernel_zero_acc_smooth_bw_grad` was in place, before/after
dumps of `acc_smooth_bw.grad` showed the same non-zero value, despite
the kernel having "run" between the dumps.

EXPECTED: kernel-side `.grad` writes persist identically to Python-side.

ACTUAL (in real solver, not in this isolated repro): kernel writes are
silently dropped.

Severity: High — silent (no exception), debugging burden is enormous,
and the obvious "matches forward-write semantics" intuition is wrong.

ASK FOR QUADRANTS TEAM: please help us minimize this further. Suspect
triggers:
  * `fastcache=True` decorator.
  * Inside-`substep_pre_coupling_grad` execution order (zero kernel
    invoked between other kernels' forward push and `.grad` reverse).
  * Buffer is a `*_bw` BW-only field whose `.grad` lives in a
    different storage class.
"""

import numpy as np

import quadrants as qd


qd.init(arch=qd.cpu)


def main():
    n = 4
    field = qd.field(dtype=qd.f64, shape=(n,), needs_grad=True)

    for i in range(n):
        field.grad[i] = 1.0 + i

    before = np.array([float(field.grad[i]) for i in range(n)])
    print(f"before kernel write: field.grad = {before}")

    @qd.kernel(fastcache=True)
    def zero_grad():
        for i in range(n):
            field.grad[i] = 0.0

    zero_grad()
    after_kernel = np.array([float(field.grad[i]) for i in range(n)])
    print(f"after  kernel write: field.grad = {after_kernel}")

    if np.allclose(after_kernel, 0.0):
        print("ISOLATED CASE WORKS — Genesis context is where it breaks.")
        print("See Case 4 for the big-kernel-context pattern.")
    else:
        print("BUG: kernel-side `.grad` write was silently dropped.")


if __name__ == "__main__":
    main()
