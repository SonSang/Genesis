"""Quadrants reverse-mode AD limitation around self-referential outer loops.

These tests pin two minimal patterns extracted from `func_solve_mass_entity`
(`genesis/engine/solvers/rigid/abd/forward_dynamics.py:660`), the LDLT solve
used by `func_compute_qacc`:

  * Pattern A — simple two-slot copy: ``out[1, i] = out[0, i] * c``. This is the
    Step 2 (`z = D^-1 w`) pattern. Quadrants AD handles it correctly.
  * Pattern B — outer-loop with same-buffer cross-iter read: ``out[0, i_d] =
    out[0, i_d] - L[j_d, i_d] * out[0, j_d]``. This is the Step 1 (`L^T w = y`,
    backward substitution) pattern. Quadrants AD currently drops the
    cross-iteration adjoint contribution, so the gradient through `L` and
    through prior `i_d_` iterations is missing.

Pattern B is the reason `J4`/`J5` topologies in
``tests/test_diff_forward_kinematics.py`` fail: with multi-DOF entities the
mass matrix's L factor has non-trivial off-diagonal entries, so this chain has
to fire, but Quadrants only forwards the direct ``out[0, i_d] = vec[i_d]``
seed.

When pattern B starts passing, the ``xfail(strict=True)`` will flip to XPASS
and break CI — that's the cue to delete the xfail and re-enable J4/J5 in
``test_diff_forward_kinematics.py``.
"""

import numpy as np
import pytest

import genesis as gs


pytestmark = [
    pytest.mark.precision("64"),
]


def _setup_step1_inputs(n, vec, out_bw, L):
    """Seed vec=[1..n], zero out_bw, fill L with a fixed lower-triangular pattern."""
    for i in range(n):
        vec[i] = float(i + 1)
        out_bw[0, i] = 0.0
        out_bw[1, i] = 0.0
        for j in range(n):
            L[i, j] = 0.0
    # Strict lower triangle entries large enough that cross-iter chain is observable.
    L[1, 0] = 0.1
    L[2, 0] = 0.2
    L[2, 1] = 0.3
    L[3, 0] = 0.4
    L[3, 1] = 0.5
    L[3, 2] = 0.6


def test_quadrants_two_slot_self_ref_ad():
    """Pattern A — Step 2 of LDLT solve. Quadrants AD must handle this correctly.

    This is the control test: if it ever fails, something fundamental about
    reverse-mode through a simple read-then-write across two field slots has
    regressed, and Pattern B's xfail diagnosis would no longer hold.
    """
    import quadrants as qd

    n = 4
    vec = qd.field(dtype=gs.qd_float, shape=(n,), needs_grad=True)
    out_bw = qd.field(dtype=gs.qd_float, shape=(2, n), needs_grad=True)

    @qd.kernel
    def kernel():
        for i in range(n):
            out_bw[0, i] = vec[i]
        for i in range(n):
            out_bw[1, i] = out_bw[0, i] * 2.0

    for i in range(n):
        vec[i] = float(i + 1)
        out_bw[0, i] = 0.0
        out_bw[1, i] = 0.0
        out_bw.grad[0, i] = 0.0
        out_bw.grad[1, i] = 1.0
        vec.grad[i] = 0.0
    kernel()
    kernel.grad()

    grad = np.array([float(vec.grad[i]) for i in range(n)])
    np.testing.assert_allclose(grad, np.full(n, 2.0), rtol=1e-10, atol=1e-10)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Quadrants AD drops the cross-iteration adjoint contribution from "
        "`out[0, i_d] = out[0, i_d] - L[j_d, i_d] * out[0, j_d]`. When this xfail "
        "flips to XPASS, fix `func_solve_mass_entity` callers / xfail markers in "
        "`tests/test_diff_forward_kinematics.py` (J4/J5)."
    ),
)
def test_quadrants_cross_iter_same_buffer_ad():
    """Pattern B — Step 1 of LDLT solve. Currently broken in Quadrants AD."""
    import quadrants as qd

    n = 4
    vec = qd.field(dtype=gs.qd_float, shape=(n,), needs_grad=True)
    out_bw = qd.field(dtype=gs.qd_float, shape=(2, n), needs_grad=True)
    L = qd.field(dtype=gs.qd_float, shape=(n, n), needs_grad=False)

    @qd.kernel
    def step1():
        for i_d_ in range(n):
            i_d = n - i_d_ - 1
            out_bw[0, i_d] = vec[i_d]
            for j_d in range(i_d + 1, n):
                out_bw[0, i_d] = out_bw[0, i_d] - L[j_d, i_d] * out_bw[0, j_d]

    # Finite-difference reference: d(sum_i w_i) / d(vec[k]).
    def _fd():
        eps = 1e-6
        fd = np.zeros(n)
        for k in range(n):
            _setup_step1_inputs(n, vec, out_bw, L)
            vec[k] = float(k + 1) + eps
            step1()
            wp = sum(float(out_bw[0, i]) for i in range(n))
            _setup_step1_inputs(n, vec, out_bw, L)
            vec[k] = float(k + 1) - eps
            step1()
            wm = sum(float(out_bw[0, i]) for i in range(n))
            fd[k] = (wp - wm) / (2 * eps)
        return fd

    fd = _fd()

    # Analytical via Quadrants reverse mode, seeded with grad[out_bw[0, :]] = 1.
    _setup_step1_inputs(n, vec, out_bw, L)
    for i in range(n):
        out_bw.grad[0, i] = 1.0
        out_bw.grad[1, i] = 0.0
        vec.grad[i] = 0.0
    step1()
    step1.grad()
    analytical = np.array([float(vec.grad[i]) for i in range(n)])

    # Expected (currently failing) behavior: analytical matches FD. With the
    # cross-iter chain dropped, Quadrants returns [1, 1, 1, 1] while FD has the
    # off-diagonal contributions [1.0, 0.9, 0.53, -0.168].
    np.testing.assert_allclose(analytical, fd, rtol=1e-5, atol=1e-7)
