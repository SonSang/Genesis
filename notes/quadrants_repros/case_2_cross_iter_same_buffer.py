"""Case 2 — Cross-iteration same-buffer *read after write* drops the
reverse-mode adjoint chain (silently).

  * `out[i]` is written exactly once per outer iteration and is *not*
    read elsewhere in that iteration — no read-then-write on the same
    entry.
  * `out[j]` (j > i) is read in the current iteration; it was written
    in an *earlier* outer iteration. That is "write then read", which
    the access rule explicitly allows (the rule forbids only the
    opposite ordering).

So the access rule is clean, yet reverse-mode AD still silently drops
the adjoint contribution that should flow through `out[j]` back to
`vec[j]`. Only the direct `acc = vec[i]` seed survives, leaving
`vec.grad` on an identity-like pattern.

Forward (N=3, C=0.5):

    i_=0 → i=2:  acc = vec[2];                              out[2] = acc = 3
    i_=1 → i=1:  acc = vec[1]; acc -= C*out[2];             out[1] = acc = 0.5
    i_=2 → i=0:  acc = vec[0]; acc -= C*out[1]; acc -= C*out[2]; out[0] = acc = -0.75

Loss = sum(out) = vec[0] + (1-C)*vec[1] + (1-C)^2*vec[2]
              = vec[0] + 0.5*vec[1] + 0.25*vec[2]

Expected gradient: [1.0, 0.5, 0.25].

ACTUAL (with `ad_stack_experimental_enabled=True`): analytical = [1, 1, 1].

Genesis impact: this pattern lives in several rigid-body solver
functions and is the root of multiple xfails:

  * `abd/forward_dynamics.py::func_solve_mass_entity` — LDLT
    backward substitution (Step 1: L^T w = y).
  * `abd/forward_dynamics.py::func_factor_mass` — Cholesky-Banachiewicz
    mass-matrix factorization (BW path).
  * `abd/forward_kinematics.py::func_forward_kinematics_entity` —
    inter-link `links_state.pos[parent_idx]` → `links_state.pos[i_l]`
    propagation along the kinematic chain.
  * `abd/forward_kinematics.py::func_forward_velocity_entity` —
    inter-link `links_state.cd_{vel,ang}[parent_idx]` →
    `links_state.cd_{vel,ang}[i_l]` propagation. This is the dominant
    silent-drop chain (`cd_ang`) behind the J4/J5 multi-step gradient
    xfails in `tests/test_diff_forward_kinematics.py`.

The cross-iter chain must fire whenever the structure has non-trivial
inter-iteration coupling (multi-DOF entities for the LDLT path;
kinematic chains with parent-child links for the forward_kinematics /
forward_velocity paths). With the chain silently dropped, J4/J5
multi-step tests xfail (and the existing pattern test
`tests/test_quadrants_self_ref_ad.py::test_quadrants_cross_iter_same_buffer_ad`
is pinned xfail strict).
"""

import numpy as np

import quadrants as qd

qd.init(arch=qd.cpu, ad_stack_experimental_enabled=True)


N = 3
C = 0.5


@qd.kernel
def step_bug(vec: qd.template(), out: qd.template()):
    """Cross-iter same-buffer read: out[j] (j > i) was written in an
    earlier outer iteration. Reverse-mode AD silently drops the
    chain through this cross-iter read."""
    for i_ in range(N):
        i = N - i_ - 1
        acc = vec[i]
        for j in range(i + 1, N):
            acc += -C * out[j]  # cross-iter same-buffer read
        out[i] = acc


@qd.kernel
def step_ok(vec: qd.template(), out: qd.template()):
    """Same loop / accumulator / write structure as `step_bug`. The
    only difference: the inner read targets `vec[j]` (read-only input)
    instead of `out[j]` (same buffer being written). No cross-iter
    same-buffer pattern. Reverse-mode AD handles this correctly."""
    for i_ in range(N):
        i = N - i_ - 1
        acc = vec[i]
        for j in range(i + 1, N):
            acc += -C * vec[j]  # read from an independent input field
        out[i] = acc


def run(kernel, expected, label):
    vec = qd.field(dtype=qd.f64, shape=(N,), needs_grad=True)
    out = qd.field(dtype=qd.f64, shape=(N,), needs_grad=True)

    for i in range(N):
        vec[i] = float(i + 1)
        out[i] = 0.0
        out.grad[i] = 1.0  # loss = sum(out)
        vec.grad[i] = 0.0

    kernel.grad(vec, out)
    kernel(vec, out)

    analytical = np.array([float(vec.grad[i]) for i in range(N)])

    print(f"--- {label} ---")
    print(f"expected   = {expected}")
    print(f"analytical = {analytical}")
    ok = np.allclose(analytical, expected, rtol=1e-10, atol=1e-12)
    print("OK" if ok else "BUG: chain dropped — analytical lands on identity-like [1, 1, ...].")
    print()
    return ok


def main():
    # step_bug: forward gives sum(out) = vec[0] + (1-C)*vec[1] + (1-C)^2*vec[2]
    #                                  = vec[0] + 0.5*vec[1] + 0.25*vec[2]
    bug_expected = np.array([1.0, 0.5, 0.25])

    # step_ok: sum(out) = sum_i [ vec[i] - C * sum_{j>i} vec[j] ]
    #                   = vec[0] + (1 - C)*vec[1] + (1 - 2C)*vec[2]
    #                   = vec[0] + 0.5*vec[1] + 0.0*vec[2]
    ok_expected = np.array([1.0, 0.5, 0.0])

    run(step_ok, ok_expected, "step_ok (no cross-iter same-buffer)")
    run(step_bug, bug_expected, "step_bug (cross-iter same-buffer)")


if __name__ == "__main__":
    main()
