"""Verify manual backward functions against Quadrants AD on standalone
kernel calls.

For each forward function we hand-derived in
`genesis/engine/solvers/rigid/abd/manual_bw.py`, this script:
  1. Runs Quadrants AD reverse-mode on the forward function with a
     known seed, captures grad
  2. Calls our manual backward function with the same seed
  3. Compares the two; OK if max|diff| < 1e-10
"""

import numpy as np

import genesis as gs
import quadrants as qd


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    from genesis.engine.solvers.rigid.abd.manual_bw import (
        d_transform_by_quat__dq,
        d_quat_mul__dlhs,
        d_quat_mul__drhs,
        d_rotvec_to_quat__drotvec,
    )
    import genesis.utils.geom as gu

    # ---------- Test 1: d_transform_by_quat__dq ----------
    print("=" * 70)
    print("Test 1: d_transform_by_quat__dq")
    print("=" * 70)

    v_f = qd.field(dtype=gs.qd_float, shape=(3,), needs_grad=False)
    quat_f = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=True)
    out_f = qd.field(dtype=gs.qd_float, shape=(3,), needs_grad=True)
    quat_grad_manual = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=False)

    @qd.kernel
    def k_forward(v: qd.template(), quat: qd.template(), out: qd.template()):
        v_vec = qd.Vector([v[0], v[1], v[2]], dt=gs.qd_float)
        q_vec = qd.Vector([quat[0], quat[1], quat[2], quat[3]], dt=gs.qd_float)
        r = gu.qd_transform_by_quat(v_vec, q_vec)
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]

    @qd.kernel
    def k_manual_bw(v: qd.template(), quat: qd.template(), out_grad: qd.template(), q_grad: qd.template()):
        v_vec = qd.Vector([v[0], v[1], v[2]], dt=gs.qd_float)
        q_vec = qd.Vector([quat[0], quat[1], quat[2], quat[3]], dt=gs.qd_float)
        og = qd.Vector([out_grad[0], out_grad[1], out_grad[2]], dt=gs.qd_float)
        qg = d_transform_by_quat__dq(v_vec, q_vec, og)
        q_grad[0] = qg[0]
        q_grad[1] = qg[1]
        q_grad[2] = qg[2]
        q_grad[3] = qg[3]

    quat_np = np.array([np.sqrt(1 - 0.01), 0.0, 0.1, 0.0])
    v_np = np.array([0.2, 0.0, 0.0])
    seed_np = np.ones(3)

    v_f.from_numpy(v_np)
    quat_f.from_numpy(quat_np)
    out_f.from_numpy(np.zeros(3))
    out_f.grad.from_numpy(seed_np)
    quat_f.grad.from_numpy(np.zeros(4))
    k_forward(v_f, quat_f, out_f)
    k_forward.grad(v_f, quat_f, out_f)
    ad_q_grad = np.array([float(quat_f.grad[i]) for i in range(4)])

    out_grad_f = qd.field(dtype=gs.qd_float, shape=(3,), needs_grad=False)
    out_grad_f.from_numpy(seed_np)
    quat_grad_manual.from_numpy(np.zeros(4))
    k_manual_bw(v_f, quat_f, out_grad_f, quat_grad_manual)
    manual_q_grad = np.array([float(quat_grad_manual[i]) for i in range(4)])

    print(f"  AD     = {ad_q_grad}")
    print(f"  manual = {manual_q_grad}")
    diff = np.abs(ad_q_grad - manual_q_grad).max()
    print(f"  max|diff| = {diff:.3e}  {'OK' if diff < 1e-10 else 'MISMATCH'}")
    print()

    # ---------- Test 2: d_quat_mul__dlhs ----------
    print("=" * 70)
    print("Test 2: d_quat_mul__dlhs (gradient w.r.t. a of quat_mul(a, b))")
    print("=" * 70)

    a_f = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=True)
    b_f = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=False)
    out4_f = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=True)
    a_grad_manual = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=False)

    @qd.kernel
    def k_mul_fwd(a: qd.template(), b: qd.template(), out: qd.template()):
        a_vec = qd.Vector([a[0], a[1], a[2], a[3]], dt=gs.qd_float)
        b_vec = qd.Vector([b[0], b[1], b[2], b[3]], dt=gs.qd_float)
        # quat_mul(a, b) = transform_quat_by_quat(b, a) per docstring
        r = gu.qd_transform_quat_by_quat(b_vec, a_vec)
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]
        out[3] = r[3]

    @qd.kernel
    def k_mul_manual_bw_a(a: qd.template(), b: qd.template(), out_grad: qd.template(), a_grad: qd.template()):
        a_vec = qd.Vector([a[0], a[1], a[2], a[3]], dt=gs.qd_float)
        b_vec = qd.Vector([b[0], b[1], b[2], b[3]], dt=gs.qd_float)
        og = qd.Vector([out_grad[0], out_grad[1], out_grad[2], out_grad[3]], dt=gs.qd_float)
        ag = d_quat_mul__dlhs(a_vec, b_vec, og)
        a_grad[0] = ag[0]
        a_grad[1] = ag[1]
        a_grad[2] = ag[2]
        a_grad[3] = ag[3]

    a_np = np.array([0.8, 0.1, 0.5, 0.3])  # arbitrary
    b_np = np.array([0.7, 0.4, 0.2, 0.55])
    a_np = a_np / np.linalg.norm(a_np)
    b_np = b_np / np.linalg.norm(b_np)
    seed4 = np.array([1.0, 0.5, -0.3, 0.7])

    a_f.from_numpy(a_np)
    b_f.from_numpy(b_np)
    out4_f.from_numpy(np.zeros(4))
    out4_f.grad.from_numpy(seed4)
    a_f.grad.from_numpy(np.zeros(4))
    k_mul_fwd(a_f, b_f, out4_f)
    k_mul_fwd.grad(a_f, b_f, out4_f)
    ad_a_grad = np.array([float(a_f.grad[i]) for i in range(4)])

    out_grad4 = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=False)
    out_grad4.from_numpy(seed4)
    a_grad_manual.from_numpy(np.zeros(4))
    k_mul_manual_bw_a(a_f, b_f, out_grad4, a_grad_manual)
    manual_a_grad = np.array([float(a_grad_manual[i]) for i in range(4)])

    print(f"  AD     = {ad_a_grad}")
    print(f"  manual = {manual_a_grad}")
    diff = np.abs(ad_a_grad - manual_a_grad).max()
    print(f"  max|diff| = {diff:.3e}  {'OK' if diff < 1e-10 else 'MISMATCH'}")
    print()

    # ---------- Test 3: d_quat_mul__drhs ----------
    print("=" * 70)
    print("Test 3: d_quat_mul__drhs (gradient w.r.t. b of quat_mul(a, b))")
    print("=" * 70)

    b_grad_manual = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=False)
    a_f2 = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=False)
    b_f2 = qd.field(dtype=gs.qd_float, shape=(4,), needs_grad=True)

    @qd.kernel
    def k_mul_fwd2(a: qd.template(), b: qd.template(), out: qd.template()):
        a_vec = qd.Vector([a[0], a[1], a[2], a[3]], dt=gs.qd_float)
        b_vec = qd.Vector([b[0], b[1], b[2], b[3]], dt=gs.qd_float)
        r = gu.qd_transform_quat_by_quat(b_vec, a_vec)
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]
        out[3] = r[3]

    @qd.kernel
    def k_mul_manual_bw_b(a: qd.template(), b: qd.template(), out_grad: qd.template(), b_grad: qd.template()):
        a_vec = qd.Vector([a[0], a[1], a[2], a[3]], dt=gs.qd_float)
        b_vec = qd.Vector([b[0], b[1], b[2], b[3]], dt=gs.qd_float)
        og = qd.Vector([out_grad[0], out_grad[1], out_grad[2], out_grad[3]], dt=gs.qd_float)
        bg = d_quat_mul__drhs(a_vec, b_vec, og)
        b_grad[0] = bg[0]
        b_grad[1] = bg[1]
        b_grad[2] = bg[2]
        b_grad[3] = bg[3]

    a_f2.from_numpy(a_np)
    b_f2.from_numpy(b_np)
    out4_f.from_numpy(np.zeros(4))
    out4_f.grad.from_numpy(seed4)
    b_f2.grad.from_numpy(np.zeros(4))
    k_mul_fwd2(a_f2, b_f2, out4_f)
    k_mul_fwd2.grad(a_f2, b_f2, out4_f)
    ad_b_grad = np.array([float(b_f2.grad[i]) for i in range(4)])

    b_grad_manual.from_numpy(np.zeros(4))
    k_mul_manual_bw_b(a_f2, b_f2, out_grad4, b_grad_manual)
    manual_b_grad = np.array([float(b_grad_manual[i]) for i in range(4)])

    print(f"  AD     = {ad_b_grad}")
    print(f"  manual = {manual_b_grad}")
    diff = np.abs(ad_b_grad - manual_b_grad).max()
    print(f"  max|diff| = {diff:.3e}  {'OK' if diff < 1e-10 else 'MISMATCH'}")
    print()

    # ---------- Test 4: d_rotvec_to_quat__drotvec ----------
    print("=" * 70)
    print("Test 4: d_rotvec_to_quat__drotvec")
    print("=" * 70)

    rv_f = qd.field(dtype=gs.qd_float, shape=(3,), needs_grad=True)
    rv_grad_manual = qd.field(dtype=gs.qd_float, shape=(3,), needs_grad=False)
    eps_f = qd.field(dtype=gs.qd_float, shape=(1,), needs_grad=False)
    EPS = 1e-15

    @qd.kernel
    def k_rv_fwd(rv: qd.template(), out: qd.template()):
        rv_vec = qd.Vector([rv[0], rv[1], rv[2]], dt=gs.qd_float)
        r = gu.qd_rotvec_to_quat(rv_vec, EPS)
        out[0] = r[0]
        out[1] = r[1]
        out[2] = r[2]
        out[3] = r[3]

    @qd.kernel
    def k_rv_manual_bw(rv: qd.template(), out_grad: qd.template(), rv_grad: qd.template()):
        rv_vec = qd.Vector([rv[0], rv[1], rv[2]], dt=gs.qd_float)
        og = qd.Vector([out_grad[0], out_grad[1], out_grad[2], out_grad[3]], dt=gs.qd_float)
        rg = d_rotvec_to_quat__drotvec(rv_vec, EPS, og)
        rv_grad[0] = rg[0]
        rv_grad[1] = rg[1]
        rv_grad[2] = rg[2]

    rv_np = np.array([0.05, -0.1, 0.07])
    seed_rv4 = np.array([0.7, 0.3, -0.4, 0.5])

    rv_f.from_numpy(rv_np)
    out4_f.from_numpy(np.zeros(4))
    out4_f.grad.from_numpy(seed_rv4)
    rv_f.grad.from_numpy(np.zeros(3))
    k_rv_fwd(rv_f, out4_f)
    k_rv_fwd.grad(rv_f, out4_f)
    ad_rv_grad = np.array([float(rv_f.grad[i]) for i in range(3)])

    out_grad4.from_numpy(seed_rv4)
    rv_grad_manual.from_numpy(np.zeros(3))
    k_rv_manual_bw(rv_f, out_grad4, rv_grad_manual)
    manual_rv_grad = np.array([float(rv_grad_manual[i]) for i in range(3)])

    print(f"  AD     = {ad_rv_grad}")
    print(f"  manual = {manual_rv_grad}")
    diff = np.abs(ad_rv_grad - manual_rv_grad).max()
    print(f"  max|diff| = {diff:.3e}  {'OK' if diff < 1e-10 else 'MISMATCH'}")


if __name__ == "__main__":
    main()
