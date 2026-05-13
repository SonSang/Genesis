"""J1 multistep — FD eps sweep at fixed N.

If our analytical backward is accurate to ~FP64 floor (1e-15), then:
  - For small eps (e.g., 1e-8): FD subtraction cancellation noise
    dominates (~eps_machine / eps) → diff grows.
  - For medium eps (1e-4 ~ 1e-5): FD truncation error O(eps²) and
    cancellation both small → diff is smallest (ana ≈ FD).
  - For large eps (1e-2): truncation O(eps²) dominates → diff grows.

The U-shape minimum tells us the actual FD floor; if our reported diff
is below that floor, our ana is well within FD truncation noise and
the "leak" we've been chasing may just be FD noise.

If instead diff is constant across eps at ~1e-9 level, it means our ana
has a real error of that magnitude.
"""

import sys
import numpy as np
import genesis as gs

sys.path.insert(0, "notes")
from diag_multistep_worst_case import TOPOLOGIES, build, loss_fn


def fd_at_eps(mjcf, n_dofs, N, seed, eps):
    sb, rb = build(mjcf, False)
    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    fd = np.zeros((N, n_dofs))
    for t in range(N):
        for d in range(n_dofs):
            sb.reset()
            for t2 in range(N):
                inp = u_list[t2].copy()
                if t2 == t:
                    inp[d] += eps
                rb.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
                sb.step()
            lp = float(loss_fn(sb).detach().cpu())
            sb.reset()
            for t2 in range(N):
                inp = u_list[t2].copy()
                if t2 == t:
                    inp[d] -= eps
                rb.control_dofs_force(gs.tensor(inp, dtype=gs.tc_float))
                sb.step()
            lm = float(loss_fn(sb).detach().cpu())
            fd[t, d] = (lp - lm) / (2 * eps)
    return fd


def ana(mjcf, n_dofs, N, seed):
    sa, ra = build(mjcf, True)
    rng = np.random.default_rng(seed)
    u_list = [rng.normal(size=n_dofs) * 0.3 for _ in range(N)]
    u_anas = [gs.tensor(u, dtype=gs.tc_float, requires_grad=True) for u in u_list]
    sa.reset()
    for t in range(N):
        ra.control_dofs_force(u_anas[t])
        sa.step()
    loss_fn(sa).backward()
    return np.array([u.grad.detach().cpu().numpy() for u in u_anas])


def main():
    gs.init(precision="64", backend=gs.cpu, performance_mode=False, logging_level="warning")
    name_map = {t[0]: t for t in TOPOLOGIES}
    _, mjcf, n_dofs = name_map["J1_free"]

    seed = 1001
    N = 5
    print(f"J1 N={N} seed={seed} — FD eps sweep")
    print("=" * 95)

    # one ana computation (independent of eps)
    a = ana(mjcf, n_dofs, N, seed)
    print(f"ana[t=0] = {a[0]}")
    print()
    print(f"{'eps':>10} | {'fd[t=0]':>50} | {'max|ana-fd|':>11}")
    print("-" * 95)

    for eps in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9]:
        fd = fd_at_eps(mjcf, n_dofs, N, seed, eps)
        diff = a - fd
        max_diff = float(np.abs(diff).max())
        print(f"{eps:>10.0e} | {str(fd[0]):>50.50} | {max_diff:>11.3e}")


if __name__ == "__main__":
    main()
