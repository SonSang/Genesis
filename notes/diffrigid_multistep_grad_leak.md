# Multi-step `control_dofs_force` gradient over-counting

## Status

- **Fixed (CPU + GPU)**: J1 freejoint, J2 revolute, J3 prismatic
- **Known-fail (xfail strict)**: J4 free+revolute, J5 chain3 — deeper silent-AD bug

## The visible symptom

In a SHAC-style multi-step training loop, the user does:

```python
for t in range(N):
    x_t = gs.tensor(..., requires_grad=True)
    robot.control_dofs_force(x_t)
    scene.step()
loss = ...
loss.backward()  # one big backward through all N steps
```

For each step's `x_t`, `loss.backward()` should produce
`x_t.grad = ∂loss / ∂x_t` per FD. Before this fix, the pattern was

```
ana[t] = FD[t] * (2k + 1) / (k + 1)   where k = N - 1 - t
       = FD[t] + FD[t+1]              (equivalent form when FD scales linearly in (N-t))
```

i.e. each step's force-grad over-counts by the next step's gradient,
producing a `(2k+1)/(k+1)` ratio that converges to 2 for early steps.
For `N=10` the front step's grad is 1.9× too large.

## Root cause for J1/J2/J3: `acc_smooth_bw.grad` leak

`func_solve_mass_entity` uses a two-slot buffer `acc_smooth_bw[0/1]` for the
LDLT solve's intermediates. Backward:

- `kernel_compute_qacc.grad` atomic_adds into `acc_smooth_bw.grad[1]`
- `kernel_solve_mass_step2_reverse_bw` atomic_adds into `acc_smooth_bw.grad[0]`
- Per-DOF `kernel_solve_mass_step1_one_dof_bw.grad` consumes `acc_smooth_bw.grad[0]`

After substep `t+1`'s backward completes, `acc_smooth_bw.grad` is *not*
zeroed — it's an internal buffer. When substep `t`'s backward runs,
`kernel_compute_qacc.grad` atomic_adds again on top of the stale residue,
and the per-DOF reverse propagates the over-counted value into `force.grad`,
then into `qf_applied.grad` → `ctrl_force.grad`.

The fix is to zero `acc_smooth_bw.grad` at the start of every backward
substep (the value `acc_smooth_bw[0]` was already being zeroed for an
unrelated cross-horizon SHAC drift; we now also zero the grad slot).

### Why a kernel-side zero loop doesn't work

The original fix attempted

```python
@qd.kernel(fastcache=True)
def kernel_zero_acc_smooth_bw_grad(dofs_state):
    for i, i_d, i_b in qd.ndrange(2, ..., ...):
        dofs_state.acc_smooth_bw.grad[i, i_d, i_b] = 0.0
```

Instrumentation confirmed the kernel runs but `acc_smooth_bw.grad` is
unchanged afterwards — Quadrants silently drops `.grad` writes from inside
`@qd.kernel` (probably tracked on the adstack for tape consistency rather
than committed to memory).

The working fix uses Python-side `qd_zero_grad` which calls
`qd_to_torch(grad, copy=False).zero_()` — an in-place `memset` on the
underlying device buffer.

## Why J4/J5 still fail: silently-dropped chain on `cdof_*` / `cinr_*` / etc.

Instrumented dump after substep `t+1`'s backward (`notes/diag_full_grad_dump.py`):

```
dofs_state:
  vel              max=4e-3   (upstream — legitimate)
  acc_smooth_bw    max=4e-5   ← fixed by this patch
  cdof_ang         max=7e-6   ← leak candidate
  cdof_vel         max=1e-3   ← leak candidate
  cdofd_ang        max=4e-9
  cdofd_vel        max=1e-5
links_state:
  cinr_inertial    max=2.7e-6
  cinr_pos         max=2.5e-4
  cd_ang/cd_vel    max=8.9e-7 / 6.5e-6
  cfrc_applied_*   max=2.7e-5
  cfrc_coupling_*  max=2.7e-5
```

All of `cdof_*`, `cinr_*`, `cd_*`, `cfrc_*` are substep-internal fields
computed forward from `qpos`. Their `.grad` should chain fully down to
`qpos.grad` within one backward substep and leave zero residue. The fact
that they don't is a Phase B-family silent AD failure: a portion of the
reverse-mode chain rule is dropped by `kernel_forward_dynamics_without_qacc.grad`
or one of its nested `@qd.func` callees.

**Single-step tests** (`test_diff_fk_control_force`) pass because the
dropped contribution is below `atol`. The single-step's `.grad` is wrong
by a tiny silent amount, but the FD comparison tolerates it.

**Multi-step tests** fail because the silently-lost contribution is left
in `cdof_*.grad` etc.; at the next substep's backward, that residue is
atomic_added into the new chain, so the cumulative error grows linearly
in `N` and breaks the FD comparison.

### Why naive zeroing of leak fields doesn't fix it

Adding `qd_zero_grad(self.dofs_state.cdof_ang)` etc. before substep replay
*does* close the cross-substep leak, but it also discards the silently-
lost-but-legitimate contribution. The effect: J4/J5 multi-step still fails
(now with a different signature), and **J1 multi-step regresses** on the
free-body rotation DOFs — because J1's free joint also has `cdof_*.grad`
residue carrying a legit contribution that the silent-drop happens to
miss but that's needed for the cross-substep rotation chain.

The true fix is to find the silent drop site (similar to the Phase B
investigation that found Step 1 cross-iter AD and the trivial Step 2 mul)
and patch the chain rule there, not to paper over with bulk zeroing.

## Verifying

CPU + GPU, fp64:

```bash
pytest tests/test_diff_forward_kinematics.py::test_diff_fk_multistep_control_force -v
# J1/J2/J3 × {cpu, gpu} → 6 PASS
# J4/J5 × {cpu, gpu}   → 4 XFAIL (strict)
```

The single-step matrix continues to pass:

```bash
pytest tests/test_diff_forward_kinematics.py -v  # 88+ cases, all pass
```

## Diagnostic helpers (in `notes/`)

- `diag_multistep_leak.py` — N=2 single-DOF prismatic minimal repro; prints
  `ana[t]` / `FD[t]` / ratio. Useful for narrowing down which substep is
  responsible after any kernel change.
- `diag_multistep_leak_j4.py` — N=2 J4 free+revolute; dumps per-DOF
  ana-vs-FD diff. Use this to verify rotation-DOF leak signature when
  attempting a Phase B-style fix on `cdof_*`.
- `diag_full_grad_dump.py` — walks the dataclass `__annotations__` of every
  state struct at substep end, prints every field whose `.grad` is non-zero
  with its abs-max and norm. Essential for finding the next leak channel.
