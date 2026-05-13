# Diffrigid Handoff — Post `func_factor_mass` / `func_solve_mass` Cleanup

날짜: 2026-05-12
브랜치: `20260503_diff_rigid_demo`
가장 최근 commits:
- `06f0ce70` [MISC] cleanup factor_mass / solve_mass BW branches and dead Phase B code
- `19aa80f0` [BUG FIX] manual backward for func_compute_qacc via IFT (fixes J4 N=1 silent drop)

---

## TL;DR

**완료**: `kernel_manual_compute_qacc_bw` (IFT-based, force.grad + mass_mat.grad seed) +
Phase B 우회 코드 / BW 분기 / `mass_mat_L_bw` 필드 전부 제거.
J4 single-step (N=1) max|diff| **2.1e-8 → 7.5e-13 ~ 2.2e-12** (FP64 floor).

**남은 핵심 두 가지**:

1. **Cross-substep `.grad` leak (Task #29)** — backward 의 첫 substep (t=N-1) 는
   FP64-floor 깨끗하지만, 두 번째부터 ~1e-4 rel error 가 들어옴. J1/J4/J5
   multistep test 가 이 leak 으로 fail. Leak source 는 `cdof_*` / `cinr_*` /
   `cd_*` / `cfrc_*` 중 일부지만, 단순 zero 로는 해결 안 됨 (legitimate chain
   contributions 도 같이 손실).

2. **`kernel_compute_mass_matrix.grad` chain dead-end** —
   `kernel_manual_compute_qacc_bw` 가 IFT 으로 `mass_mat.grad` 까지 seed 하지만,
   downstream 의 `kernel_compute_mass_matrix.grad` 가 호출 안 됨 (Quadrants AD
   reject: `reverse_segments@68 Invalid program input for autodiff`). 이 chain 이
   끊긴 게 J4/J5 multistep silent drop 의 핵심 원인 중 하나일 가능성 높음.

---

## 현재 코드 상태

### 단순화된 backward path (`substep_pre_coupling_grad`)

```python
def substep_pre_coupling_grad(self, f):
    self._is_backward = True
    self._debug_grad_dump(f"f={f} ENTRY (before prepare_backward_substep)")

    kernel_prepare_backward_substep(...)  # restore pre-integrate qpos/vel from cache

    if self._requires_grad:
        kernel_zero_acc_smooth_bw(self.dofs_state)
        qd_zero_grad(self.dofs_state.acc_smooth_bw)
        # KNOWN ISSUE: cdof_*/cinr_*/cd_*/cfrc_* .grad NOT zeroed here
        # (see Task #29). Naive zeroing breaks other chain contributions.

    self.substep(f)  # forward replay (kernel_step_1, kernel_step_2 with is_backward=True)
                     # produces mass_mat, mass_mat_L, mass_mat_D_inv, acc_smooth, vel_next, qpos_next primals

    kernel_copy_next_to_curr_no_check(...)  # qpos = post-integrate for UCS.grad

    # === UCS / forward_velocity / COM_links forward + .grad (per-link split) ===
    # (omitted — unchanged)

    kernel_begin_backward_substep(...)
    kernel_step_2.grad(...)  # integrator backward: qpos.grad / vel.grad propagation

    # Manual LDLT solve backward (IFT) — replaces:
    #   kernel_compute_qacc.grad + Phase B externals + Stage A/B/C reverses
    kernel_manual_compute_qacc_bw(
        dofs_state=self.dofs_state,
        entities_info=self.entities_info,
        rigid_global_info=self._rigid_global_info,
        static_rigid_sim_config=self._static_rigid_sim_config,
    )
    # Seeds:
    #   acc_smooth.grad += acc.grad; acc.grad = 0; acc_smooth.grad = 0
    #   force.grad += M^-1 . acc_smooth.grad
    #   mass_mat[i,j].grad += -force_contrib_i * acc_smooth_j (+ mirror for i>j)

    kernel_copy_acc(...)
    kernel_forward_dynamics_without_qacc.grad(...)
    # Body now contains only: torque_and_passive_force, update_acc,
    # update_force, bias_force. Quadrants AD reverses these in isolation.
    # `func_compute_mass_matrix` + `func_factor_mass` NOT inside — their
    # forwards still run via `kernel_step_1 → func_forward_dynamics` upstream.
```

### 새 manual kernel

`genesis/engine/solvers/rigid/abd/forward_dynamics.py` 에 `kernel_manual_compute_qacc_bw`
(약 90 줄). 핵심 식:

```
# Reverse of forward `acc[i] = acc_smooth[i]`:
seed[i_d] = acc_smooth.grad[i_d] + acc.grad[i_d]
acc.grad[i_d] = 0; acc_smooth.grad[i_d] = 0

# LDLT reverse solve (M = L D L^T → M^{-1} = L^{-T} D^{-1} L^{-1}):
# Same algorithm as forward solve_mass since M is symmetric.
Step 1: solve L^T u = seed       (descending i_d)
Step 2: v = D^{-1} u
Step 3: solve L delta = v        (ascending i_d)
force.grad[i_d] += delta[i_d]

# IFT for mass_mat (lower-tri storage):
mass_mat.grad[i, i] -= delta[i] * acc_smooth[i]
mass_mat.grad[i, j] -= delta[i] * acc_smooth[j] + delta[j] * acc_smooth[i]  (i > j)
```

Scratch buffer: `dofs_state.acc_smooth_bw[0/1]` (forward intermediates 가 dead 한
시점에 overwrite).

### 단순화된 forward (`func_forward_dynamics`, `kernel_forward_dynamics`)

`kernel_step_1 → func_forward_dynamics` 는 **변화 없음** — 여전히 6 개 func
(compute_mass_matrix, factor_mass, torque, update_acc, update_force,
bias_force, compute_qacc) 호출. 단 `func_factor_mass` / `func_solve_mass_entity`
는 이제 `is_backward` 파라미터 없음, forward 전용.

`kernel_forward_dynamics_without_qacc` 는 `func_compute_mass_matrix` 와
`func_factor_mass` 호출 제거 (`.grad` 호출 시 Quadrants AD 가 그것들의
reverse 를 만들지 않도록).

---

## 검증 상태

### 통과

| 검사 | 결과 |
|---|---|
| `notes/diag_j4_n1_mismatch.py` (J4 N=1, 3 seeds) | max\|diff\| ≤ 2.2e-12 |
| `notes/diag_j1_n1_relerror_sweep.py` (J1 N=1, 10 seeds) | rel error ~1e-12 |
| `notes/diag_j1_n2_relerror_sweep.py` (J1 N=2 t=1) | rel error ~1e-12 |
| `pytest test_diff_fk_freejoint[single-fp64-cpu]` | PASS |
| `pytest test_diff_fk_revolute[single-fp64-cpu]` | PASS |
| `pytest test_diff_fk_prismatic[single-fp64-cpu]` | PASS |
| `pytest test_diff_fk_free_with_revolute[single-fp64-cpu]` | PASS |

### 실패 (known issue, Task #29)

| 검사 | 결과 |
|---|---|
| `notes/diag_j1_n2_relerror_sweep.py` (J1 N=2 t=0) | **rel error ~1e-4** (cross-substep leak) |
| `pytest test_diff_fk_multistep_control_force[J1_free-cpu]` (N=10) | FAIL, translation diff 8.6e-6 |
| `pytest test_diff_fk_multistep_control_force[J4_free_rev-cpu]` (N=10) | XFAIL (여전히, leak + mass chain dead-end 합쳐서) |
| `pytest test_diff_fk_multistep_control_force[J5_chain3-cpu]` (N=10) | XFAIL |

---

## 남은 작업

### 1. Cross-substep `.grad` leak (Task #29) — 가장 시급

**증상**: backward 의 t=N-1 (첫 번째 처리) 는 FP64 floor 깨끗. t=N-2 부터 1e-4 rel
error 가 들어옴. N=10 multistep test 에서 translation grad 가 0.22% 어긋남.

**진단 결과** (`notes/diag_j1_n2_substep_leak.txt`):

ENTRY 시점에 stale `.grad` 가 있는 fields (substep t=0 시작 시):
```
qpos.grad           1.489e-04   ← 정상 chain (loss → qpos[0])
vel.grad            1.489e-06   ← 정상 chain (qpos[0]_next ← vel[0])
links_state.cd_vel.grad          1.351e-09   ← LEAK?
links_state.cd_ang.grad          6.387e-12   ← LEAK?
dofs_state.cdof_vel.grad         1.416e-12   ← LEAK?
dofs_state.cdof_ang.grad         1.615e-27   ← LEAK?
dofs_state.cdofd_vel.grad        8.804e-10   ← LEAK?
links_state.cinr_pos.grad        1.443e-10   ← LEAK?
links_state.cinr_mass.grad       2.588e-13   ← LEAK?
links_state.cfrc_applied_vel.grad  1.489e-08  ← LEAK?
links_state.cfrc_coupling_vel.grad 1.489e-08  ← LEAK?
```

**시도한 fix** (실패):

* 위 모든 fields 의 `.grad` 를 `qd_zero_grad` 로 zero: J1 multistep **더
  악화** (8.6e-6 → 6.2e-5). 일부 fields 의 `.grad` 는 *legitimate cross-substep
  chain* 의 일부.

* 가장 큰 두 개 (`cfrc_applied_vel`, `cfrc_coupling_vel` = 1.5e-8) 만 zero:
  **변화 없음** (8.6e-6). 이들은 substep 내부에서 properly consume.

**다음 시도 후보**:

1. **Bisection 으로 leak field 식별**: 위 9 개 fields 중 절반씩 zero 하면서
   J1 N=2 t=0 의 rel error 변화 확인. binary search.

2. **`_debug_grad_dump` 의 field 목록 확장**: 위 list 에 없는 fields
   (`joints_state.xanchor/xaxis`, `links_state.{i_pos, i_quat, pos_bw,
   quat_bw}`, `qf_smooth/applied/bias/passive`) 도 dump 받아 새로운 leak source
   확인.

3. **Forward chain 추적**: `kernel_forward_dynamics_without_qacc.grad` 가
   어떤 fields 를 write 하는지 분석 (예: 위 의심 fields). 그 chain 이 chain
   완료되지 않고 다음 substep 까지 carryover 되면 leak.

4. **각 substep 시작 시 stale .grad → zero, 끝에는 그대로**: leak 후보를 *반드시*
   가져가야 할 chain 과 분리하기 위한 실험.

### 2. `kernel_compute_mass_matrix.grad` chain wire-in (Task #36 의 연장)

**현재 상태**: `kernel_manual_compute_qacc_bw` 가 `mass_mat.grad` 를 seed 하지만,
다운스트림 chain 끊김 — `kernel_compute_mass_matrix.grad` 가 호출 안 됨.

**원인**: Quadrants AD 가 `kernel_compute_mass_matrix` body 의 mixed
for-loops + statements 를 reverse 컴파일 시 reject (`reverse_segments@68
Invalid program input for autodiff`).

**해결 방향 (사용자가 시사함)**:

* **manual backward 작성** (`kernel_manual_compute_mass_matrix_bw`).
  `kernel_manual_compute_qacc_bw` 와 같은 패턴.
  - 입력: `mass_mat.grad` (이미 manual qacc kernel 이 seed 함)
  - 출력: `cinr_*.grad`, `cdof_*.grad`, `crb_*.grad`, `links_state.pos.grad`,
    `links_state.quat.grad` 등 (`mass_mat = func(cinr, cdof, ...)` 의 chain rule)

* Forward 식 (`func_compute_mass_matrix` in `forward_dynamics.py:274`):
  ```
  for i_d, j_d:
      mass_mat[i_d, j_d] = f_ang[i_d].dot(cdof_ang[j_d]) + f_vel[i_d].dot(cdof_vel[j_d])
                          * mass_parent_mask[i_d, j_d]
  ```
  Reverse:
  ```
  f_ang.grad[i_d] += cdof_ang[j_d] * mass_mat.grad[i_d, j_d] * mass_parent_mask[...]
  f_vel.grad[i_d] += cdof_vel[j_d] * mass_mat.grad[i_d, j_d] * mass_parent_mask[...]
  cdof_ang.grad[j_d] += f_ang[i_d] * mass_mat.grad[i_d, j_d] * mass_parent_mask[...]
  cdof_vel.grad[j_d] += f_vel[i_d] * mass_mat.grad[i_d, j_d] * mass_parent_mask[...]
  ```
  여기서 `f_ang`, `f_vel` 의 forward 식도 따라가야 함 (crb_inertia 부터).

* 이게 완성되면 mass_mat chain → cdof_*.grad → links_state.{pos,quat}.grad →
  qpos.grad 로 흐름. J4/J5 multistep silent drop 의 핵심 누락 부분 해결될 가능성.

---

## 진단 인프라

`notes/diag_*.py` (모두 `notes/diag_multistep_worst_case.py` 의 `measure()` /
`TOPOLOGIES` 활용):

| 파일 | 용도 |
|---|---|
| `diag_multistep_worst_case.py` | J1~J5 × N∈{4,16,32} × 3 seeds sweep — overall progress check |
| `diag_j4_n1_mismatch.py` | J4 N=1 per-DOF ana vs FD (3 seeds) — 핵심 single-step 검증 |
| `diag_j1_n1_relerror_sweep.py` | J1 N=1 (10 seeds) rel error — baseline (cross-substep 없음) |
| `diag_j1_n2_relerror_sweep.py` | J1 N=2 t=0 vs t=1 rel error — **cross-substep leak signature** |
| `diag_j1_n2_substep_leak.py` | J1 N=2 with `GENESIS_DEBUG_GRAD=1` — substep entry 시점 모든 fields `.grad` dump |
| `diag_j1_n4_perdof.py` | J1 N=4 per-DOF per-step ana vs FD — leak 의 step-by-step 누적 양상 |

기존 dump 인프라 (`_debug_grad_dump` in `rigid_solver.py:1283`) 에 ENTRY
dump 가 추가됨 (line 1370). `GENESIS_DEBUG_GRAD=1` 으로 활성.

`notes/diffrigid_solve_mass_manual_bw_handoff.md` — 이번 commit 작업 시점
이전의 핸드오프 (manual backward 작성 시작 시점).

---

## 파일별 변경 요약

`genesis/engine/solvers/rigid/abd/forward_dynamics.py` (-490 줄):
- Removed: `func_solve_mass_entity_step1_one_dof_bw`,
  `kernel_solve_mass_step1_one_dof_bw`,
  `kernel_factor_mass_stage_{a, b_pair, c}_bw`,
  `kernel_solve_mass_step2_reverse_bw`
- Removed: `func_factor_mass` 의 BW 분기 (else 절), `is_backward` 파라미터
- Removed: `func_solve_mass_entity` / `func_solve_mass_batch` /
  `func_solve_mass` 의 BW 분기, `out_bw` / `is_backward` 파라미터
- Removed: `kernel_forward_dynamics_without_qacc` 의
  `func_compute_mass_matrix` / `func_factor_mass` 호출
- Added: `kernel_manual_compute_qacc_bw` 의 IFT mass_mat.grad seeding 부분

`genesis/engine/solvers/rigid/rigid_solver.py` (-78 줄):
- Removed: Stage A/B/C 의 .fwd + .grad 호출 (substep_pre_coupling_grad)
- Removed: `kernel_solve_mass_step1_one_dof_bw` × N 호출 (.fwd + .grad)
- Removed: `kernel_solve_mass_step2_reverse_bw` 호출
- Removed: `kernel_compute_qacc.grad` 호출
- Removed: 위 kernel 들의 imports
- Replaced with: `kernel_manual_compute_qacc_bw` 단일 호출
- Added: ENTRY-time `_debug_grad_dump` (diagnostics)
- Removed: `mass_mat_L_bw` 의 dump entry

`genesis/utils/array_class.py` (-8 줄):
- Removed: `mass_mat_L_bw` field (RigidGlobalInfo schema + allocator)
- Removed: `mass_mat_shape_bw` 계산 + 검증 코드

---

## 사용자 명시 가이드 (작업 도중)

1. **BW 에 따라 연산 누락 시키는 건 절대 안 됨** — `func_factor_mass` /
   `func_solve_mass_entity` 의 BW 분기 모두 제거. Forward path 는 모든
   mode 에서 동일하게 standard Cholesky / LDLT solve.

2. **Manual backward kernel 로 우회로 정리**. Quadrants AD 한계 우회
   (silent drop, Phase B externals) 는 임시 단계로 봤음. 이제 manual kernel
   하나로 통합 → 깨끗.

3. **IFT 끝까지 적용**. `force.grad` 만이 아니라 `mass_mat.grad` 까지 직접
   seeding. 그러면 `func_factor_mass` 의 backward chain (Stage A/B/C reverse)
   완전히 dead.

4. **응답은 한국어 + 코드 주석/문자열은 영어**.

5. **Context compact 후 자동 resume 금지**. 사용자에게 한국어로 먼저 허락 받기.

---

## 즉시 시작 가능한 다음 단계

가장 적절한 순서:

1. **Task #29 (cross-substep leak)** 의 bisection 진단부터.
   `notes/diag_j1_n2_relerror_sweep.py` 를 baseline 으로,
   `rigid_solver.py:substep_pre_coupling_grad` 의 acc_smooth_bw zero 구간에
   후보 fields zero 호출을 절반씩 추가/제거하면서 rel error 측정.

2. 만약 leak 가 fields 자체가 아니라 chain rule 의 식 자체에서 발생한다면
   (= zero 가 fix 안 됨), `kernel_compute_mass_matrix.grad` chain 의 dead-end
   가 원인일 수 있음. → Task #36 의 연장으로
   `kernel_manual_compute_mass_matrix_bw` 작성으로 전환.
