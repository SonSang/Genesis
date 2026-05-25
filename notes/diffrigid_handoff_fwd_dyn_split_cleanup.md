# Diffrigid Handoff — Step 5 split / manual reverse cleanup

날짜: 2026-05-24
브랜치: `20260512_diff_rigid_demo`
선행 commit: `423932f7` (primal-consistency fix)

---

## TL;DR

직전 commit `423932f7` (forward replay primal consistency) 이후, 과거에 *J4 N=2 cinr_pos.grad silent drop* 회피를 위해 도입했던 **Step 5 split + 3개 manual reverse** 가 **불필요한 dead complexity** 임이 확인됨. 전체 cleanup:

- **881 줄 삭제 / 9 줄 추가** (`net -872 LoC`)
- monolithic `kernel_forward_dynamics_without_qacc.grad` 한 줄 호출로 복귀
- test parity 유지 (23 passed / 2 expected fail — J4/J5 multistep)

---

## 도입 배경 (제거 대상 코드의 역사)

| Commit | 도입 내용 | 당시 동기 |
|---|---|---|
| `c4c33e33` | `kernel_split_*` 4개 (torque_and_passive_force, update_acc, update_force, bias_force) + `kernel_manual_update_force_bw` | J4 N=2 cinr_pos.grad 의 silent drop 가설 — monolithic kernel 안에서 sub-func 들이 같은 buffer 에 write 할 때 Quadrants AD 가 cross-sub-func chain 을 silent drop 한다는 가설. *Per-kernel boundary 가 chain 을 isolate* 한다는 시도. |
| `a01911e8` | `kernel_mm_*` 6개 (crb_initialize, crb_aggregate, compute_f, assemble, armature, implicit_damping_corr) | `kernel_split_compute_mass_matrix.grad` 가 50% rel-err contribution. mass_matrix 내부를 추가 sub-block 으로 분할해서 정확한 wrong source 추적. |
| `10694069` | `kernel_manual_mm_assemble_bw`, `kernel_manual_mm_crb_aggregate_bw` | 분할 후 일부 sub-block 의 auto-AD 를 manual chain rule 로 대체. |

총 영향:
- forward_dynamics.py 에 400줄 분량의 split wrapper kernel 정의
- manual_bw.py 에 405줄 분량의 3개 manual reverse kernel + helper func (`d_motion_cross_force`, `d_inertial_mul`) + 광범위 docstring 분석
- rigid_solver.py 의 backward call site 가 97줄 분량의 16개 kernel 호출 chain 으로 fragmented

**기록된 효과** (각 commit message 에서):
- J1~J5 N=1: pass (다른 fix 들 + split 조합으로)
- J4 N=2 max rel: 6.97 → 6.97 (split + manual update_force_bw, 변화 없음)
- 그 외 multistep: 개선 없음

즉 도입 시점부터 *split 자체는 J4/J5 leak 을 풀지 못함* 이 기록되어 있었지만, *J1~J5 N=1 PASS 의 필수 조건* 일 수도 있다는 보수적 판단으로 유지되었음.

---

## 검증 — `423932f7` 적용 후 split 의 진짜 효과

`423932f7` (forward replay 순서 + vel copy + cache→state load) 가 backward primal 의 일관성을 정리한 *지금* 의 코드 베이스에서, split block 전체를 monolithic `kernel_forward_dynamics_without_qacc.grad` 한 줄로 임시 교체:

| 케이스 군 | split block 사용 | monolithic 사용 |
|---|---|---|
| freejoint (4) | ✓ | ✓ |
| revolute (4) | ✓ | ✓ |
| prismatic (4) | ✓ | ✓ |
| free_with_revolute J4 (4) | ✓ | ✓ |
| revolute_chain3 J5 (4) | ✓ | ✓ |
| multistep J1_free | ✓ | ✓ |
| multistep J2_revolute | ✓ | ✓ |
| multistep J3_prismatic | ✓ | ✓ |
| multistep J4_free_rev | ✗ | ✗ |
| multistep J5_chain3 | ✗ | ✗ |

**byte-exact 동일한 PASS/FAIL 패턴**.

결론: split 의 *측정 가능한 효과 = 0*. 과거 J1~J5 N=1 PASS 의 진짜 이유는 *다른 fix 들* (manual `compute_qacc_bw` IFT, `kernel_COM_links` forward+grad pair, manual `uc_bw`, primal consistency 등) 이었음. split 은 *cross-sub-func silent drop 가설* 에 기반한 처방이었지만 *실제 wrong source 가 primal inconsistency 였기 때문에 split 처방은 무효* — 다른 fix 들이 따라잡고 split 의 효과는 사라짐.

---

## 제거 내역

### `genesis/engine/solvers/rigid/abd/forward_dynamics.py` (-400 LoC)

- 주석 헤더 "Step 5 split: per-sub-func kernel wrappers" 제거
- 11개 wrapper kernel 정의:
  - `kernel_split_compute_mass_matrix`
  - `kernel_mm_crb_initialize`, `kernel_mm_crb_aggregate`, `kernel_mm_compute_f`, `kernel_mm_assemble`, `kernel_mm_armature`, `kernel_mm_implicit_damping_corr`
  - `kernel_split_torque_and_passive_force`, `kernel_split_update_acc`, `kernel_split_update_force`, `kernel_split_bias_force`

### `genesis/engine/solvers/rigid/abd/manual_bw.py` (-381 LoC, +24 LoC `d_motion_cross_motion` 복원)

- Step 5 sub-3 manual reverse + helpers:
  - `d_motion_cross_force`, `d_inertial_mul` (helper, 다른 곳 사용 없음)
  - `kernel_manual_update_force_bw`
- Step 5 sub-4 mm manual reverses:
  - `kernel_manual_mm_assemble_bw`
  - `kernel_manual_mm_crb_aggregate_bw`
- 광범위 docstring 분석 주석 (motion_cross_force / inertial_mul 의 chain rule 도출, tree aggregation reverse 설명 등)

`d_motion_cross_motion` 헬퍼는 `kernel_manual_forward_velocity_bw` 가 여전히 사용하므로 *복원*.

### `genesis/engine/solvers/rigid/rigid_solver.py` (-98 LoC)

- import 정리 (13개 kernel)
- backward call site (line ~1652 이하 97줄) → monolithic `kernel_forward_dynamics_without_qacc.grad` 한 호출 (19줄)

---

## 남긴 dead code 후보 (별도 cleanup)

지금 cleanup 범위에는 **포함 안 됨** (별도 commit / handoff 권장):

1. **`kernel_manual_COM_links_phase5_bw`** (`manual_bw.py:~1200-1370` 영역) — 호출 site 없음. 사용 안 됨. 같은 종류의 dead manual reverse. 의존 helper (`d_qd_quat_to_R__dquat`, `d_qd_transform_inertia_by_trans_quat`, `d_qd_transform_pos_quat_by_trans_quat`) 도 *그 kernel 안* 에서만 사용 가능성 — verify 후 함께 제거.
2. **`kernel_zero_acc_smooth_bw`** + 그 호출 (rigid_solver.py:`reset_grad` line ~1319) — 이전 handoff `diffrigid_handoff_compute_qacc_dead_code_cleanup.md` 에서 dead 라고 식별됨. multi-horizon stress test 통과 확인 후 제거.
3. **`kernel_compute_qacc` wrapper kernel** (forward_dynamics.py) — `func_compute_qacc` 만 forward path 에서 사용, wrapper kernel 자체는 호출 site 없음.
4. **`kernel_forward_kinematics_entity`** (forward_kinematics.py:1119) — 비슷한 dead wrapper 가능성. verify 필요.

---

## J4/J5 multistep 의 남은 leak 에 대해

이번 cleanup 으로 *split 이 root cause 가 아님* + *primal consistency 가 root cause 가 아님* 두 가지 가설이 동시에 falsify 됨. *남은 가설들*:

- **Quaternion w-component 의 FMA-fusion divergence** (`diffrigid_handoff_n_ge_2_residual.md` 의 결론) — forward FMA-fusion chain vs manual reverse 의 mathematical-clean 식 의 FP arithmetic order divergence. 각 substep ~1e-12 → N=16 cumulative ~1e-9. *fundamental Quadrants AD / FP precision 한계*.
- 메모리 `[feedback_multistep_grad_leak]` 의 "J4/J5 silent-AD chain loss 깊은 곳" (xfail) — 동일 결론.

따라서 J4/J5 multistep 은 *별도 mechanism* 이고 *수치 한계* 라는 게 가장 가능성 높은 답. 이번 cleanup 과 무관.

---

## Cleanup 의 정성적 이득

| 측면 | 변화 |
|---|---|
| backward call site | 97 줄 fragmented chain → 19 줄 monolithic 호출 |
| 코드 베이스 LoC | -872 |
| 의존 kernel 수 | 16개 (split + manual) → 1개 (monolithic) |
| Cache 미사용 / dead 의심 영역 | 3 → 1 (kernel_manual_COM_links_phase5_bw 남음) |
| backward path 이해도 | 사용자가 *split 의 의미* 를 파악해야 했음 → *한 줄 호출* 로 의도 명확 |
| 향후 J4/J5 root cause 추적 | split 노이즈 제거 → primal/quaternion FP 한계 가설에 집중 가능 |
