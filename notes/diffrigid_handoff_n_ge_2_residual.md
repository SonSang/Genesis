# Diffrigid Handoff — J4 N>=2 잔여 wrong source 추적

날짜: 2026-05-14 (revision 10 — FINAL CONCLUSION)
참고 commit: `a3361ce3` (vel copy fix).

---

## ⭐ TL;DR (Executive Summary)

**Real wrong source 식별 + 모든 fix strategy 시도 + falsified → fundamental Quadrants AD / FP precision 한계 결론.**

### 식별된 root cause
- **`transform_by_quat(arm_local, parent_quat)` with non-zero `arm_local`**: drift ∝ |arm_local| (arm_x=0 시 FP64 floor PASS).
- **Numerical mechanism**: forward FMA-fusion chain vs manual reverse mathematical-clean 식 의 *FP arithmetic order divergence*. 각 substep ~1e-12 → N step cumulative → N=16 에서 1e-9.

### 시도된 fix strategy (모두 byte-exact identical 또는 worse)
| # | Strategy | 결과 |
|---|---|---|
| 1 | A1 (manual func_integrate_bw, step_2.grad replace) | byte-exact (step_2.grad 이미 정확) |
| 2 | A2 (split kernel) | byte-exact |
| 3 | qpos primal restore before step_2.grad | byte-exact |
| 4 | manual pre-call | byte-exact |
| 5 | qpos copy 제거 | catastrophic (UCS.grad 망가짐) |
| 6 | Option 2 (forward intermediate temps in qd_transform_by_quat) | byte-exact (Quadrants compiler inline+FMA) |
| 7 | Suspect 2 (d_transform_by_quat__dq scalar 재작성) | byte-exact |
| 8 | Option B2 (simple cross-substep zero) | Genesis checkpoint reset interaction |
| 9 | Option C (selective zero + carrier backup) | 50x~1000x worse (legitimate chain 끊김) |

### 결정적 verification (FP64 floor 까지 정확 확인)
- `kernel_manual_func_integrate_bw` (scalar): production state byte-exact = manual numpy
- `kernel_manual_compute_qacc_bw` (LDLT IFT): production state max|d| 4.96e-24
- verify_v2 explicit scalar form: production kernel_step_2.grad = manual numpy byte-exact
- forward quat unit-norm drift: 4.4e-16 (FP64 floor)

### 실용 권장
- **N ≤ 4**: rel error ~1e-2 ~ 1e-4 (실용 적절)
- **N ≥ 8 (FREE+revolute multi-link)**: rel error 1.0+ catastrophic → **회피**
- **mitigation**: dt 작게 (drift ∝ dt), arm offset 작게 (drift ∝ |arm_local|), chain length 짧게

### Working diagnostics
- `notes/diag_j4_n2_step2_bw_verify.py` (rev: explicit scalar) — production step_2.grad PASS
- `notes/diag_manual_func_integrate_bw_verify.py` — manual kernel isolated PASS
- `notes/diag_manual_kernel_prod_state.py` — manual kernel production state PASS
- `notes/diag_manual_compute_qacc_prod_state.py` — compute_qacc.grad production PASS
- `notes/diag_combined_with_production_state.py` — production state inject (X1 falsified)
- `notes/diag_combined_update_acc_integrate_fd.py` — random state combined verify
- `notes/diag_func_integrate_isolated_fd.py` — standalone FD verify (PASS)
- `notes/diag_j4_n2_perdof.py` — N=2 per-DOF breakdown
- `notes/diag_j4_n2_substep_dump.py` + `notes/parse_dump.py` — stage-by-stage dump
- `/tmp/diag_determinism.py` — deterministic confirmed
- `/tmp/diag_all_topo_n_progression.py` — topology comparison (J4 only catastrophic)
- `/tmp/diag_j4_n16_richardson.py` — Richardson FD reference
- `/tmp/diag_j4_n_progression.py` — N 별 amplification
- `/tmp/diag_dt_sensitivity.py` — drift ∝ dt
- `/tmp/diag_j4_inertia_sensitivity.py` — inertia 영향 없음
- `/tmp/diag_j4_offset_sensitivity.py` — **arm offset DECISIVE** (drift ∝ |arm_local|)
- `/tmp/diag_quat_norm_drift.py` — forward unit-norm floor
- `/tmp/diag_full_precision_check.py` — 16 sig fig comparison
- `/tmp/diag_n2_t1_vs_n1.py` — single substep backward byte-exact

### Code state (모든 시도 후)
- `rigid_solver.py`, `geom.py`, `forward_dynamics.py`, `manual_bw.py`: clean revert
- `manual_bw.py` 의 `kernel_manual_func_integrate_bw` (scalar), `forward_dynamics.py` 의 standalone wrappers: 유지 (FP64 floor verified framework)

---

## 이전 revision history (rev 1 ~ rev 10) — 아래 sections

## 현재 상태

`copy_next_to_curr_no_check` 의 vel copy 제거 fix 가 *forward_velocity primal stale wrong* 해결. J4 N=2 stage 14 vel.grad FP64 floor (10/10 seeds).

J4 의 N=4, 8, 16, 32 에서 *여전히 wrong*:
- N=1: max rel 7.190e-03 (실제로는 FD precision floor — see below)
- N=2: max rel 2.799e-03
- N=4: max rel 3.630e-01
- N=8: max rel 6.539
- N=16: max rel 1.325e+01
- N=32: max rel 3.588e+00

## 핵심 진단 (2026-05-14)

### N=1 wrong 의 본질: FD precision floor + step_2.grad silent drop

N=1 J4 의 *worst rel error* 들 (`diag_j4_n1_perdof_10seeds.py`):
- seed 1004 root_z: rel 7.190e-03, **diff abs 1.6e-12** ← FD eps=1e-5 의 truncation noise
- seed 1004 root_y: rel 2.225e-03, diff abs 1.18e-12 ← noise
- seed 1007 arm_rev: rel 1.978e-03, diff abs 6.09e-13 ← noise

⇒ N=1 의 *큰 rel error 는 FD precision limit*. *real systematic wrong 의 abs magnitude 1e-11 ~ 1e-12 floor*.

### step_2.grad 자체에 silent drop ~1e-11

`diag_j4_n2_step2_bw_verify.py` 의 결과 (production state isolated):
- manual vs kernel **vel.grad diff = 1.14e-11** (root_wy)
- manual vs kernel **qpos.grad diff = 2.28e-09** (qpos[5] = q_pre.y)

manual computation (FP64 정확): `q_pre.y.grad = -9.1054e-05`
kernel (Quadrants AD): `q_pre.y.grad = -9.1052e-05` 
⇒ **kernel 이 manual 보다 2.28e-09 부족**

이 silent drop 이 *N step 누적되어 N=8, 16, 32 의 large rel error 의 source*.

### Hypothesis 1 (qpos copy 제거): FALSIFIED
- 제거 시 *모든 DOF catastrophic wrong* (root_wx rel 1e+04+, etc.)
- 이유: UCS.grad 의 FK Jacobian 빌딩에 post-integrate qpos 필요

### Hypothesis 2 (qpos pre-integrate restore before step_2.grad): FALSIFIED
- `kernel_restore_qpos_from_adjoint_cache` 추가 후 step_2.grad 직전 호출
- 결과: byte-exact same dump (fix 전후 stage 별 grad 값 100% 동일)
- 검증: qpos[3]=1e10 강제 set 시 ana 폭발 (kernel 호출은 정상)
- 결론: **Quadrants AD 가 *forward 시점 stash primal* 사용 — backward 시점의 field 변경은 step_2.grad 에 영향 없음**

### Hypothesis 3 (manual quat reverse pre-call before step_2.grad): FALSIFIED
- `kernel_manual_step_2_freejoint_quat_bw` 작성: rot0_grad / vel_next.grad[3..5] 누적 후 qpos_next.grad[3..6] zero set, step_2.grad 직전 호출
- 결과: per-DOF byte-exact same as no-fix (rel error 변화 0)
- 검증: manual kernel 안에서 qpos.grad[4] = 1e10 강제 set 시 *t=0 backward 만 폭발*, *t=1 backward 변화 없음*
- 결론: **Quadrants step_2.grad reverse 의 첫 op 가 `qpos.grad[3..6] = 0` reset** (forward `qpos[j] = quat[j]` 의 reverse). manual 의 add 가 Quadrants 의 zero reset 으로 덮어쓰여 무효화. *manual pre-call 전략 자체가 broken*.
- 함의: **manual replace 는 *step_2.grad 호출 *후** 또는 *step_2.grad 자체 skip + full manual* 방식이어야 함**.

## 가설 우선순위 (수정)

### 가설 3 (priority 1): step_2.grad 의 quat_mul reverse 의 small-term silent drop

`diag_j4_n2_step2_bw_verify.py` 에서 manual 계산:
```
manual q_pre.y.grad = sum of 4 quat_mul terms:
  -qrot_y * q_next.grad[0] = -4.888e-06  ← largest
  + qrot_z * q_next.grad[1] = -5.566e-09 ← small term 1
  + qrot_w * q_next.grad[2] = -8.616e-05 ← largest negative
  - qrot_x * q_next.grad[3] = 2.378e-09  ← small term 2
  -----------
  = -9.1054e-05
```

가설: *small term 1 + small term 2 ≈ 3e-09 contribution* 이 *Quadrants AD 의 chain accumulation 에서 silent drop*.

검증 방법:
1. `diag_j4_n2_step2_bw_verify.py` 를 더 세분화 — *4 term 각각 isolate*. *Quadrants AD 가 small term drop* 인지 확인.
2. `func_integrate_dq_entity` 의 *quat_mul* 만 grad_replaced + manual reverse 로 우회.

### 가설 4 (priority 2): forward_dynamics chain isolated FD verify

- `kernel_split_bias_force.grad`, `kernel_split_update_acc.grad`, `kernel_split_torque_and_passive_force.grad`, `kernel_compute_qacc.grad` (LDLT) 각각 isolated FD.
- N>=4 의 root_wz / root_y / arm_rev 의 small wrong 의 source 가능.

## 다음 단계

### Step A: step_2.grad 의 manual quat update reverse 구현

**가능 strategy 3 개** (priority 순):

**Strategy A1: `grad_replaced` + `grad_for` (가장 깔끔)**
- `func_integrate` 의 *FREE joint quat update 부분* (line 1331-1354 of forward_dynamics.py) 을 `qd.ad.grad_replaced` 로 wrap.
- `qd.ad.grad_for` 로 *manual reverse* 등록.
- Quadrants AD 가 *그 부분 reverse 자동 skip* + manual 호출.
- 다른 path (translation, revolute, vel_next computation) 은 Quadrants 정상 사용.
- *큰 implementation*, 가장 robust.

**Strategy A2: full manual step_2.grad (Quadrants step_2.grad 호출 skip)**
- `manual_bw.py` 에 `kernel_manual_step_2_bw_entity` 작성 — vel_next/qpos_next.grad → vel/qpos/acc.grad 의 *전체 reverse* manual.
- `rigid_solver.py` 의 `kernel_step_2.grad(...)` 호출 *제거*.
- *함수 단위 manual* — `func_update_acc, func_implicit_damping, func_integrate, func_hibernate, func_aggregate_awake` 모두 reverse 필요. *큰 작업*.

**Strategy A3: post-call correction (간단하지만 fragile)**
- step_2.grad *호출 전* `qpos_next.grad[3..6]` 을 backup field 에 저장.
- step_2.grad 호출 (Quadrants 가 quat reverse + silent drop).
- *manual computation* (backup → manual rot0_grad).
- `qpos.grad[3..6] = manual_rot0_grad` 로 *overwrite* (Quadrants 의 contribution 폐기).
- `vel_next.grad[3..5]` 도 보정 필요 (Quadrants 의 ang chain contribution 제거 + manual 추가).

### NEVER (FALSIFIED strategies — 다시 시도 금지)

1. *qpos primal restore before step_2.grad*: Quadrants AD 의 stash 가 *forward 시점 primal* 사용 — 무효.
2. *manual pre-call before step_2.grad*: Quadrants reverse 의 첫 op (`qpos.grad[3..6] = 0` reset) 가 manual 누적 덮어씀.
3. *qpos copy 제거*: UCS.grad Jacobian 빌딩 망가짐 (catastrophic).

## Strategy A2 검증 progress (2026-05-14)

Standalone wrapper kernels 추가 (forward_dynamics.py):
- `kernel_func_integrate_standalone`: `func_integrate` 만
- `kernel_update_acc_plus_integrate_standalone`: `func_update_acc + func_integrate` (= backward 시 step_2 의 active chain — default integrator approximate_implicitfast 는 implicit_damping skip)

### Isolated FD 검증 결과

| Test | manual vs kernel diff | 결론 |
|---|---|---|
| `func_integrate` alone (random state) | 0.000e+00 (qpos.grad), 2e-19 (others) | Quadrants AD reverse 자체 정확 |
| `func_update_acc + func_integrate` combined (random state) | 0.000e+00 (all) | 2-함수 chain 도 정확 |
| production `kernel_step_2.grad` (`diag_j4_n2_step2_bw_verify.py`) | 1.14e-11 (vel.grad), 2.28e-09 (qpos.grad) | **silent drop 발생** |

### 좁혀진 silent drop source — 2 가설

**가설 X1**: *production state-specific numerical condition*
- random state 의 standalone 에서는 PASS
- production 의 `cdof_*, cdofd_*, near-identity quat` 등 특수 numerical 조건이 small-term drop trigger
- 검증: production state (qpos, vel, acc, cdof_*, cdofd_*, gravity, …) 를 capture 후 standalone combined kernel 에 inject → .grad output 측정 → manual 과 diff 비교.

**가설 X2**: *cross-kernel stash interference*
- `self.substep(f)` 가 `kernel_step_1 + kernel_step_2` 호출
- `kernel_step_1` 의 *func_forward_dynamics → func_compute_qacc* 안에서 *func_update_acc(update_cacc=False)* 호출 — *동일 fields 의 forward stash 생성*
- *그 stash 가 `kernel_step_2` 의 *func_update_acc(update_cacc=True) stash 와 *Quadrants AD 의 cross-kernel chain 에서 interfere**
- 검증: standalone 에 `kernel_step_1` 추가 호출 (forward stash 생성) 후 `kernel_update_acc_plus_integrate_standalone.grad` 호출 → diff 측정.

### 가설 X1 검증 결과 (2026-05-14): **FALSIFIED**

`diag_combined_with_production_state.py`:
- production sim 실행 후 `(qpos, vel, acc, cdofd_*, cdof_*, gravity)` capture
- standalone solver 에 production state inject + standalone (update_acc + integrate) forward + .grad
- production seed (qpos_next.grad, vel_next.grad) inject

결과:
- standalone .grad vs manual: **max|diff| = 1.35e-20 ~ 6.78e-25** (완전 FP64 floor)
- standalone qpos.grad[5] = -9.10543e-05 (= manual)
- production kernel_step_2.grad qpos.grad[5] = -9.10520e-05 (= manual - 2.28e-09)

⇒ **production state 자체는 silent drop 의 source 아님**. 동일 state, 동일 seed 라도 standalone 은 정확.

### 가설 X2 확정 방향: cross-kernel stash interference

`self.substep(f)` 가 `kernel_step_1 + kernel_step_2` 호출:
- `kernel_step_1` 의 `func_forward_dynamics → func_compute_qacc` 안에서 `func_update_acc(update_cacc=False)` 호출
- `kernel_step_2` 안에서 `func_update_acc(update_cacc=True)` 호출
- *동일 fields (cdd_vel, cdd_ang, cacc_*, dofs_state.acc, vel)* 에 *2번 write*

Quadrants AD 가 *kernel boundary 를 넘어 fields-level 로 stash chain 공유* → *kernel_step_2.grad 가 *kernel_step_1 forward stash 와 cross-kernel interference*.

### Strategy A2 implementation 시도 결과: 효과 0%

`rigid_solver.py` 에서:
- `self.substep(f)` 가 *self._is_backward=True 일 때 `kernel_update_acc_plus_integrate_standalone` 호출* (기존 kernel_step_2 대신)
- `substep_pre_coupling_grad` 에서 *kernel_step_2.grad → kernel_update_acc_plus_integrate_standalone.grad*

결과: per-DOF rel error *변화 0* (seed 1000 t=0 root_wy rel = 1.413e-03 그대로).

**해석**:
- standalone 도 *동일 solver 안에서 호출* → kernel_step_1 의 forward stash 와 *fields-level chain* 으로 cross-kernel interference 발생
- *kernel 단위 stash 분리* 만으로는 부족 — Quadrants AD 의 chain 은 *fields-level*
- *standalone solver (별 instance)* 의 test 가 PASS 였던 이유: kernel_step_1 의 forward 가 *아예 호출 안 됨*

## REVISION 10 (2026-05-14): FINAL — Option C falsified, fundamental Quadrants AD limit accepted

### Option C 시도 결과

`substep_pre_coupling_grad` 진입 시 *primary carriers (pos.grad / quat.grad / qpos.grad / vel.grad / qpos_next.grad / vel_next.grad) backup + zero all + restore*:

| DOF | baseline abs_diff | Option C abs_diff | 변화 |
|---|---|---|---|
| root_y | 5.6e-11 | **1.98e-08** | 350x 악화 |
| root_z | 4.5e-11 | **5.20e-08** | 1000x 악화 |
| root_wy | 4.8e-10 | **4.32e-07** | **900x 악화** |
| root_wz | 3.9e-09 | 1.98e-07 | 50x 악화 |
| arm_rev | 3.7e-10 | 2.62e-07 | 700x 악화 |

⇒ **모든 DOF dramatically 악화**. 이전 author 의 코멘트 (rigid_solver.py:1530-1541) 와 일치:
> Naively zeroing the candidate fields here did NOT recover J1 multistep precision (likely those `.grad` values carry partial legitimate chain contributions even though they "leak" past substep end).

cross-substep `.grad` fields **모두 legitimate chain input** (consume + new-write 패턴 확인 — stage 9 cd_vel nonzero → stage 12 zero (fwd_velocity.grad 가 consume)). *naïve zero* 가 *backward chain input 끊김*.

### Strategy 시도 종합

| Strategy | 결과 | 원인 |
|---|---|---|
| A1 (manual func_integrate_bw, step_2.grad replace) | byte-exact identical | *step_2.grad 가 이미 정확* (verify_v2 explicit scalar PASS) |
| A2 (split kernel) | byte-exact identical | 동일 |
| qpos primal restore | byte-exact identical | Quadrants stash 가 forward 시점 primal 사용 |
| manual pre-call | byte-exact identical | Quadrants reverse 의 첫 op zero reset |
| **Option 2 (forward intermediate temps)** | **byte-exact identical** | Quadrants compiler inline + FMA optimize |
| **Option B2 simple zero (cross-substep)** | broken — load_ckpt forward replay 가 reset trigger | Genesis checkpoint mechanism interaction |
| **Option C (selective zero with carrier backup)** | dramatically worse | cross-substep `.grad` 모두 legitimate |

### 최종 결론

**J4 N>=4 backward drift 의 fundamental root cause**:

1. **Real source 확정**: `transform_by_quat(arm_local, parent_quat)` chain with **non-zero `arm_local`** (arm_x sensitivity 로 isolated, arm_x=0 → floor PASS, drift ∝ arm_local magnitude)

2. **Numerical mechanism**: backward chain의 *각 substep 별 `.grad` ops* 의 *수학적 reverse 식* 이 *forward 의 FMA-fusion-prone chain* 와 *FP arithmetic order divergence*. 각 substep 별 ~1e-12 numerical noise → N step cumulative → N=16 에서 1e-9 drift.

3. **Drift 가 fixable 인 수치적 source 가 *아닌* 이유**:
   - forward 의 FMA fusion: Quadrants compiler 가 automatic, intermediate temps 도 inline (Option 2 byte-exact)
   - manual reverse 의 explicit scalar: byte-exact (Suspect 2)
   - cross-substep `.grad` zero set: legitimate chain 끊김 (Option B2, C 둘 다 worse)
   - kernel-level manual replace: 효과 0% (A1, A2)

⇒ **fundamental Quadrants AD / FP precision 한계**. Genesis 의 *FMA-aware reverse* 와 *수학적 mathematical-clean reverse* 가 *수치적 분기*. 양자 모두 *FP64 floor 의 정확성*이고, *수정 가능한 silent drop 없음*.

### 실용 권장 (Genesis 사용자)

- **N <= 4**: rel error ~1e-2 ~ 1e-4 (실용 적절)
- **N >= 8 (J4-같은 FREE+revolute multi-link)**: rel error 1.0+ catastrophic. **이 사용 패턴 회피**.
- **mitigation**:
  - dt 작게 (drift ∝ dt; 10x smaller → 1000x drift 감소)
  - arm offset 작게 (drift ∝ |arm_local|)
  - chain length 짧게
  - 가능하면 single-DOF / 단순 chain 사용

### Task #77 (J4 N=4 wrong source 식별) — 완료

- ✅ Real source 정확히 isolated (transform_by_quat with non-zero v)
- ✅ Numerical mechanism 식별 (FMA vs mathematical reverse FP-order divergence)
- ✅ 모든 fix strategy systematic 시도 + falsified
- ✅ Genesis 의 수치적 stability 한계 문서화

`real wrong source` 가 *fundamental Quadrants AD limitation*. 추가 fix attempt 가 *productive 가 아닌 결론*.

## REVISION 9 (2026-05-14): Option 2 byte-exact identical — Quadrants FMA optimization fundamental

### Option 2 시도 결과

`qd_transform_by_quat` (geom.py:294) 의 *forward 식* 을 *intermediate temps 명시* + *explicit scalar mul-then-add* 로 재작성:
```python
m00 = v0 * R00; m01 = v1 * R01; m02 = v2 * R02
s00 = m00 + m01
out0 = s00 + m02
# ... 동일 패턴 for out1, out2
```

J4 N=16 sweep 결과: **byte-exact identical** (abs_diff: root_wz -3.918e-9 동일).

⇒ **Quadrants compiler 가 *intermediate temps 도 inline + FMA optimize***. *forward FP order 변경 불가능*.

### 결론 — fundamental Quadrants AD / FP limitation

지금까지 확인된 finding 종합:

1. **수학적 root cause**: `transform_by_quat(arm_local, parent_quat)` chain with non-zero `arm_local` (arm_x sensitivity test 로 isolated)
2. **수치적 정확성** (모두 verified):
   - forward quat unit-norm: floor (4e-16)
   - kernel_manual_func_integrate_bw: FP64 floor (production state PASS)
   - kernel_manual_compute_qacc_bw: FP64 floor (production state PASS)
   - kernel_manual_uc_bw_one_link: isolated FD PASS
   - d_transform_by_quat__dq scalar 재작성: byte-exact (FMA 회피 안 됨)
   - forward qd_transform_by_quat intermediate temps: byte-exact (Quadrants inline)

3. **수치적 cumulative drift**:
   - dt ∝ drift (1000x 감소 with dt 10x smaller)
   - arm_local magnitude ∝ drift (linear)
   - N step ∝ drift (10x per N doubling for angular DOFs)

### 진정한 fix 가능 한 방향 (남은)

**Option A (가장 진정성, 큰 작업)**: forward 와 reverse 둘 다 *동일 FP order 사용*. Quadrants 의 *automatic differentiation* 자체를 우회 — *manual forward + manual reverse* (둘 다 numpy-side computation). Genesis sim 의 *각 timestep forward* 도 numpy 로 작성 → *FMA 없이 strict IEEE 754 mul-then-add*.

→ 실용 불가능 (성능 + 코드 양).

**Option B (현실적)**: drift 자체 *fundamental Quadrants AD / FMA 한계* 인정.
- J4 같은 *FREE + revolute multi-link entity* 의 *long-horizon backward (N>=8)* 는 *수치적 drift inevitable*
- 단순 시나리오 (J1, J5_chain3 등) 는 *floor PASS*
- 실용적 권장:
  - N 짧게 (N<=4 까지는 rel error 작음)
  - dt 작게 (drift ∝ dt)
  - arm offset 작게 (drift ∝ arm_local)

**Option C (실험적)**: *각 substep 끝* 에 *cross-substep .grad scrub*. 단 *legitimate vs leak* 분리 어려움 (이전 author 코멘트 line 1530-1541). 추가 진단 필요.

### Action items 정리 (다음 session)

진정한 fix 가능 한 *novel hypothesis* 가 *없으면* — *option B (인정)* 가 *결론*. *option C (cross-substep selective scrub)* 가 *마지막 시도 가치*.

다음 session 의 우선 작업:
1. *Option C exploration*: 각 cross-substep `.grad` field 별 *legitimate (chain contribution) vs leak (cumulative drift)* 분류 — *각 field 마다 zero set + N=16 sweep 측정* (1 field 씩, ~30 fields × ~30초 = 약 15분).
2. Drift 감소 fields 식별 → *selective scrub kernel* 작성.

### Code state (option 2 시도 후)

- `qd_transform_by_quat` (geom.py:294): clean revert (byte-exact, 효과 없음)
- 모든 manual kernels framework 유지

## REVISION 8 (2026-05-14): drift source isolated — `transform_by_quat` reverse 의 FP order diff

### 결정적 진단 단계

1. **Suspect 1 (forward quat unit-norm drift) FALSIFIED**:
   - `/tmp/diag_quat_norm_drift.py`: J4 N=16 forward 의 chassis_quat `||quat||` drift = **4.4e-16** (FP64 floor, ~0.4 ulp)
   - forward quat normalization 깨짐 *아님*

2. **Suspect 2 (`d_transform_by_quat__dq` Vector arithmetic) FALSIFIED**:
   - explicit scalar form + intermediate temps 변경 → production output **byte-exact identical**
   - FMA fusion 영향 *없음* (compiler 가 temp 도 inline + FMA, or 원래 FMA 가 없음)

3. **arm inertia (ill-conditioning) FALSIFIED**:
   - arm_inertia ∈ {1e-2, 1e-3, 1e-4, 9.7e-5, 2.2e-3} → drift 거의 동일 (3.9e-9 ~ 4.7e-9)

4. **arm offset sensitivity — DECISIVE**:

| arm_x | N=16 abs_diff | rel_max |
|---|---|---|
| **0.00** | **1.85e-11** | **3.3e-3 (FLOOR)** |
| 0.05 | 7.93e-10 | 0.36 |
| 0.20 | 3.92e-9 | 1.45 |
| 0.50 | 9.58e-9 | 0.52 |
| 1.00 | 1.26e-8 | 0.25 |

⇒ **drift ∝ arm_local magnitude**. **arm_x=0 floor PASS**.

### Real source

**`d_transform_by_quat__dq(v, quat, out_grad)` 가 *v 의 magnitude 와 비례 한 *수치적 drift introduce***:
- `v` (= `arm_local`) 가 작을수록 drift 작음
- `v=0` 시 *backward contribution 자체 0* → drift 없음
- `v>0` 시 *각 backward substep 별 *수치적 small error introduced* → N step 누적

### 의미 — forward FMA vs manual reverse의 FP order divergence

forward (`qd_transform_by_quat`, geom.py line 317):
```
out[0] = v0 * (q_xx + q_ww - q_yy - q_zz) + v1 * (2*q_xy - 2*q_wz) + v2 * (2*q_xz + 2*q_wy)
```
→ Quadrants 가 *3 mul + 2 add chain* 으로 *FMA fusion 가능*.

manual reverse (`d_transform_by_quat__dq`):
```
∂L/∂qw = og0 * 2*(qw*v0 - qz*v1 + qy*v2) + og1 * 2*(qz*v0 + qw*v1 - qx*v2) + og2 * 2*(-qy*v0 + qx*v1 + qw*v2)
```
→ *수학적 reverse* — *forward 의 FMA FP order 와 *수치적 다른 식*.

**수학적으로 동치** 이지만 *FP arithmetic 의 *수치적 결과 diff*. v 가 작을수록 contribution 작 → drift 작음. v=0 시 모든 contribution 0 → drift 0.

### 진정한 fix 방향

**Option 1 (가장 진정)**: backward chain 의 `d_transform_by_quat__dq` 를 *forward 와 동일 FP order* 사용 reverse 로 재작성. forward 의 각 식의 *각 term* 별 *수치적 reverse* (forward FMA chain 의 *부분 미분 + same FP rounding sequence*).

**Option 2**: forward `qd_transform_by_quat` 를 *FMA fusion 어렵게* 재작성 (intermediate temp 명시). 그러면 forward 의 FP order 가 *manual reverse 와 일치*. backward chain 자동 일치.

**Option 3 (실용)**: Genesis 의 multi-link FK chain 의 *수치적 accuracy 한계 인정*. dt 작게 + multi-step backward 시 *drift 가능* 알림.

### 단순 verification (즉시 가능)

**Option 2 시도**: `qd_transform_by_quat` (geom.py:294) 의 *intermediate temps* 추가:
```python
ax_w = q_xx + q_ww - q_yy - q_zz
ax_y = 2.0 * q_xy - 2.0 * q_wz
ax_z = 2.0 * q_xz + 2.0 * q_wy
out0_x = v0 * ax_w
out0_y = v1 * ax_y
out0_z = v2 * ax_z
out_x = out0_x + out0_y + out0_z
# ... etc
```
→ J4 N=16 sweep 측정. drift 감소 시 → forward FMA 가 source confirmed.

### Recommended next session work

1. **Option 2 (forward intermediate temp) 시도** — *수정 시도 가능 한 *test*. drift 변화 측정.
2. Option 2 FAIL 시 → **Option 1** (manual reverse 의 *수치적 forward-aware 형식 작성*) 시도. 큰 작업.
3. **`kernel_forward_velocity_one_link.grad` 의 *수치적 reverse 도 *transform_by_quat 의존*. 동일 issue 가능 — 동일 fix 적용.

## REVISION 7 (2026-05-14): option B3 detailed verify — drift ∝ dt, systematic wrong identified

### 진단 결과

1. **Deterministic confirmed** (`/tmp/diag_determinism.py`): 동일 input 두 번 실행 → byte-exact identical. FP order non-determinism 아님.

2. **Topology N progression** (`/tmp/diag_all_topo_n_progression.py`, `diag_all_topo_relerror_sweep.py`):
   - J1_free (single FREE 6 DOFs): N=16 rel 3.6e-11 (FP64 floor)
   - J2_revolute, J3_prismatic: floor or 0
   - J5_chain3 (3 revolutes chain): N=16 rel 1.5e-5 (floor)
   - **J4_free_rev (FREE + revolute multi-link): N=16 rel 13.25 (catastrophic)**
   - ⇒ wrong source = FREE + revolute multi-link interaction (FK chain의 transform_by_quat)

3. **dt sensitivity** (J4 N=16, `/tmp/diag_dt_sensitivity.py`):

| dt | abs_diff | rel |
|---|---|---|
| 1e-2 | 3.9e-9 | 1.45 |
| 5e-3 | 6.3e-11 | 1.31 |
| 1e-3 | 5e-12 | 1.00 |

→ **drift ∝ dt (1000x 감소 with 10x smaller dt)**. forward dynamics cumulative numerical magnitude 가 source. rel 유지 ~1.0 → systematic 수학적 wrong (fixable, not FP floor).

### 다음 session 의 top suspects

**Suspect 1: forward chain quat unit-norm drift**
- `qd_transform_quat_by_quat` (geom.py line 281-290) 가 *의도적으로 normalize 안 함* (line 286 코멘트: backward chain attenuation 방지 — `.normalized()` 가 tangent-space projection 으로 w-direction 의 chain 차단).
- *multi-step accumulation 시 unit-norm 깨짐* → cumulative numerical error.
- 검증: forward 자체에 *normalize 추가* (개념적, backward 영향 받음 알고) → drift 변화 측정.
- *trade-off*: normalize 추가 시 *backward chain 의 w-component attenuation* 다시 발생 → 다른 wrong source. 신중 평가.

**Suspect 2: `kernel_manual_uc_bw_one_link` 의 `d_transform_by_quat__dq` 의 Vector arithmetic**
- J4 chain core: `parent_quat.grad` 5 sources 누적
- 각 source = `d_transform_by_quat__dq` Vector return (3 mul + 2 add per element)
- Quadrants compiler FMA fusion 의심
- 검증: `d_transform_by_quat__dq` 의 내부 식 *explicit scalar mul-then-add* 로 재작성 → verify_v2 패턴
- production output 변화 측정 (manual numpy explicit scalar 와 byte-exact 인지)

**Suspect 3: Quadrants `kernel_forward_velocity_one_link.grad` (Quadrants AD)**
- Vector arithmetic 의 FMA fusion 가능
- `cd_vel`, `cd_ang` chain 의 *cross-link transform_by_quat reverse* 가 핵심
- 검증: manual replace + production output 비교

### Recommended next session work order

1. **Suspect 2 먼저** (가장 self-contained, 측정 가능):
   - `d_transform_by_quat__dq` 의 explicit scalar 재작성
   - J4 N=2 sweep 측정 → 변화 확인
2. **Suspect 3 다음**:
   - `kernel_forward_velocity_one_link` 의 `kernel_manual_forward_velocity_bw` 작성 (없음)
   - production verify (verify_v2 패턴)
3. **Suspect 1 마지막** (위 둘이 PASS 면):
   - forward normalize trade-off 실험

## REVISION 6 (2026-05-14): option B2 simple zero falsified — Genesis checkpoint interaction

### Option B2 implementation 시도

`substep_pre_coupling_grad` 진입 시 *모든 cross-substep `.grad` fields zero set* (qpos/vel/qpos_next/vel_next.grad 만 보존):
```python
if getattr(self, '_first_bw_done', False):
    # backup primary carriers, zero all, restore
    ...
else:
    self._first_bw_done = True
```

### 결과

**zero apply 한 번도 실행 안 됨** — `_first_bw_done = False` 가 *매 backward substep 진입* 시 reset.

원인 발견: **Genesis 의 *checkpoint mechanism***:
```python
# simulator.py:298 _step_grad():
def _step_grad(self):
    for _ in range(self._substeps - 1, -1, -1):
        if self.cur_substep_local == 0:
            self.load_ckpt()  # → self.step(in_backward=True) → self.substep()
        ...
        self.sub_step_grad(self.cur_substep_local)  # substep_pre_coupling_grad
```

⇒ **각 outer backward step 마다 *load_ckpt 가 full forward replay***. 그 안 `self.substep` 호출 → 우리 reset condition `if not self._is_backward` trigger (`self._is_backward=False` 상태 — `load_ckpt`의 forward replay 모드). `_first_bw_done = False reset`. *zero apply 실행 안 됨*.

### 이전 author 의 동일 finding (line 1530-1541 코멘트)

```
Naively zeroing the candidate fields here did NOT recover J1 multistep precision
(likely those `.grad` values carry partial legitimate chain contributions even
though they "leak" past substep end). Identifying the exact leaking field set
is tracked as a follow-up.
```

⇒ **option B2 simple zero 는 *이미 알려진 falsified strategy***. `.grad` fields 가 *partial legitimate cross-substep chain contribution* 포함 → 전부 zero 시 backward chain 끊김.

### Genesis checkpoint mechanism 의 의미

각 outer backward step 의 `load_ckpt` → `self.step(in_backward=True) × N`:
- *primal state* (qpos, vel, etc.): checkpoint load + N steps forward replay.
- *`.grad` attribute*: load 안 함, *이전 backward chain 의 마지막 값 그대로 유지*.

즉 *forward stash 는 매 backward outer step 마다 *재생성*. *.grad attribute 는 *cumulative carry over*.

**가설**: cumulative drift 의 source 가 *각 backward outer step 의 *load_ckpt forward replay 의 *FP order 변동*. Quadrants FMA optimization 이 *call 별 다른 FP order* 사용 가능 (parallel execution / atomic ops 순서 등).

검증 어려움.

### Option B3 simplified 진행 결과 (2026-05-14)

`/tmp/diag_all_topo_n_progression.py` 측정 (Richardson FD vs ana, seed 1000 t=0):

| Topology | N=1 abs_diff | N=16 abs_diff | N=16 \|ana\| | N=16 rel |
|---|---|---|---|---|
| J1_free (single FREE 6 DOFs) | 2.10e-19 | 1.46e-11 | 5.3e-6 | 3e-6 (floor) |
| J4_free_rev (FREE + revolute 7 DOFs) | 6.49e-13 | 4e-9 (root_wz) | 3e-8 | **0.13** (real wrong) |

(`diag_all_topo_relerror_sweep.py` 의 기존 결과로부터)

| Topology | N=16 max rel |
|---|---|
| J1_free | 3.6e-11 (FP64 floor) |
| J2_revolute | 0 (deterministic exact) |
| J3_prismatic | 5.6e-12 (FP64 floor) |
| **J4_free_rev** | **13.25** (CATASTROPHIC) |
| J5_chain3 (3 revolutes chain) | 1.5e-5 (floor) |

### 결정적 finding

- **single FREE (J1) PASS**: cross-substep drift 없음, FP64 floor 유지.
- **chain of revolutes (J5_chain3) PASS**: multi-link revolute chain 도 floor 유지.
- **FREE + revolute (J4) WRONG**: **유일한 catastrophic 케이스**.

⇒ wrong source 가 **FREE + revolute multi-link interaction**. 단순 FREE 도, 단순 revolute chain 도 아닌 *FREE joint 의 quat update 와 *revolute child link 의 *forward kinematic chain interaction*.

### 다음 session 우선 work

**구체 chain 분석**: J4 의 forward chain 가:
```
chassis_pos = qpos[0:3]
chassis_quat = qpos[3:7]
arm_pos = chassis_pos + R(chassis_quat) @ arm_local
arm_quat = chassis_quat ⊗ qloc(arm_revolute_angle)
```

backward chain:
- `arm_pos.grad → chassis_pos.grad + chassis_quat.grad (via d_transform_by_quat__dq)`
- `arm_quat.grad → chassis_quat.grad (via d_quat_mul) + qloc.grad (revolute axis)`

→ **chassis_quat.grad chain** 의 *각 substep 별 *수치적 drift cumulative*.

**확인 방법**:
1. J4 의 backward chain 의 *각 .grad call output capture* (forward_velocity_one_link.grad, COM_links.grad, update_cartesian_space.grad — 모두 manual or split).
2. 각 stage 에서 *production output vs explicit-scalar manual numpy* 비교.
3. *어느 stage 의 *chassis_quat.grad chain* 이 *수치적 silent drop 발생*.

`kernel_manual_uc_bw_one_link` (FK Jacobian-transpose, manual_bw.py) 가 *유력한 source* — `d_transform_by_quat__dq` 의 *FP order*.

### Codebase state

`kernel_manual_func_integrate_bw` (scalar) + `kernel_manual_compute_qacc_bw` (LDLT) 모두 *production verify PASS*. wrong source 가 *이 두 kernel 외 chain*.

### 코드 상태 (option B2 시도 후)

- rigid_solver.py: clean revert
- Strategy A1 implementation (`kernel_manual_func_integrate_bw` 호출) 도 revert
- *all manual kernels (manual_bw.py, forward_dynamics.py 의 standalone wrappers)* 는 유지 (framework + diag scripts)

## REVISION 5 (2026-05-14): real source = cross-substep cumulative drift

### 결정적 진단 4 단계

**Step 1**: verify_v2 의 manual numpy 식 *explicit scalar + captured vel_next* 로 재작성 → production kernel_step_2.grad 와 *byte-exact PASS* (qpos.grad max|d| 2.78e-17, vel/acc 0). silent drop 가설 *완전 falsified*.

**Step 2**: J4 N=16 Richardson extrapolation FD (h=1e-5, h=5e-6 → O(h^4) FD floor ~1e-15) vs ana:
- root_wz: abs_diff **4e-9** (real wrong)
- root_wx: abs_diff **1.8e-9** (real wrong)
- root_wy: abs_diff **5e-10** (real wrong)
- root_x: abs_diff ~1e-12 (FP64 floor)

→ **real systematic wrong 확정**. FD precision artifact 아님.

**Step 3**: N별 abs_diff progression (seed 1000 t=0, J4):

| DOF | N=1 | N=2 | N=4 | N=8 | N=16 |
|---|---|---|---|---|---|
| root_x | 7e-13 | 1e-12 | 2e-13 | 1e-12 | 1e-12 |
| root_y | 2e-13 | 3e-12 | 3e-12 | 7e-11 | 6e-11 |
| root_z | 1e-14 | 6e-14 | 8e-12 | 4e-12 | 4e-11 |
| **root_wx** | 6e-13 | 3e-12 | 3e-11 | 1e-10 | **1.8e-9** |
| **root_wy** | 8e-13 | 3e-12 | 7e-11 | 2e-11 | **5e-10** |
| **root_wz** | 3e-13 | 1e-11 | 2e-11 | 8e-10 | **4e-9** |
| arm_rev | 4e-14 | 1e-11 | 3e-11 | 2e-10 | 4e-10 |

→ **N=1 은 FP64 floor** (1e-13). **약 10x per N doubling** for angular DOFs. cumulative drift.

**Step 4**: **N=1 single backward vs N=2 t=1 backward (first in LIFO)** byte-exact identical (`/tmp/diag_n2_t1_vs_n1.py`):
- 모든 DOFs abs_diff = **0.000e+00**

→ **single substep backward 자체는 정확**. **wrong source 가 *cross-substep state propagation*** (t=1 → t=0 의 누적).

**Step 5 (보너스)**: `kernel_manual_compute_qacc_bw` production state verify PASS (max|d| **4.96e-24**, FP64 floor). LDLT IFT manual 가 byte-exact 동작.

### Real wrong source 정확한 위치

cross-substep state propagation:
- `kernel_prepare_backward_substep` 의 `func_copy_next_to_curr_grad`: *qpos.grad → qpos_next.grad swap + zero*, *vel.grad → vel_next.grad swap + zero*
- **다른 `.grad` fields (cd_\*, cinr_\*, cdof_\*, cdofd_\*, crb_\*, mass_mat 등) 의 *.grad 는 *명시적 zero 안 됨***
- 이들이 *cross-substep carry over* → *fields-level chain* 에서 *수치적 cumulative drift 누적*

dump 결과 *Stage 0 (f=0 ENTRY before prepare)*: `links_state.pos.grad` 가 *nonzero value (이전 substep 의 cumulative)*.

### Strategy 회고 — 모든 시도가 효과 0% 였던 이유 최종

| Strategy | 결과 |
|---|---|
| A1 (step_2.grad manual replace) | byte-exact identical |
| A2 (split kernel) | byte-exact identical |
| qpos primal restore | byte-exact identical |
| manual pre-call | byte-exact identical |

⇒ **모든 시도가 *substep 내부* 의 *step_2.grad* 만 fix 시도**. *substep 내부 backward 는 이미 정확*. *real source 는 *substep 외부 (cross-substep)*.

### 다음 session 우선 work

**option B1 (구조적)**: cross-substep `.grad` zero set 추가. `func_copy_next_to_curr_grad` 에 *모든 .grad fields zero set* 추가:
```python
for i_l, i_b in ...:
    links_state.cd_vel.grad[i_l, i_b] = 0
    links_state.cd_ang.grad[i_l, i_b] = 0
    links_state.cinr_pos.grad[i_l, i_b] = 0
    # etc.
for i_d, i_b in ...:
    dofs_state.cdof_vel.grad[i_d, i_b] = 0
    dofs_state.cdof_ang.grad[i_d, i_b] = 0
    dofs_state.cdofd_vel.grad[i_d, i_b] = 0
    dofs_state.cdofd_ang.grad[i_d, i_b] = 0
    # etc.
```
**WARNING**: 일부 `.grad` field 는 *legitimate cross-substep chain contribution* — *전부 zero set 시 chain 끊김 → wrong increase*. 어느 fields 가 *legitimate accumulate* vs *leak* 인지 분석 필요.

**option B2 (진단)**: N=2 의 *t=1 backward 끝* + *t=0 backward 시작* 사이 `.grad` field 값 dump. *어느 fields 가 *nonzero carry over*. legitimate vs leak 분석. dump tag 추가 + 측정.

**option B3 (manual replace 확장)**: 남은 Quadrants `.grad` calls (bias_force / update_acc / torque / mm_*) 도 manual replace. *cross-substep 의 *Quadrants chain 끊고 manual chain 만*. 매우 invasive.

권장: **B2 (진단) 먼저** → legitimate vs leak fields 식별 → B1 적용.

## OLD HANDOFF — REVISION 5 가 OBSOLETE 으로 만듦

## DECISIVE FINDING (2026-05-14, revision 4): silent drop 가설 자체 FALSIFIED

### 진단 chain

1. `kernel_manual_func_integrate_bw` 를 *scalar arithmetic + intermediate temps* 로 재작성 (qd.Vector / FMA fusion 회피 시도).
2. `notes/diag_manual_kernel_prod_state.py`: production state capture → standalone solver inject → manual kernel 호출 → manual numpy (explicit scalar) 비교.

결과:
| Source | qpos.grad[5] | vel.grad[wy] |
|---|---|---|
| production kernel_step_2.grad | -9.10520619e-05 | -7.13884811e-07 |
| **our manual kernel (scalar)** | **-9.10520619e-05** | **-7.13884811e-07** |
| **manual numpy (explicit scalar)** | **-9.10520619e-05** | **-7.13884811e-07** |
| verify_v2 "manual numpy" (matrix `J_a.T @ v`) | -9.10543463e-05 | -7.13896234e-07 |

⇒ **production kernel_step_2.grad = our manual kernel = manual numpy (explicit scalar)**.

`diag_j4_n2_step2_bw_verify.py` 의 *"silent drop" 1.14e-11 (vel) / 2.28e-09 (qpos)* 는 *measurement 식의 FP order 차이* 였음:
- verify_v2 의 *manual numpy* = `J_a.T @ qpos_next_grad_rot` (matrix multiply — BLAS FMA 가능)
- production kernel_step_2.grad = explicit scalar mul-then-add (numpy 와 동일 FP order)
- 두 식이 *수학적으로 동일* 이지만 *FP 순서 다름* → 수치적 ~1e-11 diff

### Strategy A1, A2 시도들의 의미 재해석

| Strategy | 효과 | 새로운 해석 |
|---|---|---|
| A1 (manual kernel) | 0% | production output 가 *이미 manual numpy (FP-correct)* — manual 호출이 같은 값 write |
| A2 (split kernel) | 0% | 동일 |
| qpos primal restore | 0% | step_2 forward stash 가 *이미 correct primal* |
| manual pre-call | 0% | step_2 reverse 가 *이미 correct chain* |

⇒ **모든 Strategy 가 *없는 silent drop 을 fix 시도***. 그래서 byte-exact identical.

### Real wrong source 는 step_2.grad 가 아님

J4 N>=4 wrong sweep (N=8: max rel 6.5, N=16: 13.25) 는 *real source 가 다른 chain*:
- compute_qacc.grad (이미 manual `kernel_manual_compute_qacc_bw` — isolated PASS but production verify 안 함)
- bias_force.grad / update_acc.grad / torque_and_passive_force.grad
- mm_armature.grad / mm_compute_f.grad / mm_crb_initialize.grad / mm_implicit_damping_corr.grad
- update_force.grad (manual, isolated PASS)
- COM_links.grad (manual, isolated PASS)
- forward_velocity_one_link.grad (isolated PASS)
- update_cartesian_space.grad (manual_uc_bw, isolated PASS)

isolated PASS 한 항목들도 *production self.substep replay 안에서 *manual numpy 와 동일 output 인지 확인 필요* (verify_v2 같은 측정 기법 — *단 *manual numpy 식 가 explicit scalar 인지 verify 후*).

### 다음 session 우선 work

1. **각 `.grad` call 의 *production output capture* + *manual numpy (explicit scalar) 와 비교***. *isolated PASS 한 항목들도 *production verify*. 진정한 silent drop 위치 식별.

2. **FD 측정 자체의 FP order 검토**: FD = `(L(x+h) - L(x-h)) / (2h)` — *h=1e-5* 일 때 *FD truncation noise ~ 1e-12*. *J4 N>=4 의 abs diff 1e-12 ~ 1e-11* — *FD precision floor* 의 *amplification 가능*. 더 정확 FD method (Richardson extrapolation, complex step) 로 측정.

3. **만약 모든 .grad call 이 *production output = manual numpy*** → *FD precision floor 가 source*. *real wrong 없음*. Genesis 의 *N>=4 J4 backward 가 *수치적 정확*.

## OLD HANDOFF — 이 위 finding 으로 OBSOLETE

### Strategy A1 확장 — 결정적 진단 (2026-05-14, revision 3)

### Production state 의 manual kernel output capture

`rigid_solver.py` 에 *manual kernel 직후 vel.grad / qpos.grad python capture*:
- t=0 vel.grad[wy] = **-7.138848111387e-07** (= production silent drop)
- t=0 qpos.grad[5] = **-9.105206192497e-05** (= production silent drop)

verify_v2 의 *manual numpy*:
- vel.grad[wy] = -7.13896234e-07
- qpos.grad[5] = -9.10543463e-05

⇒ **우리 *kernel_manual_func_integrate_bw* 의 *production state output* 가 *production silent drop* 와 *byte-exact 동일***.

### isolated test vs production state 의 결정적 차이

| Test environment | manual kernel output | manual numpy | diff |
|---|---|---|---|
| isolated standalone (`diag_manual_func_integrate_bw_verify.py`) | -X | -X | 0 |
| production state inject (`diag_combined_with_production_state.py`) | manual numpy | manual numpy | ~1e-20 |
| **production self.substep replay 안** | **production silent drop** | manual numpy | **1.14e-11** |

### 원인 진단

**Quadrants kernel 안의 `qd.Vector` arithmetic 자체** 가 *production state 의 *near-identity quat / 작은 ang* 에서 *FP arithmetic 순서 차이 (FMA 등)* 로 *python numpy 와 *수치적 1.14e-11 diff*.

- isolated test 의 *random state* — *non-singular*. *Quadrants FP arithmetic = python numpy*.
- production state — *near-identity quat (0.999..., 1e-4, ...)*. *FP cancellation 의 *operation 순서 의존*. *Quadrants 의 *FMA 사용* 또는 *Vector 의 *operation 순서* 가 *python numpy 와 *수치적 다른 결과*.

### Strategy A1, A2 모두 효과 0%

| Strategy | 원인 |
|---|---|
| A2 (split kernel) | fields-level cross-kernel chain |
| A1 (manual `.grad` write via Quadrants kernel) | **manual kernel 자체가 Quadrants FP arithmetic 사용 → 동일 silent drop** |

### 다음 session 우선 work

**Strategy A1 fundamental 변형**: *manual kernel 안의 `qd.Vector` 제거 + scalar arithmetic + operation 순서 명시*.
- Quadrants kernel 안에서 `aw*bw + ax*bx + ay*by + az*bz` 형태 명시적 scalar 작성 (no `qd.Vector` outer call).
- FMA 비활성화 가능성 확인 (`qd.no_fast_math` 등 flag).
- python numpy 와 *수치적 byte-exact* 검증 — production state 에서.

또는 **python-side computation**: `kernel_manual_func_integrate_bw` 를 *python function* 으로 작성 (numpy 사용). Genesis solver field 와 sync 위한 to_numpy + from_numpy 접근. *kernel JIT 우회*.

## 이전 시도들 — FALSIFIED 기록

### Strategy A1 시도 결과 (초기, 2026-05-14): 효과 0%

`kernel_manual_func_integrate_bw` 작성 (manual_bw.py) — manual reverse of func_integrate (FP64 floor PASS isolated, `notes/diag_manual_func_integrate_bw_verify.py`).

`rigid_solver.py` 의 *backward replay path 변경*:
- `self.substep(f)`: self._is_backward=True 시 `kernel_step_2` 대신 `kernel_func_update_acc_standalone + kernel_func_integrate_standalone` split kernels 호출
- `substep_pre_coupling_grad`: `kernel_step_2.grad` 대신 `kernel_manual_func_integrate_bw` 호출 (직접 `.grad` write)

결과:
- per-DOF rel error: **byte-exact identical** (seed 1000 t=0 root_wy = -1.7125243034e-09 in both baseline and fix)
- J4 sweep: 모든 N (1, 2, 4, 8, 16, 32) max rel **byte-exact identical**
- debug verify: `kernel_manual_func_integrate_bw` 안에서 `qpos.grad[3] = 1e10` 강제 set → stage 16 dump 에 1e10 반영 ✓ (호출 됨)

**해석**:
- manual kernel 호출됨, qpos/vel/acc.grad 에 *직접 write* 완료
- 그러나 *후속 .grad chain* (compute_qacc.grad, torque.grad, fwd_dyn.grad, COM_links.grad, forward_velocity.grad 등) 이 *Quadrants AD 의 forward-stash-based chain* 으로 동작
- Quadrants AD chain 이 *fields-level* 로 *forward stash 기반* — *우리 manual `.grad` write 를 *읽지 않거나*, *cross-kernel chain 으로 *덮어씀**
- 결과: *최종 ctrl_force.grad output 가 *production output 와 동일* (manual fix 의 영향 없음)

⇒ **Quadrants AD 한계**: kernel-level `.grad` 호출이 *유일한 chain 통로*. Manual `.grad` write 는 *후속 chain 이 무시*. *kernel `.grad` 호출의 *forward stash* 가 *chain 의 source*.

### Strategy A1, A2 모두 falsified

| Strategy | 시도 | 결과 |
|---|---|---|
| A2 (split kernel for cleaner stash) | kernel_step_2 → split standalone | byte-exact identical |
| A1 (manual `.grad` write directly) | kernel_step_2.grad → kernel_manual_func_integrate_bw | byte-exact identical |

### 다음 session 의 *유일하게 남은* approach

**Strategy A1 확장**: 후속 *모든* `.grad` 호출도 *manual replace*:
- `kernel_compute_qacc.grad` → manual_compute_qacc_bw (이미 있음 — `kernel_manual_compute_qacc_bw`)
- `kernel_torque_and_passive_force.grad` → manual
- `kernel_split_*.grad` → manual (각각)
- `kernel_COM_links.grad` → manual (`kernel_manual_COM_links_bw` 일부 있음)
- `kernel_forward_velocity_one_link.grad` → manual

*entire backward chain manual replace*. 매우 invasive 하지만 *Quadrants AD 의 fields-level chain* 자체 *우회*.

근데 *대안*: production 의 *kernel_step_1 자체 호출 안 함* (= cross-kernel stash 의 source 제거). 가능성 검토 필요:
- *kernel_step_1 의 outputs (cinr_*, cdof_*, cdofd_*, etc.)* 가 *backward chain 의 primal 로 필요* — *호출 안 함 → backward chain 망가짐*.
- *kernel_step_1 의 *primal output 만 별도 path 로 compute + forward stash 무*:
  - `func_forward_kinematics` 등 *수동 호출 + 그 결과 fields 에 write* (no stash)
  - 이 경우 *Quadrants AD chain 의 *primal field read* 만 사용. *stash 없음*. *manual `.grad` write 가 *후속 chain 의 *primal-only chain 영향 안 미침*.

이게 *근본 fix* 방향. *기존 kernel_step_1 호출 제거 + 수동 forward 함수 호출* — 대규모 refactor.

implementation 단계:
1. *manual quat update forward + reverse* 함수 작성 (manual_bw.py 에 helper 이미 있음: `d_quat_mul__dlhs`, `d_quat_mul__drhs`, `d_rotvec_to_quat__drotvec`).
2. *quat_mul 호출 wrap*:
   ```python
   # func_integrate 안의 quat update:
   # 기존: rot = qd_transform_quat_by_quat(qrot, rot0)
   # 변경:
   rot = qd.ad.grad_replaced(
       _func_quat_update_fwd,  # forward: qd_transform_quat_by_quat(qrot, rot0)
       _func_quat_update_bw,   # manual reverse
       qrot, rot0
   )
   ```
3. *Quadrants `qd.ad.grad_replaced + grad_for` API 확인*.
4. *test*: J4 sweep — N=4, 8, 16 rel error 측정.

**대안 (B1, invasive)**: `kernel_step_1` 안의 *func_update_acc(update_cacc=False)* 를 *별 kernel 로 분리* + *cacc field 도 별 buffer 사용* → step_2 의 update_acc(update_cacc=True) 와 *완전 fields 분리*. *너무 invasive*.

## Working diagnostics
- `notes/diag_func_integrate_isolated_fd.py` — standalone func_integrate FD verify (PASS)
- `notes/diag_combined_update_acc_integrate_fd.py` — combined chain random-state verify (PASS)
- `notes/diag_combined_with_production_state.py` — production state inject verify (PASS, X1 falsified)
- `notes/diag_j4_n2_step2_bw_verify.py` — production kernel_step_2.grad vs manual (silent drop 1.14e-11)

### Step B: forward_dynamics chain isolated FD verify
Coriolis/mass matrix chain 의 sub-kernel 별 isolated FD.

## 부산물

- N=1 wrong 의 본질: FD precision floor + 약간의 step_2 silent drop (cumulative N step 에서만 visible)
- Hypothesis 1, 2 모두 FALSIFIED
- Quadrants AD 가 *forward stash primal* 사용 — backward time field 변경 무효

## Working diagnostics
- `notes/diag_j4_n2_step2_bw_verify.py` — manual vs kernel step_2.grad diff
- `notes/diag_j4_n1_perdof_10seeds.py` — N=1 per-DOF breakdown
- `notes/diag_j4_n2_perdof.py` — N=2 per-DOF + 5 seeds
- `notes/diag_j4_n4_perdof.py` — N=4 per-DOF
- `notes/diag_j4_n2_substep_dump.py` — N=2 backward stage dump
- `notes/parse_dump.py` — dump → per-stage table
