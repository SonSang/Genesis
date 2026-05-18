# Diffrigid Handoff — J4 N=2 step_2.grad Silent Drop Hunt

날짜: 2026-05-13
브랜치: `20260512_diff_rigid_demo`
가장 최근 push 된 커밋: `80bef6ad` ([MISC] J1 multistep rel-error sweep)

---

## TL;DR

J4 multistep (free joint + revolute) 의 *catastrophic* rel-error 의 root cause 가
**`kernel_step_2.grad` 의 rotation chain 에서 silent drop** 임을 확인.

구체적으로 `quat_mul(q_pre, qrot)` 의 reverse 가 J4 N=2 의 t=0 substep
context 에서 *contribution 의 일부만* 흘려보냄 → cancellation 짝 누락 →
큰 spurious value → catastrophic rel error.

Minimal repro v1~v7 모두 *standalone 으로는 silent drop 재현 안 됨*. 
trigger 는 `kernel_step_2` 의 더 specific 한 *context* 에서 발생.

---

## Full topology sweep 결과 (현재 코드 기준)

`notes/diag_all_topo_relerror_sweep.txt`. 10 seeds × N∈{1,2,4,8,16,32} J4 만 발췌:

| Topology | N=1 | N=2 | N=4 | N=8 | N=16 | N=32 |
|---|---|---|---|---|---|---|
| J1_free | 4.3e-11 | 1.2e-3 | 6.7e-3 | 2.6e-2 | 5.3e-2 | 7.8e-2 |
| J2_revolute | 0 | 0 | 0 | 0 | 0 | 0 |
| J3_prismatic | 2.6e-12 | 2.8e-12 | 5.7e-12 | 8.3e-12 | 5.6e-12 | 1.1e-11 |
| **J4_free_rev** | **7.2e-3** | **1.2e+1** | **2.9e+2** | **5.6e+3** | **1.2e+4** | 2.9e+3 |
| J5_chain3 | 2.0e-2 | 3.0e-3 | 2.1e-3 | 4.0e-4 | 2.3e-4 | 1.8e-3 |

J4 의 N=1 의 7.2e-3 은 mask atol=1e-10 의 *FP64 floor 가짜 알람*  
(실제 diff=1e-12 수준, FD가 1e-10 이라 rel inflated). J4 N≥2 부터는 real bug.

J5 가 N 따라 *감소* — multi-revolute 라서 chain 이 averaging 되는 듯.

`notes/diag_j4_n2_perdof.py` 의 결과: J4 N=2 t=0 의 per-DOF rel error:
- root_x: 1e-7~1e-9 (FP64 floor) ✓
- root_y, root_z: 7%~136%
- **root_wx: 6630 ~ inf** (worst)
- root_wy, root_wz: 3%~150%
- arm_revolute: 7%~1240%

root_x 만 정확. **chassis 의 quat 변화 가 chain 에 들어가는 모든 DOF 가 부정확**.

---

## Chain 추적 (root_wx, seed=1000)

`notes/diag_j4_n2_substep_dump_parsed_wx.txt` (GENESIS_DEBUG_GRAD=2).

**Stage 별 root_wx 관련 grad** (J4 freejoint qpos[3..6] = qw,qx,qy,qz, vel[3..5] = wx,wy,wz):

| Stage | qpos.grad[3..6] | vel.grad[3..5] | force.grad[3] |
|---|---|---|---|
| t=0 post-UCS.grad (= post-FK chain) | 1.60e-1, 4.29e-5, -8.61e-5, -8.86e-6 | 2.6e-12, -6.1e-7, -6.5e-8 | - |
| t=0 begin_backward_substep | (cleared, qpos_next.grad 로 swap) | (cleared) | - |
| **t=0 step_2.grad** | 1.60e-1, **-4.31e-5**, -9.59e-5, 3.25e-5 | **-4.30e-7**, -1.11e-6, 2.02e-7 | - |
| t=0 compute_qacc.grad (manual) | (same) | (same) | **-3.91e-8** (= ana[t=0][3]) |

핵심 발견:
- step_2.grad 의 *quat_mul reverse* 가 `q_pre.x.grad` 변화 (+4.29e-5 → -4.31e-5)
- 그게 `vel.grad[3]` (wx) 에 -4.30e-7 chain
- `compute_qacc.grad` 가 force.grad[3] = M⁻¹·acc.grad[3] 으로 chain → -3.91e-8
- 다음 fwd_dyn.grad 에서 ctrl_force.grad[3] = ana[t=0][3] = -3.91e-8 = catastrophic

---

## Manual verification 결과

### `kernel_manual_compute_qacc_bw` — ✅ 무죄

`notes/diag_j4_n2_qacc_bw_verify.py`:
```
force.grad (kernel) - force.grad_manual = M⁻¹ · acc.grad:
  max|diff| = 6.78e-21   (FP64 floor)
  max rel   = 6.97e-16   (FP64 epsilon)
```

manual 과 kernel 의 force.grad 가 FP64 정밀도로 완벽 일치. 따라서 chain
의 *upstream* (acc.grad) 가 bug source.

### `step_2.grad` rotation reverse — ❌ Silent drop

`notes/diag_j4_n2_step2_bw_verify.py`:

q_pre.x.grad 의 4 개 chain contribution 분해 (J_a^T row 1):

| Term | Value | 운명 |
|---|---|---|
| `-qrot.x · q_next.w.grad` | **-4.299e-5** | ✓ kernel 유지 |
| `qrot.w · q_next.x.grad` | **+4.292e-5** | ✗ **dropped** |
| `-qrot.z · q_next.y.grad` | -1.12e-8 | ✗ dropped (small) |
| `qrot.y · q_next.z.grad` | -2.71e-10 | ✗ dropped (small) |

Manual sum = -7.99e-8 (큰 두 항이 cancel)  
Kernel = -4.31e-5 ≈ 첫 항만

**중요**: forward 식이 `q_next = quat_mul(q_pre, qrot)` 인데 (`qd_transform_quat_by_quat(qrot, rot0) == quat_mul(rot0, qrot)` 이므로),
- J_a 는 *qrot 의 함수만* — `q_pre` forward primal 값과 무관
- 따라서 *forward primal 부정확이 원인 아님*. 명확한 silent drop.

---

## Minimal repro 시도 (모두 재현 실패)

`notes/quadrants_repros/case_quat_mul_*.py`:

| Variant | 추가 component | 결과 |
|---|---|---|
| **v1** | basic: qd.Vector + quat_mul + per-element write | OK (FP64 floor) |
| **v2** | + outer ndrange + static index | OK |
| **v3** | + cross-ndrange (vel_next intermediate) (`@qd.func` split 필요) | OK |
| **v4** | + real rotvec_to_quat (cos/sin/sqrt) | OK |
| **v5** | + batch dim + dynamic q_start (array-loaded) | OK |
| **v6** | + `if joint_type == FREE` conditional | OK |
| **v7** | + translation `pos += vel*dt` + rot_offset variable + vel_next shape (6,1) | OK |

모든 variant 에서 `q_pre.grad` 가 numpy chain rule (J_a^T @ q_next.grad) 과
FP64 floor 수준에서 일치. 즉 *isolated 환경에서 silent drop 재현 안 됨*.

**남은 후보 (v8+ 또는 Genesis source binary search 대상)**:
1. `kernel_step_2` 의 `func_update_acc` + `func_implicit_damping` 추가 호출
2. Genesis 의 `LinksInfo` / `JointsInfo` dataclass-like struct indirection
   (`I_l = [i_l, i_b] if static(batch_links_info) else i_l`)
3. `func_check_index_range` + `for i_1 in qd.static(range(1))` 중첩 layer
4. `if links_info.n_dofs[I_l] > 0` outer if
5. `prepare_backward_substep` 후의 *forward primal restore + self.substep replay* 의 specific 상태

---

## 다음 세션 진행 옵션

### Option A: Genesis source binary search (Recommended)

`kernel_step_2` 의 body 의 일부 component 를 *toggle* 하면서 J4 N=2 결과 변화 확인:
- `func_update_acc` 호출 skip → silent drop 사라지는지
- `func_implicit_damping` 호출 skip → 사라지는지
- `func_integrate` 의 translation branch 만 skip → 사라지는지

각 변경 후 `python notes/diag_j4_n2_perdof.py` 로 비교.

이게 *real context* 에서 가장 효율적인 좁힘.

### Option B: Manual `kernel_step_2_bw` 작성

`kernel_manual_compute_qacc_bw` 와 같은 패턴. step_2.grad 의 chain rule 식이
이미 manual 으로 검증됨 (`diag_j4_n2_step2_bw_verify.py` 의 manual reverse). 그
식을 그대로 kernel 으로 작성:

```
# Translation
qpos.grad[0..2] += qpos_next.grad[0..2]
vel_next.grad[0..2] += dt · qpos_next.grad[0..2]
# Revolute (DOF 6)
qpos.grad[7] += qpos_next.grad[7]
vel_next.grad[6] += dt · qpos_next.grad[7]
# Rotation
J_a = quat_mul_jac_a(q_pre, qrot)
J_b = quat_mul_jac_b(q_pre, qrot)
q_pre_grad = J_a^T @ qpos_next.grad[3..6]
qrot_grad = J_b^T @ qpos_next.grad[3..6]
J_rot = rotvec_to_quat_jac(ang, eps)
ang_grad = J_rot^T @ qrot_grad
vel_next.grad[3..5] += dt · ang_grad
qpos.grad[3..6] += q_pre_grad
# Integrator step (vel_next = vel + dt*acc)
vel.grad += vel_next.grad
acc.grad += dt · vel_next.grad
```

forward primal 값 (q_pre, qrot, ang) 은 self.substep(f) 직후의 forward state 사용.

이게 J4 catastrophic fix 의 직접 경로. trigger 정확한 지점은 *별도 task* 또는 *Quadrants 팀 보고* 로 분리.

### Option C: Quadrants 팀에 보고용 minimal repro 만들기

위 trigger 좁히기를 v8+ 으로 계속 → 결국 *최소 trigger* 의 minimal repro 확보 → Quadrants 팀에 보고.

---

## 검증 인프라 (notes/)

| 파일 | 역할 |
|---|---|
| `diag_all_topo_relerror_sweep.py` / `.txt` | J1~J5 × N sweep, 10 seeds. Scene cache 로 segfault 회피 |
| `diag_j4_n1_perdof_10seeds.py` | J4 N=1 의 per-DOF rel error — N=1 자체는 FP64 floor 확인 |
| `diag_j4_n2_perdof.py` | J4 N=2 의 t=1 vs t=0 per-DOF — t=1 OK, t=0 catastrophic |
| `diag_j4_n2_substep_dump.py` + `parse_j4_n2_dump.py` | J4 N=2 의 backward chain stage 별 dump (GENESIS_DEBUG_GRAD=2) |
| `parse_j4_n2_wx_chain.py` + `diag_j4_n2_substep_dump_parsed_wx.txt` | root_wx 의 chain stage 별 추출 |
| **`diag_j4_n2_qacc_bw_verify.py`** | compute_qacc_bw 검증 — FP64 floor ✓ |
| **`diag_j4_n2_step2_bw_verify.py`** | step_2.grad rotation 검증 — silent drop 확인 + 4 contributions 분해 |
| `quadrants_repros/case_quat_mul_v1~v7*.py` | Minimal repro 시리즈 — 모두 OK |

---

## 코드 상태 (uncommitted)

`git status --short` 의 modifications:
- `genesis/engine/solvers/rigid/rigid_solver.py`: 확장된 `_debug_grad_dump` field 목록 + ENTRY tag
- 위 diagnostic scripts (modified 또는 새로 생성)

## 사용자 명시 가이드

- `kernel_begin_backward_substep` 가 `qpos.grad → qpos_next.grad`, `vel.grad → vel_next.grad` swap 한다는 점 명심
- step_2.grad 의 forward primal 에 대한 의문 (이전처럼 forward primal 부정확 가능성 있는지) — 분석상 *J_a 가 qrot 함수만* 이라 q_pre forward 값 무관. 단 qrot 값 (vel_next forward primal) 의존성 있음
- minimal repro 의 component 분리 방향 (cross-ndrange 면 @qd.func split 필요 — direct kernel body 에 두 ndrange 는 reverse_segments@68 reject)
- 사용자는 *trigger 지점 정확히 좁히기* 선호. 단순 fix 보다 root cause 추적.
- 응답은 한국어 + 코드 주석/문자열은 영어

---

## 즉시 시작 가능한 다음 단계

```bash
# 1. Genesis source binary search 시작 — kernel_step_2 body 의 일부 toggle
#    (func_update_acc 호출 주석 처리 후 J4 N=2 결과 확인)

python notes/diag_j4_n2_perdof.py     # baseline (catastrophic)
# 그 후 rigid_solver.py 또는 forward_dynamics.py 에서 specific func 제거
python notes/diag_j4_n2_perdof.py     # 차이 비교
```

또는 v8+ 작성:
```bash
# 2. v8: add func_update_acc-like extra ndrange
#    notes/quadrants_repros/case_quat_mul_v8_with_update_acc.py 작성
```
