# Diffrigid Handoff — J4 N=2 xanchor/xaxis Chain Added to manual_uc_bw

날짜: 2026-05-13
브랜치: `20260512_diff_rigid_demo`
참고 commit: `a01911e8` (마지막 commit, 그 이후 변경 미커밋)

---

## 🎯 현재 상태

### 미커밋된 변경 (모두 manual_bw.py + rigid_solver.py)

1. **manual_uc_bw_one_link 의 REVOLUTE branch (parent_idx != -1)**: `d_quat_mul__dlhs/__drhs` lhs/rhs swap fix.
   - 기존 (BUGGY): `qloc_grad = d_quat_mul__dlhs(qloc, parent_quat, ...)`
   - 수정: `parent_quat_grad_from_quat = d_quat_mul__dlhs(parent_quat, qloc, arm_quat_grad)` / `qloc_grad = d_quat_mul__drhs(parent_quat, qloc, arm_quat_grad)`
   - 이유: forward `qd_transform_quat_by_quat(qloc, parent_quat) = quat_mul(parent_quat, qloc)` (geom.py:281-290)

2. **REVOLUTE root (parent_idx == -1)**: 같은 swap fix.
3. **PRISMATIC (parent_idx != -1)**: 같은 swap fix.
4. **FIXED (parent_idx != -1)**: 같은 swap fix.

5. **REVOLUTE branch (parent_idx != -1) 에 xanchor/xaxis reverse chain 추가**:
   ```python
   # NEW: xanchor / xaxis reverse chain.
   # forward (forward_kinematics_entity_one_link line 720-742):
   #   xanchor[i_j] = R(quat_curr) · joints_info.pos[i_j] + pos_curr
   #   xaxis[i_j]   = R(quat_curr) · axis
   xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
   xaxis_grad = joints_state.xaxis.grad[i_j, i_b]
   parent_quat_grad_from_xanchor_via_quat = d_transform_by_quat__dq(joint_pos_off, parent_quat, xanchor_grad)
   parent_quat_grad_from_xaxis = d_transform_by_quat__dq(axis, parent_quat, xaxis_grad)
   parent_quat_grad_from_xanchor_via_pos = d_transform_by_quat__dq(arm_local, parent_quat, xanchor_grad)
   # parent_pos.grad += xanchor.grad (pos_curr chain)
   # parent_quat.grad += sum of all chain contributions
   # P8: zero xanchor.grad / xaxis.grad
   ```

6. **FREE branch (chassis)** 에 xanchor reverse chain 추가:
   ```python
   # xanchor[free_joint] = qpos[0:3] (forward FREE branch)
   xanchor_grad = joints_state.xanchor.grad[i_j, i_b]
   qpos.grad[q_start+j] += xanchor_grad[j]  # for j in 0..2
   # P8: zero xanchor.grad / xaxis.grad
   ```

7. **kernel_manual_uc_bw_one_link signature 에 `joints_state` 추가** + rigid_solver.py 두 호출 사이트 모두 `joints_state=self.joints_state` 추가.

### Pyright 진단 (noisy 무시 가능)
- `manual_bw.py` 의 `joint_pos_off` 가 새 변수, kernel decorator 의 `fastcache=True` 추론 lint.

---

## 검증 종합 (모두 FD 또는 numpy vs Quadrants AD)

| 항목 | 결과 |
|---|---|
| `d_transform_by_quat__dq` | ✅ random + near-identity 모두 FP64 floor |
| `d_quat_mul__dlhs / __drhs` | ✅ Option B (J4 FK chain) |
| `d_rotvec_to_quat__drotvec` | ✅ Option B |
| `d_motion_cross_force` | ✅ FP64 floor |
| `d_inertial_mul` | ✅ FP64 floor |
| `inertial_mul` standalone Quadrants vs numpy | ✅ FP64 floor |
| `qd_transform_inertia_by_trans_quat` Quadrants vs FD | ✅ FP64 floor (docstring 의 `new_trans` fix 정확) |
| `kernel_manual_compute_qacc_bw` IFT seed | ✅ numpy 식과 FP64 floor 일치 |
| Forward primal (mass_mat / L / D_inv) N=1 vs N=2 | ✅ 정확 일치 |
| FD 자체 (Richardson eps sweep) | ✅ eps=1e-3 ~ 1e-6 모두 5+ 자리 일치 |
| manual_uc_bw 의 J4 FK chain (모든 fix 후) numpy FD | ✅ FP64 floor (random non-identity quat) |
| **Stage 14 의 qpos.grad vs FD (J1, J2, J3)** | ✅ FP64 floor 모두 일치 (검증법 sound 확정) |
| **Stage 14 의 qpos.grad vs FD (J4, xanchor fix 후)** | ✅ **모두 FP64 floor (5e-10 ~ 1.3e-3)** |
| **Stage 14 의 vel.grad vs FD (J4, xanchor fix 후)** | ❌ **vel[1..6] 모두 wrong** (sign flip 포함) |

---

## 결정적 finding (현재 까지)

### qpos.grad chain 은 완전 fix됨 ✅
manual_uc_bw 의 4가지 lhs/rhs swap fix + xanchor/xaxis chain 추가로 *stage 14 의 qpos.grad 가 FD 와 FP64 floor 일치*. J4 의 quaternion-related chain 정확.

### vel.grad chain 은 여전히 wrong ❌
stage 14 의 vel.grad 의 source 는 *prev BW substep* (1st BW substep 의 stage 9) 의 cross-substep 운반된 값. *forward kinematics chain 이 vel.grad 에 contribution 없음* (FK 가 vel 의 함수 아니라).

vel.grad wrong 의 source 는 **forward dynamics chain (step_2 / fwd_dyn)** 의 어딘가 — *cross-substep state* 영향.

### J4 N=2 max rel 결과 (xanchor fix 후 mixed)
| seed | before | after | 변화 |
|---|---|---|---|
| 1000 | 6.97 | **8.59** | 악화 |
| 1001 | 1.64 | 1.21 | 개선 |
| 1002 | 1.53 | **0.64** | 큰 개선 |
| 1003 | 3.04 | 1.36 | 개선 |
| 1004 | 2.34 | **10.32** | 큰 악화 |

→ *qpos.grad 가 정확해진 만큼 vel.grad wrong 의 cascade 가 더 큰 영향* — fix 가 부분적.

---

## 다음 단계 (Compact 후 시작점)

### Step 1: stage 9 vs FD 측정 (1st BW substep 끝 = step t=1 의 input vel/qpos.grad)

stage 14 검증법과 동일 패턴이지만 *1st substep 끝 시점*. *stage 9 의 vel.grad* 가 *∂L/∂vel[after_t0]* 의 *완전한* 값 (= step t=1 의 reverse 의 마지막 output).

manual_uc_bw 의 xanchor chain 이 *2nd BW substep* 에서만 효과 (cur_substep_global=0 의 initial block 에서 추가 chain). *1st BW substep* 에서는 *FK reverse 없음* (initial block 호출 안 됨).

따라서:
- stage 9 의 qpos.grad: *partial* (1st substep 의 mid-SPC 만)
- stage 9 의 vel.grad: *complete* (forward kinematics 가 vel 함수 아니라 partial 영향 없음)

**stage 9 의 vel.grad** 가 FD 와 일치한다면 cross-substep 운반 자체는 정확. wrong source 가 2nd BW substep 의 *step_2.grad/fwd_dyn.grad chain* 에. 안 일치하면 1st BW substep chain 의 wrong.

진단 스크립트: `/tmp/diag_stage14_check_J4.py` 변형해서 capture 시점을 *1st BW substep 의 fwd_dyn.grad 직후* 로 변경. `cur_substep_global == 1` 일 때 capture.

### Step 2: stage 9 wrong 이면 prev BW substep 의 어느 chain 의 wrong 인지 추적
- prev substep 의 mid-SPC chain (fvol.grad → COM_links.grad → manual UC.grad → step_2.grad → compute_qacc.grad → fwd_dyn.grad)
- 어디서 vel.grad 가 wrong 으로 들어오는지 sub-stage dump

### Step 3: 또는 stage 9 정확이면 2nd BW substep 의 chain 추적
- mid-SPC chain (fvol.grad → COM_links.grad → manual UC.grad)
- *fvol.grad* 의 vel.grad output 정확성 (이미 Option B FD 검증 - 정확)
- 또는 *2nd BW substep 의 step_2.grad / fwd_dyn.grad* 의 chain wrong

---

## 검증 인프라

| 파일 | 용도 | 위치 |
|---|---|---|
| stage 14 검증 J1/J2/J3 sanity | `/tmp/diag_stage14_check_J1J2J3.py` |
| stage 14 검증 J4 | `/tmp/diag_stage14_check_J4.py` |
| FD 검증 helpers | `/tmp/diag_d_transform_by_quat_fd_check.py`, `/tmp/diag_motion_cross_force_fd_check.py`, `/tmp/diag_j4_uc_bw_fd_check.py` |
| qd_transform_inertia_by_trans_quat standalone | `/tmp/diag_inertia_transform_standalone.py` |
| Richardson FD | `/tmp/diag_richardson_fd.py` |
| Dump (현재 commit) | `notes/j4_n2_dump_current.txt`, `notes/j4_n2_stages_current.txt` |
| Dump parser | `notes/parse_dump.py` |

---

## Stage 매핑 (현재 dump 의 22 stages)

| Stage | 위치 |
|---|---|
| 0-9 | 1st BW substep (cur_substep_global == 1, step t=1 의 reverse) |
| 0 | ENTRY |
| 4 | post-UCS.grad (manual UC) — step t=1 의 final FK 의 reverse 결과 |
| 5 | post-begin_bw_sub (qpos.grad ↔ qpos_next.grad swap) |
| 6 | post-step_2.grad (integrate reverse → vel.grad / acc.grad / qpos.grad) |
| 8 | post-torque_and_passive_force.grad (= ctrl_force.grad set, ana[t=1]) |
| 9 | post-fwd_dyn.grad (= **1st substep 끝, cross-substep 운반 시작점**) |
| 10-21 | 2nd BW substep (cur_substep_global == 0, step t=0 의 reverse) |
| 14 | **post-UCS.grad (manual UC)** — *∂L/∂qpos[after_t0] / ∂L/∂vel[after_t0]* |
| 16 | post-step_2.grad (step t=0 의 dynamics reverse) |
| 18 | post-torque (= ana[t=0]) |
| 20-21 | initial block (step t=0 의 *initial FK reverse*) |

---

## 가이드 활용 self-check

- ✅ Step 1: Detect (J4 N=2 winner)
- ✅ Step 2: Localize (step t=0 BW catastrophic on rotational DOFs)
- ✅ Step 3: cross-substep state propagation
- ✅ Step 4: chain dump + numpy FD verify (모든 manual code + ML kernel 검증)
- ✅ Step 5: 5+ manual replacements + 5+ FD verifications
  - manual_uc_bw 의 식 오류 (lhs/rhs swap) 4곳 + xanchor/xaxis chain 누락 fix
  - qpos.grad chain 완전 fix 확정
- 🚧 Step 5 ongoing: vel.grad chain wrong source 추적
- ⏸ Step 6: 결과 분기
- ⏸ Step 7: forward primal 검사 (이미 거침, 부차 확인)

---

## 다음 세션 시작 가이드

1. 이 문서 + `notes/diffrigid_handoff_j4_n2_candidates_summary.md` + `notes/diffrigid_debugging_guide.md` 읽기.
2. 현재 commit `a01911e8` 위에 미커밋 변경 (manual_bw.py + rigid_solver.py) 가 있음. `git diff` 로 확인.
3. **Step 1 진단부터 시작**: stage 9 의 vel.grad 가 FD 와 일치하는지 확인.
4. 결과 따라 prev / current BW substep 의 어디서 vel.grad wrong 이 들어오는지 추적.
