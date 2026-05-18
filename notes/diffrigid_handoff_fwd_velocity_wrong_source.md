# Diffrigid Handoff — J4 N=2 vel.grad wrong source 발견

날짜: 2026-05-14
브랜치: `20260503_diff_rigid_demo` (현 commit `5090aa6a` 위에 미커밋 변경)

## 결론 (핵심 finding)

**Stage 14 vel.grad wrong 의 진짜 source = `kernel_forward_velocity_one_link.grad` 의 chain**.

기존 dump (`notes/j4_n2_dump_current.txt`) 분석 결과:

| Stage | 위치 | vel.grad[1] (chassis 의 vel[1]) | FD reference |
|---|---|---|---|
| 9 | 1st BW substep 끝 (post-fwd_dyn.grad) | **-1.480e-06** | -1.480e-06 ✓ |
| 10 | 2nd BW substep ENTRY | -1.480e-06 ✓ | (cross-substep 운반 정확) |
| 11 | 2nd BW substep entry | -1.480e-06 ✓ | |
| **12** | **after post-fwd_velocity.grad** | **-6.488e-07** ✗ | (wrong, 8.3e-07 차이) |
| 13 | after post-COM_links.grad | -6.488e-07 (변화 없음) | |
| 14 | after post-UCS.grad | -6.488e-07 ✗ | -1.480e-06 |

→ Stage 11 → 12 transition 에서 vel.grad 가 wrong 으로 변경. *forward_velocity_one_link.grad* 의 chain 이 wrong contribution.

## 사전 작업 정리 (지금까지 시간 들인 잘못된 가설)

### 잘못된 가설: kernel_COM_links 의 silent drop
이전 핸드오프 문서 (`diffrigid_handoff_j4_n2_xanchor_fix.md`) 에서 *kernel_COM_links.grad 의 cross-link silent drop* 가설을 따라 진행했음. 그러나:

- **Quadrants AD 의 kernel_COM_links.grad 는 정확** (BW=True forward + .grad 모드, production 과 동일).
- 이전 isolated FD test 의 wrong 은 *BW=False forward + .grad* 의 *slot 1 primal 안 채움* mismatch 때문이었음 (FD reference 와 manual reverse 의 *primal slot 가정* mismatch).
- production 의 `rigid_solver.py:1563-1587` 는 *BW=True forward replay + .grad* 사용 — 정확.

### 부산물로 얻은 정확한 코드

`manual_bw.py` 끝에 추가된 *full manual kernel_COM_links_bw* (line 1357 이후) 는 **FP64 floor 까지 정확** (BW=1 forward 가정). 검증된 helpers:
- `d_qd_quat_to_R__dquat` (normalize chain 포함)
- `d_qd_transform_inertia_by_trans_quat` (R chain 포함)
- `d_qd_transform_pos_quat_by_trans_quat`

이 코드는 *production 의 kernel_COM_links.grad 가 이미 정확하므로* wire-in 필요 없음. *추후 production 우회용* 으로 보관.

### Phase 5 split (`func_j_pos_quat_propagation_entity`, `kernel_COM_links_main`, `kernel_j_pos_quat_propagation`)

`forward_kinematics.py` 에 추가된 split functions/kernels. *kernel_COM_links 가 정확함을 확인한 이상* split 도 필요 없음. *정확성 검증된 코드*. 보관.

## 다음 단계 (실제 wrong source 추적)

### Step 1: forward_velocity_one_link.grad isolated FD verify

`kernel_forward_velocity_one_link` 의 backward 만 isolated 테스트:
- forward: kernel_forward_velocity_one_link (BW=True forward replay)
- reverse: kernel_forward_velocity_one_link.grad (BW=True)
- inputs: dofs_state.vel, dofs_state.cdof_ang, dofs_state.cdof_vel, links_state.cd_ang/vel (parent), 등
- outputs: dofs_state.cdofd_ang/vel, links_state.cd_vel/ang, links_state.cd_vel_bw/ang_bw

FD reference vs Quadrants AD 비교. 만약 wrong → kernel_forward_velocity_one_link.grad 의 silent drop 식별.

### Step 2: wrong 의 *specific input chain* 좁힘

vel 또는 cd_* 등 어느 input grad 가 wrong 인지 entry-by-entry verify.

### Step 3: production 의 forward_velocity 의 cross-link chain 점검

`func_forward_velocity_entity` line 1336-1338:
```python
if links_info.parent_idx[I_l] != -1:
    cvel_vel = W(links_state.cd_vel_bw, I_j0, links_state.cd_vel[links_info.parent_idx[I_l], i_b], BW)
    cvel_ang = W(links_state.cd_ang_bw, I_j0, links_state.cd_ang[links_info.parent_idx[I_l], i_b], BW)
```

cross-link read (parent 의 cd_vel/cd_ang) — Phase 5 의 pos[i_p]/quat[i_p] read 와 비슷한 패턴.

만약 isolated FD verify 가 wrong → manual replacement 필요. Phase 5 처럼 wire-in 방식 결정.

## 인프라

| 파일 | 용도 |
|---|---|
| `/tmp/diag_com_links_isolated_fd.py` | kernel_COM_links isolated FD verify (BW=1) |
| `/tmp/diag_phase5_isolated_fd.py` | Phase 5 isolated FD verify |
| `notes/parse_dump.py` | Stage 별 grad parse |
| `notes/j4_n2_dump_current.txt` | 22-stage dump (forward_velocity wrong 발견 source) |

## 미커밋 변경 요약

1. `forward_kinematics.py`:
   - Phase 5 split functions/kernels 추가 (line 500-700, 2153-2240)
   - 기존 func_COM_links_entity 그대로 유지
2. `manual_bw.py`:
   - kernel_manual_COM_links_phase5_bw (Phase 5 만)
   - kernel_manual_COM_links_bw (full manual, helpers 포함)
   - d_qd_quat_to_R__dquat, d_qd_transform_inertia_by_trans_quat, d_qd_transform_pos_quat_by_trans_quat helpers

이 코드는 *정확성 검증된 코드*. 보관. 그러나 *production wire-in 필요 없음* (kernel_COM_links 가 이미 정확).

## 다음 세션 시작 가이드

1. 이 문서 + `notes/j4_n2_dump_current.txt` 의 stage 12 transition 결과 확인.
2. **/tmp/diag_fwd_velocity_isolated_fd.py** (Phase 5 isolated FD verify 의 패턴 따라) — *isolated FD 검증 결과 PASS* (J4_free_rev FP64 floor).
3. **그러나 production 에서는 wrong** — 즉 *production state 에서만 wrong* (isolated state 다름).
4. 추적 방향:
   - **production state vs isolated state 차이** 식별
   - 가능성: 2nd substep forward_velocity 가 *forward replay (BW=True)* 후 *primal cd_*_bw / cd_*가 wrong 값* (= step t=0 의 forward 결과 인데 *expected와 다름*)
   - 또는 *cross-substep 운반된 cd_*.grad seed 가 wrong* (1st substep 의 chain 자체가 vel.grad 만 cancel 되고 cd_*.grad 자체 wrong)

## Stage 10-12 transition dump 분석

```
[10] f=0 ENTRY (before prepare_backward_substep, 2nd substep)
   vel       = [4.000e-03, -1.480e-06, 3.629e-06, -7.473e-11, -1.664e-07, 6.483e-08, 9.523e-08]
   cd_vel    = [-4.545e-10, -1.383e-07, 4.306e-07, -5.134e-10, -6.925e-08, 5.731e-07]
   cd_ang    = [2.831e-11, -6.116e-08, -4.518e-08, -4.298e-11, -2.671e-07, -6.379e-08]

[11] f=0 entry (2nd substep 내부)
   동일 값

[12] f=0 after post-fwd_velocity.grad
   vel       = [4.000e-03, -6.488e-07, 4.954e-06, -9.070e-11, -6.103e-07, -6.489e-08, -1.146e-07]
   cd_vel    = 0 (consume)
   cd_ang    = 0 (consume)
```

cd_*.grad seed 가 *cross-substep 운반된 1st substep 의 output*. 그 seed 가 forward_velocity.grad 의 chain 으로 *vel.grad 에 wrong contribution*.

→ wrong source 의 *진짜 origin* 은 *cross-substep 운반된 cd_*.grad seed 가 *2nd substep 의 forward_velocity.grad input* 으로 어울리지 않을 수도*. 또는 *2nd substep 의 forward_velocity primal* 이 stale.

## 도구 / 검증 인프라

| 파일 | 용도 |
|---|---|
| `/tmp/diag_com_links_isolated_fd.py` | kernel_COM_links isolated FD (BW=1, PASS) |
| `/tmp/diag_phase5_isolated_fd.py` | Phase 5 isolated FD (BW=1, PASS) |
| `/tmp/diag_fwd_velocity_isolated_fd.py` | kernel_forward_velocity_one_link isolated FD (BW=1, **PASS**) |
| `notes/parse_dump.py` | KEYS 확장됨 (cd_vel, cd_ang, cdof_*, cdofd_*, cinr_pos) |
| `notes/j4_n2_dump_current.txt` | 22-stage dump |

## 추가 finding (production state primal stale 가설)

Stage 9 (1st substep 끝) 의 primal summary:
- `dofs_state.vel: max=4.000e-03` (이전 state 의 vel)
- `links_state.cd_vel: max=5.731e-07`, `cd_ang: max=2.671e-07` (forward 의 output primal — 이전 forward replay 결과)
- `links_state.cd_vel_bw: max=0`, `cd_ang_bw: max=0` (forward replay 안 끝남)

Stage 10 (2nd substep ENTRY) 의 primal summary:
- `dofs_state.vel: max=4.000e-03` (cross-substep 운반된 동일 값)
- `links_state.cd_vel: max=5.731e-07`, `cd_ang: max=2.671e-07` (동일)
- `links_state.cd_vel_bw: max=0`, `cd_ang_bw: max=0` (아직 forward replay 안 함)

→ Stage 10 시점 dofs_state.vel = `4.000e-03`. 이건 *step t=0 의 input vel (initial vel = 0?)* 또는 *state[1] (1st substep 의 forward 후 vel)*.

만약 **state[1].vel** 이면 *2nd substep 의 forward replay 가 wrong vel 사용* → 모든 derived primal (cd_vel_bw, cinr_*, etc.) wrong → *forward_velocity.grad 가 wrong primal 로 chain* → vel.grad wrong.

### 다음 step

1. **substep_pre_coupling_grad 의 cur_substep_global == 0 (2nd substep) 진입 시 dofs_state.vel 의 expected value 확인**:
   - production: dump 의 stage 10 dofs_state.vel
   - expected: *step t=0 의 input vel = scene.reset() 직후의 initial vel (control 적용 전)*
   - 만약 mismatch → *backward path 의 prepare_backward_substep 가 vel 을 *step t=0 의 input* 으로 *복원 안 함**

2. rigid_solver.py 의 *prepare_backward_substep* 또는 *begin_backward_substep* 의 *primal swap* 확인.

3. 만약 primal stale → *forward replay 전에 *step t=0 의 input vel* 으로 복원* — *kernel_step_2 forward + reset* 또는 다른 메커니즘.

## 이번 자율 진행 summary

1. **kernel_COM_links.grad silent drop 가설** → **반증** (BW=1 mode 에서 정확)
2. **manual_COM_links_bw 작성** → 정확 (FD FP64 floor) but production 에 wire-in 불필요
3. **kernel_forward_velocity_one_link.grad isolated** → 정확 (PASS)
4. **Stage 11 → 12 transition wrong 식별** → forward_velocity.grad chain in production state
5. **primal stale 가설** → 다음 session 의 verify 시작점
