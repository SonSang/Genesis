# Diffrigid Handoff — Backward primal consistency (forward replay reorder + cache load)

날짜: 2026-05-24
브랜치: `20260512_diff_rigid_demo`

---

## TL;DR

`substep_pre_coupling_grad` 의 forward replay path 가 *real forward (BW=False) 의 primal* 과
일관된 chain 으로 채워지도록 세 가지를 동시에 정리:

1. **Forward replay 순서를 dependency-respecting 으로** (FK → COM → vel forward, 그 뒤 reverse 는 vel.bw → COM.grad → FK.bw)
2. **`kernel_copy_next_to_curr_no_check` 에 vel copy 추가** (qpos 만 post-integrate 였던 mixed state → vel 도 post-integrate)
3. **`kernel_begin_backward_substep` 의 `func_copy_cartesian_space` 호출을 cache → state load 방향으로** (인자 swap)

세 변경이 **단독으로는 regression** 인데 **셋이 모두 함께 적용되면 baseline 대비 regression 0** (`test_diff_forward_kinematics.py` 23 passed / 2 expected fail). 시점 별 primal 의도가 일관되게 정리되어야 backward chain 이 정확.

J4/J5 `multistep_control_force` 는 이번 변경과 무관하게 여전히 fail (별개 mechanism, follow-up).

---

## 배경 — 두 시점의 forward primal

`substep[f]` 의 forward (BW=False) 안에 **두 개의 forward chain** 이 존재:

| Forward chain | 사용 primal | 호출 위치 |
|---|---|---|
| `func_forward_dynamics` (kernel_step_1) | substep[f] 시작 = substep[f-1] 끝의 *post-integrate* state | `kernel_step_1` 의 `func_update_cartesian_space + func_forward_velocity` 가 guard 로 skip 되고, 이전 substep 의 BW=False 분기가 남긴 값 사용 |
| `func_update_cartesian_space + func_forward_velocity` (kernel_step_2 BW=False 분기 끝) | *substep[f] 의 post-integrate qpos + vel* | `func_copy_next_to_curr` 직후 |

즉 substep[f] 안에 **`forward_dynamics` 시점 primal** 과 **`post-integrate UCS/vel forward` 시점 primal** 두 가지가 *서로 다른 시점의 state*. backward 는 두 chain 의 reverse 를 *각자 정확한 primal* 로 실행해야 함.

---

## 발견된 inconsistency

`substep_pre_coupling_grad` 의 흐름 (이전 코드):

```
prepare_backward_substep
  → load_adjoint_cache (pre-integrate qpos/vel/acc 를 current 로)
  → func_update_cartesian_space (BW=True)            ← pre-integrate primal 채움
  → func_copy_cartesian_space  (current → cache)     ← cache 에 pre-integrate primal 저장
self.substep(f)   BW=True
  → kernel_step_1: forward_dynamics 만 실행 (UCS/vel forward 는 guard 로 skip)
  → kernel_step_2: integrate → vel_next/qpos_next, BW=False 분기 전체 skip
kernel_copy_next_to_curr_no_check
  → qpos_next → qpos (post-integrate)
  → vel 은 ❌ 안 옮김 (load_adjoint_cache 의 pre-integrate 값 유지)
kernel_forward_velocity (BW=True)  ← cdof_*, xanchor 등 pre-integrate read (mixed)
kernel_manual_forward_velocity_bw
kernel_COM_links (BW=True)         ← pre-integrate xanchor/quat read
kernel_COM_links.grad
kernel_forward_kinematics_fk_only (BW=True) ← post-integrate qpos read (마지막에야)
kernel_manual_fk_only_bw
begin_backward_substep
  → func_copy_cartesian_space  (current → cache)     ❌ 또 save (load 가 의도였을 듯)
step_2.grad → forward_dynamics.grad
```

문제 세 가지가 동시에 존재했음:

1. **vel 이 안 옮겨짐** → forward replay 가 *pre-integrate vel 로 cdofvel_* 계산* (real forward 는 post-integrate vel 사용)
2. **Forward replay 순서가 dependency 역방향** → vel forward 가 *FK forward 가 채울 xanchor* 를 *이전 값* 으로 read
3. **cache 가 read 되지 않고 또 save** → `prepare` 가 저장한 pre-integrate primal 이 *post-integrate primal 로 덮어쓰여짐 + read 안 됨*

이전에 PASS 였던 이유: `*_state_adjoint_cache` 의 *read consumer 가 0개* (manual reverse 들이 모두 *current state* 를 직접 read) → cache 의 잘못된 write 방향이 *영향 없이* 살아 있었음. 그러나 *current state 의 mixed primal* 자체가 *우연히 forward 와 가까운 값* 이라 fp64 floor 안쪽 drift 로 묻혔던 것.

---

## 적용한 fix (3 변경 동시)

### (1) Forward replay 순서 (`rigid_solver.py`)

두 site (substep_pre_coupling_grad 의 post-coupling 부분 + initial-state 부분):

**기존**:
```
vel forward → vel.bw → COM forward → COM.grad → FK forward → FK.bw
```

**새 순서**:
```
FK forward → COM forward → vel forward          (dependency-respecting forward chain)
vel.bw → COM.grad → FK.bw                       (reverse 역순)
```

근거: `func_forward_kinematics_entity` 가 xanchor/xaxis/pos/quat 채우고, `func_COM_links` 가 그것 read 해서 cinr_*/cdof_*/cdofvel_* 채우고, `func_forward_velocity_entity` 가 cdof_* read 해서 cd_*/cdofd_* 채움. 새 순서가 *real forward 의 read/write 의존성을 그대로* 재현.

### (2) vel copy in `kernel_copy_next_to_curr_no_check`

`dofs_state.vel[i_d, i_b] = dofs_state.vel_next[i_d, i_b]` 추가.

- 기존 주석에 적힌 *"forward_velocity 의 correct primal 이 pre-integrate vel"* 주장은 *kernel_step_2 BW=False 분기 의 `func_copy_next_to_curr → func_forward_velocity` 가 post-integrate vel 로 실행* 한다는 사실과 inconsistent. 주석 정정.
- `func_COM_links_entity_main:532-533` 의 `cdofvel_* = cdof_* · vel` 도 *post-integrate vel* 이어야 정확.

### (3) `kernel_begin_backward_substep` 의 cache → state load (인자 swap)

`func_copy_cartesian_space` 는 단방향 `RHS_state → LHS_cache` write 함수. 인자 두 그룹을 swap 해서 호출하면 *cache → state* (load) 방향으로 동작.

```python
func_copy_cartesian_space(
    dofs_state=dofs_state_adjoint_cache,                   # RHS = cache (source)
    links_state=links_state_adjoint_cache,
    joints_state=joints_state_adjoint_cache,
    geoms_state=geoms_state_adjoint_cache,
    dofs_state_adjoint_cache=dofs_state,                   # LHS = current (target)
    links_state_adjoint_cache=links_state,
    joints_state_adjoint_cache=joints_state,
    geoms_state_adjoint_cache=geoms_state,
    static_rigid_sim_config=static_rigid_sim_config,
)
```

이로써 `step_2.grad` 와 그 후 forward_dynamics 쪽 manual reverse 들 (`update_force_bw`, `mm_assemble_bw`, etc.) 이 read 하는 *current state* 가 *forward_dynamics 가 실제 본 pre-integrate primal* 로 복원됨.

---

## 시점별 primal 정리 (수정 후)

```
prepare_backward_substep
  → current = pre-integrate primal (load_adjoint_cache + FK/COM/vel BW=True)
  → cache   = pre-integrate primal (save 완료)
self.substep(f) BW=True
  → integrate 만 실행 → vel_next, qpos_next 채움
kernel_copy_next_to_curr_no_check
  → current.{qpos, vel} = post-integrate
{FK, COM, vel} forward (BW=True)
  → current.{xanchor, ..., cd_*, cdofd_*} = post-integrate primal (chain consistent)
{vel.bw, COM.grad, FK.bw}
  → post-integrate primal 의 reverse  (kernel_step_2 BW=False 분기 의 forward 의 reverse)
begin_backward_substep (cache → state load)
  → current = pre-integrate primal (forward_dynamics 시점)
step_2.grad + forward_dynamics 쪽 manual reverse
  → pre-integrate primal 의 reverse  (kernel_step_1 의 forward_dynamics 의 reverse)
```

두 forward chain 의 reverse 가 *각자 정확한 시점의 primal* 로 작동.

---

## 검증

### Test 결과

```bash
rm -rf ~/.cache/quadrants/qdcache
CUDA_VISIBLE_DEVICES="" conda run -n genesis python -m pytest \
    tests/test_diff_forward_kinematics.py -v -n 0
```

| 케이스 군 | HEAD 이전 (baseline) | 변경 후 |
|---|---|---|
| freejoint (4 case) | ✓ | ✓ |
| revolute (4) | ✓ | ✓ |
| prismatic (4) | ✓ | ✓ |
| free_with_revolute J4 (4) | ✓ | ✓ |
| revolute_chain3 J5 (4) | ✓ | ✓ |
| multistep J1_free | ✓ | ✓ |
| multistep J2_revolute | ✓ | ✓ |
| multistep J3_prismatic | ✓ | ✓ |
| **multistep J4_free_rev** | ✗ | ✗ (변경 없음) |
| **multistep J5_chain3** | ✗ | ✗ (변경 없음) |

**23 passed / 2 expected fail** — baseline 과 동일 PASS 패턴.

### 중간 단계 결과 (왜 셋이 동시에 필요한가)

| 변경 조합 | passed/failed |
|---|---|
| HEAD (baseline) | 23 / 2 |
| reorder only | 18 / 7 (fp64 freejoint/J4/J5 batched 5개 NEW FAIL) |
| reorder + vel copy | 17 / 8 (J1_free multistep 도 NEW FAIL) |
| **reorder + vel copy + cache→state load** | **23 / 2 (baseline 복원)** |

중간 상태들이 모두 regression 인 이유: forward replay 가 post-integrate primal 을 current 에 남기면, *forward_dynamics 쪽 manual reverse 들* 이 그 mixed/post primal 을 read → wrong reverse. cache load 가 *forward_dynamics 시점 으로 current 를 되돌려서* 그 문제를 해결.

---

## 코드 위치

- **`genesis/engine/solvers/rigid/rigid_solver.py`**:
  - `substep_pre_coupling_grad` 의 post-coupling forward replay 영역 (line ~1494)
  - 동일 함수의 initial-state forward replay 영역 (line ~1750)
- **`genesis/engine/solvers/rigid/abd/diff.py`**:
  - `kernel_copy_next_to_curr_no_check` (line ~360): vel copy 추가 + 주석 정정
  - `kernel_begin_backward_substep` (line ~244): `func_copy_cartesian_space` 호출 인자 swap

`*_state_adjoint_cache` (read consumer 가 dead 였던 cache 들) 가 이번에 처음으로 *real consumer* (begin_backward_substep 의 load) 를 가짐.

---

## Follow-ups

1. **J4/J5 `multistep_control_force` leak** — 이번 fix 와 무관. 그동안 `[feedback_multistep_grad_leak]` 에 적힌 silent-AD chain loss 의 root cause 가 *primal consistency 가 아님* 이 확인됨. 별도 가설 (xanchor/quat 의 FMA-fusion divergence, cinr_pos.grad chain attenuation 등) 으로 추적 필요. 다만 이번 정리로 *primal 변수* 가 제거되어 stage diagnostic 의 noise 가 줄어들 가능성.
2. **`func_copy_cartesian_space` 의 단방향성** — 현재는 인자 swap trick 으로 load 처럼 사용. 가독성 위해 `func_load_cartesian_space` 라는 별도 함수를 추가하거나 `func_copy_cartesian_space` 에 direction enum 추가하는 게 깔끔. 별도 cleanup.
3. **`*_state_adjoint_cache` 의 field 범위** — 현재 cache 가 보관하는 field 들 (line 314-340) 이 *forward_dynamics 가 실제 read 하는 field* 와 일치하는지 검증 필요. 누락된 field 가 있으면 fp64 floor 에 미세 drift 가능. `func_forward_dynamics` body 의 read 슬롯 grep 으로 확인 follow-up.
4. **주석/handoff 정리** — `kernel_zero_acc_smooth_bw` docstring, `func_solve_mass_entity` 주석 등에 *옛 path* 가정의 회고 주석이 더 남아 있을 수 있음. 일관 cleanup 별도.
