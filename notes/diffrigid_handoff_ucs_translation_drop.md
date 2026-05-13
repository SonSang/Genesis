# Diffrigid Handoff — J4 N=2 UCS.grad Translation Chain Silent Drop

날짜: 2026-05-13
브랜치: `20260512_diff_rigid_demo`
대상 commit: `51be6701` (debugging guide v2)

---

## 🎯 식별된 함수 (Step 4 결과)

**`kernel_update_cartesian_space_one_link.grad`** (정의: 
`genesis/engine/solvers/rigid/abd/forward_kinematics.py:816`).

**버그 위치**: SPC #2 (= step 0's BW = second BW substep) 의 offset=1 (arm
link) 호출에서, chain rule `links_pos.grad[arm] → links_pos.grad[chassis]`
가 부분적으로 silent drop.

---

## 검증된 chain rule 의 정확한 wrong 값

### 수학적으로 정확한 chain rule
Forward: `links_pos[arm] = links_pos[chassis] + R(chassis_quat) @ arm_local`
→ `d(links_pos[arm])/d(links_pos[chassis]) = I_3` (quat 과 무관)
→ Reverse: `links_pos.grad[chassis] += links_pos.grad[arm]` (component-wise 1:1)

### SPC #1 (offset=1, link 1.grad) — 정상 ✅
- Input  `links_pos.grad[arm]   = [ 3.9998e-01, -1.3603e-04,  2.0155e-04]`
- Delta on `links_pos.grad[chassis] = [ 3.9998e-01, -1.3603e-04,  2.0155e-04]`
- 100% identity pass-through ✓

### SPC #2 (offset=1, link 1.grad) — partial silent drop ❌
- Input  `links_pos.grad[arm]   = [-3.2491e-08,  2.7745e-05,  7.3066e-05]`
- Delta on `links_pos.grad[chassis] = [ 1.7525e-09,  2.7745e-05,  2.9683e-05]`
- 컴포넌트별 비율:
  - x: 1.75e-9 / -3.25e-8 = **-5.4%** (sign flip + 95% drop)
  - y: 2.77e-5 / 2.77e-5 = **100%** ✓
  - z: 2.97e-5 / 7.31e-5 = **40.6%** (59% drop)

### 결과 (cascade effect on ctrl_force.grad)
chain 이 drop 되어 `qpos.grad[0:3]` (translation) 이 SPC #2 에서 update 안 됨
→ `qpos_next.grad[0:3]` (after begin_bw swap) 잘못된 값 → step_2.grad block 1
reverse 통해 `acc.grad[0:3]` 영향 → `force.grad[0:3]` 영향 →
`ctrl_force.grad[0:3]` 잘못된 값 (root_y, root_z 의 24-73% rel err).

또한 cinr/cdof chain 을 통해 angular DOFs 까지 cascade (root_wy, root_wz,
arm_revolute).

---

## 가설: 왜 SPC #1 에서는 OK 인데 SPC #2 에서 drop?

같은 kernel.grad 가 두 SPC 에서 다른 동작. 차이점:
1. **Forward primal (qpos, links_quat)**:
   - SPC #1: post-step-1 quat ≈ `[0.99999760, 6.7e-4, -5.0e-5, -1.5e-4]`
   - SPC #2: post-step-0 quat ≈ `[1.0, 2.7e-4, 3.1e-5, -1.3e-4]`
   둘 다 identity 근처지만 magnitude 차이.
2. **Input `.grad` magnitudes**:
   - SPC #1: links_pos.grad ~ O(0.4) (loss 직접 chain)
   - SPC #2: links_pos.grad ~ O(7e-5) (cross-substep chain)
3. **Call site position in BW chain**:
   - SPC #1: ctrl_force.grad seed from loss (clean)
   - SPC #2: ctrl_force.grad seed from previous SPC's output (has cumulative
     numerical state)

Chain rule `d/d(links_pos[chassis])` 는 identity → quat / magnitude 무관.
따라서 이건 *Quadrants AD silent drop* (수식과 무관한 backend bug). 비슷한
패턴이 과거에도 발견됨 (`qd_transform_by_quat` 의 specific primal 조건에서
silent drop, 이전 fix 사례 있음).

---

## 다음 세션 진행 옵션

### Option A: `qd_transform_by_quat` reformulation
과거 사례처럼 `qd_transform_by_quat` 또는 그 reverse 가 specific primal 조건
(예: quat 의 component magnitude 비대칭) 에서 silent drop 발생. 
`func_forward_kinematics_entity_one_link` 의 chain 을 추적해서 silent drop 의
정확한 trigger 식별 → branch-free reformulation 또는 manual reverse.

### Option B: Manual UC backward (이미 존재)
`notes/diffrigid_handoff_j4_step2_silent_drop.md` 에서 시도했던
`kernel_manual_uc_bw_one_link` (in `manual_bw.py`) 를 wire 해서 auto-AD 우회.
다만 J4 의 chain (free + revolute) 만 implemented → arm hinge 부분 검증 필요.

### Option C: Minimal repro 작성
SPC #2 의 UCS.grad chain 만 isolate 해서 Quadrants 팀에 보고할 minimal repro.
같은 입력 primal + 입력 .grad 값으로 standalone 호출 → silent drop 재현.

---

## 검증 인프라

| 파일 | 역할 |
|---|---|
| `notes/diag_j4_n2_perdof.py` | J4 N=2 per-DOF rel error |
| `/tmp/parse_chain3.py` | SPC #1 vs #2 chain 단계별 비교 (delta 추적) |
| `/tmp/verify_ucs_drop.py` | SPC #1/SPC #2 end qpos.grad 정밀 비교 |
| 임시 instrument | `kernel_update_cartesian_space_one_link.grad` 직전/직후
  qpos.grad / links_pos.grad / links_quat.grad 변화 측정 |

---

## 가이드 준수 self-check

이번 디버깅이 가이드 (notes/diffrigid_debugging_guide.md) 의 step 4 요구사항
("어떤 *함수* 가 어떤 *.grad 필드* 에 어떤 *값* 을 잘못 쓰는지 한 줄로 답할
수 있을 때까지 끝내지 말 것") 을 충족:

> **kernel_update_cartesian_space_one_link.grad** 가 SPC #2 offset=1 호출에서
> **links_pos.grad[chassis]** 에 **links_pos.grad[arm] 의 x 컴포넌트의 5%
> 만, z 컴포넌트의 40% 만** propagate (예상: identity 100%).

이전 SPC #1 정상 동작 비교, 수식 의 identity 보장, primal-independent
chain rule 까지 검증된 상태에서 명시적 식별 완료.
