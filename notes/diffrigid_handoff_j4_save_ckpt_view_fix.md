# Diffrigid Handoff — J4 N≥2 save_ckpt View Aliasing Bug (FIXED)

날짜: 2026-05-13
브랜치: `20260512_diff_rigid_demo`

---

## TL;DR

J4 multistep 의 *catastrophic* rel-error 의 root cause:

**`RigidSolver.save_ckpt` 가 `qd_to_numpy(...)` 를 `copy=True` 없이 호출 →
zerocopy backend 에서 numpy view 를 저장 → 이후 forward substep 의
`kernel_save_adjoint_cache` 가 `_rigid_adjoint_cache` buffer 를 덮어쓰면
*기존에 저장된 ckpt 도 같이 변함* (view aliasing).**

결과 chain:
1. Forward step 0 끝: `save_ckpt("0")` 가 `_rigid_adjoint_cache.qpos`
   의 view 를 저장. 이때 cache[0] = INITIAL, cache[1] = post-step-0.
2. Forward step 1 시작: `kernel_save_adjoint_cache(f=0)` 가
   `_rigid_adjoint_cache.qpos[0]` 을 *post-step-0* 으로 덮어씀 (
   step 1 의 start state). View aliasing 으로 `self._ckpt["0"]["qpos"][0]`
   도 post-step-0 으로 변함.
3. Forward step 1 끝: `save_ckpt("1")` 호출.
4. BW for step 0: `load_ckpt("0")` 가 `ckpt["0"]["qpos"][0]` 을 읽는데,
   이미 *post-step-0* 으로 변경됨 (initial 이 아님).
5. 그러면 step 0 의 BW 전체가 wrong primal (post-step-0 의 quat 을
   pre-step-0 = identity 라고 잘못 알고) 로 backward 계산.

### Fix
`genesis/engine/solvers/rigid/rigid_solver.py:save_ckpt` — **3줄만 변경**:
```python
self._ckpt[ckpt_name]["qpos"]     = qd_to_numpy(..., copy=True)
self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(..., copy=True)
self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(..., copy=True)
```

### 결과 (J4 N=2 seed=1000 root_wx)
- baseline (broken): ana=-3.911e-08, fd=-5.898e-12 → rel **6630x**
- after fix: ana=-3.056e-11, fd=-5.898e-12 → rel **4.2x** (FD precision floor)

**factor 10⁴ 개선.**

---

## 이전 handoff 의 가설이 잘못된 이유

이전 `notes/diffrigid_handoff_j4_step2_silent_drop.md` 의 *silent drop in
`kernel_step_2.grad`* 가설은 **false positive** 였음.

### 잘못된 추론 chain
1. 이전 handoff 가 `quat_mul(rot0, qrot)` 의 reverse 에서 일부 contribution
   만 살아남는다고 (silent drop) 보고함.
2. 검증으로 numpy chain rule (full Jacobian) 결과와 kernel 출력을 비교.
3. `Manual sum = -7.99e-8` vs `Kernel = -4.31e-5` 의 차이를 *silent drop*
   증거로 결론.

### 실제 원인
- numpy chain 은 `rot0 = identity` 라고 *가정* 하고 계산.
- 그러나 실제 kernel 이 작동할 때 `qpos` (= forward primal of `rot0`) 는
  view aliasing 으로 *post-step-0 quat* 였음 (identity 가 아님).
- 같은 chain rule 식이지만 *primal 이 다르므로* 결과 다름.
- Kernel 의 chain 자체는 정확. *primal 만 잘못된 것*.

### 검증

`kernel_step_2.grad` (auto-AD) vs 새 manual `kernel_manual_step_2_integrate_bw`
의 결과 비교 (둘 다 save_ckpt fix 적용 후):

| | auto-AD only (split 없음) | manual (split + override) |
|---|---|---|
| J4 N=2 seed=1000 root_wx | -3.056e-11 | -3.056e-11 |
| J4 N=2 max rel | 10.32 | 10.32 |
| J4 N=8 max rel | 6.17 | 6.17 |
| J4 N=32 max rel | 7.97 | 7.97 |

**모든 N, 모든 topology 에서 결과 100% 동일**. silent drop 가설이 맞았다면 
manual 이 auto-AD 보다 *덜* 틀려야 했음. 동일하다는 것은 auto-AD 가
이미 *정확한 chain* 을 계산했음을 의미.

---

## 최종 변경 (committed state)

**3줄만**:
```diff
- self._ckpt[ckpt_name]["qpos"]     = qd_to_numpy(self._rigid_adjoint_cache.qpos)
- self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_vel)
- self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_acc)
+ self._ckpt[ckpt_name]["qpos"]     = qd_to_numpy(self._rigid_adjoint_cache.qpos,     copy=True)
+ self._ckpt[ckpt_name]["dofs_vel"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_vel, copy=True)
+ self._ckpt[ckpt_name]["dofs_acc"] = qd_to_numpy(self._rigid_adjoint_cache.dofs_acc, copy=True)
```

kernel_step_2 split, kernel_manual_step_2_integrate_bw, BW path 변경 등은
**불필요**. 이번 세션에서 추가했던 것들 → revert 됨.

---

## 진단 인프라 (검증 도구)

| 파일 | 역할 |
|---|---|
| `/tmp/fd_richardson.py` | Richardson FD 로 true gradient 측정 — eps 수렴 분석 |
| `/tmp/numpy_verify.py` | manual reverse 의 수식 검증 — Quadrants kernel output 과 numpy 계산 비교. 모두 FP64 floor (1e-17). |
| `/tmp/check_qpos_at_manual.py` | manual kernel 시점의 qpos primal 값 확인 — post-step-0 검출 |
| `/tmp/check_cache.py` | rigid_adjoint_cache 의 substep 별 변화 확인 — cache[0] = cache[1] 발견 |
| `/tmp/check_ckpt.py` | `RigidSolver.load_ckpt` 의 호출 순서 + ckpt 값 확인 — **view aliasing 발견의 결정적 도구** |

---

## 잔여 이슈 (Task #49)

J4 N=2 max rel ≈ 10 (root_y, root_z, root_wy, root_wz, arm_revolute). 
root_wx 만 FP64 floor. 다른 DOF 들이 10~120% rel err.

가능한 추가 root cause:
1. `links_state.pos/quat` staleness at UCS.grad time (substep BW replay 가
   copy_next_to_curr 를 skip 해서 links_state 가 pre-integrate primal 상태)
   — 시도했지만 일부 DOF 가 *악화* 되어 revert.
2. `kernel_update_cartesian_space_one_link.grad` 의 split 에서 일부 silent drop.
3. `kernel_forward_dynamics_without_qacc.grad` 의 mass matrix / bias force chain
   silent issue.
4. 다른 곳의 view aliasing 미발견.

다음 세션 방향:
- 한 DOF 씩 (예: root_y) backward chain 의 각 stage 에서 ana vs numpy-chain 비교
- View aliasing 추가 검색: `grep "qd_to_numpy\|qd_to_torch" | grep -v "copy="` 의
  결과를 BW chain 컨텍스트에서 audit
