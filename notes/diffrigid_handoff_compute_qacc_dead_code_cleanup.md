# Diffrigid Handoff — `func_compute_qacc` backward dead-code cleanup

날짜: 2026-05-24
대상 commit: cleanup commit on `20260512_diff_rigid_demo` branch
이전 manual backward 도입: `kernel_manual_compute_qacc_bw` (`func_compute_qacc` 의 IFT-based reverse)

---

## TL;DR

`kernel_manual_compute_qacc_bw` 가 `func_compute_qacc` 의 backward 를 IFT 로 완전히 대체한 이후, **이전 path 가 의존하던 `acc_smooth_bw` value/grad 보호 코드가 더 이상 필요 없다는 사실을 검증하고 제거**.

- `substep_pre_coupling_grad` 진입 직전의 `kernel_zero_acc_smooth_bw(self.dofs_state)` + `qd_zero_grad(self.dofs_state.acc_smooth_bw)` 두 줄 제거.
- `rigid_solver.py` 의 회고성 주석 (Quadrants-traced `kernel_compute_qacc.grad`, Phase B externals, BW-mode Step 1 skip 등) 제거 — 모두 dead code path 의 잔재.
- `kernel_manual_compute_qacc_bw` docstring 에 **Inputs vs Outputs** 섹션 추가: factored mass (`mass_mat_L`, `mass_mat_D_inv`) 를 read 하고 dense `mass_mat.grad` 를 write 한다는 비대칭성, 그리고 이 kernel 이 backward path 에서 `mass_mat.grad` 를 populate 하는 유일한 지점임을 명시.

테스트: `test_diff_forward_kinematics.py` — 23 passed / 2 fail (`J4_free_rev`, `J5_chain3` multistep_control_force, pre-existing). cleanup 전후 PASS/FAIL 패턴 동일.

---

## 제거 근거 (검증 3 포인트)

이전 `kernel_zero_acc_smooth_bw` + `qd_zero_grad` 가 막던 leak 가설:

| # | Leak 가설 | 검증 결과 |
|---|---|---|
| 1 | value: 이전 substep 의 per-DOF Step 1 writes 가 이번 substep 의 `func_solve_mass_entity` BW 모드 Step 2 로 leak | `func_solve_mass_entity` body 에서 **`is_backward` / BW 분기 자체가 제거됨**. `vec`/`out` 만 사용, `acc_smooth_bw` 를 직접 read 하지 않음. 시나리오 발생 불가능. |
| 2 | `.grad`: `kernel_compute_qacc.grad` + `kernel_solve_mass_step2_reverse_bw` 가 `acc_smooth_bw.grad` 에 atomic_add 누적 → multi-step over-counting | `grep "kernel_compute_qacc.grad\|kernel_solve_mass.*_bw"` → **실제 호출 site 0개, 모두 docstring/주석 reference**. `kernel_solve_mass_step1_one_dof_bw` / `step2_reverse_bw` 는 *정의도 없음*. 자동 trace 경로 자체가 dead. |
| 3 | forward replay 가 `acc_smooth_bw` 의 stale 값을 input 으로 read | `kernel_manual_compute_qacc_bw` 내부에서 모든 read 가 same-launch self-write 에 dominate 됨. seed line 이 `[0]` 슬롯을 즉시 덮어쓰고, Step 1/2/3 의 `[1]` 슬롯 cross-iter read 는 같은 kernel 내 이전 iteration 에서 쓰여진 값만 읽음. 첫 iteration 의 inner range 는 항상 빈 range. |

### `kernel_manual_compute_qacc_bw` 안의 self-overwrite-before-read 패턴

```python
# Seed: [0] 슬롯 explicit 덮어쓰기
for i_d in range(dof_start, dof_end):
    acc_smooth_bw[0, i_d, i_b] = acc_smooth.grad[i_d, i_b] + acc.grad[i_d, i_b]

# Step 1 (descending i_d): [1, j_d] read (j_d > i_d)
# 첫 iteration (i_d = dof_end-1): inner range = range(dof_end, dof_end) = empty → no stale read
# 이후 iteration: [1, j_d] read 는 이전 iteration 의 write 에서 옴

# Step 2: [1, i_d] read (Step 1 write 결과) → [0, i_d] 덮어쓰기

# Step 3 (ascending i_d): [1, j_d] read (j_d < i_d)
# 첫 iteration (i_d = dof_start): inner range = range(dof_start, dof_start) = empty
# 이후 iteration: 같은 Step 3 의 이전 iteration write 에서 옴
```

→ kernel launch 이전 buffer 상태와 결과 무관.

---

## 제거한 코드

`genesis/engine/solvers/rigid/rigid_solver.py`:

```diff
-        # Clear `acc_smooth_bw` value AND its `.grad` before this backward
-        # substep's work. Two leak channels:
-        # ... (30 줄 주석 + leak channel 가설 설명)
-        if self._requires_grad:
-            kernel_zero_acc_smooth_bw(self.dofs_state)
-            qd_zero_grad(self.dofs_state.acc_smooth_bw)
         self.substep(f)
```

그리고 `kernel_manual_compute_qacc_bw` 호출 위/아래의 회고 주석 (옛 Quadrants-traced 경로, Phase B externals, J4/J5 corruption 회상 등) 도 3 줄로 축약.

`genesis/engine/solvers/rigid/abd/forward_dynamics.py` (`kernel_manual_compute_qacc_bw` docstring):

```diff
-    Replaces the Quadrants-traced reverse path (... 옛 path 설명)
-    Scratch: reuses `dofs_state.acc_smooth_bw[0/1]` since its forward
-    intermediate values are dead by the time this kernel runs.
+    Inputs vs outputs (note the asymmetry...):
+        Reads:  acc.grad, acc_smooth.grad, mass_mat_L, mass_mat_D_inv, acc_smooth
+        Writes: force.grad, mass_mat.grad, (acc.grad, acc_smooth.grad → 0)
+    The dense `mass_mat` is *not* read here ... but its `.grad` is the
+    parameter the IFT chain naturally exposes, so this kernel is the only
+    place `mass_mat.grad` gets populated in the backward path.
```

---

## 남긴 코드 (의도적)

`RigidSolver.reset_grad()` 안의 `kernel_zero_acc_smooth_bw(self.dofs_state)` (line ~1319) **는 유지**.

- 위치: horizon 간 reset (SHAC-style training 의 `loss.backward()` → `scene.reset(snapshot)` → 다음 horizon `loss.backward()` 경로).
- 이론적으로는 위 검증 결과에 따라 이 site 도 dead path. 그러나 `reset_grad` 는 *cross-horizon* 시점이라 *value 잔재가 다른 의도치 않은 경로로 read 될 가능성* 을 완전히 배제하려면 별도 multi-horizon stress 가 필요. follow-up.

---

## 검증 절차 (재현용)

### `acc_smooth_bw` 의 모든 reference

```bash
grep -rn "acc_smooth_bw" --include="*.py" genesis/
```

read 하는 곳:
- `genesis/engine/solvers/rigid/abd/forward_dynamics.py` 안의 `kernel_manual_compute_qacc_bw` 본체만.

write 하는 곳 (0 으로 set):
- `genesis/engine/solvers/rigid/abd/accessor.py:811` 의 `kernel_zero_acc_smooth_bw`.

### auto-traced `.grad` consumer

```bash
grep -rn "kernel_compute_qacc\.grad\|kernel_solve_mass_step.*_bw\|kernel_solve_mass.*\.grad" \
    --include="*.py" genesis/
```

→ 호출 site 0개. `kernel_solve_mass_step1_one_dof_bw` / `step2_reverse_bw` 는 정의도 없음.

### `func_solve_mass_entity` body 의 BW 분기

```bash
sed -n '572,608p' genesis/engine/solvers/rigid/abd/forward_dynamics.py
```

→ `is_backward` 인자 자체가 없음. `vec`/`out` 만 사용, `acc_smooth_bw` 미사용.

### 테스트

```bash
rm -rf ~/.cache/quadrants/qdcache
CUDA_VISIBLE_DEVICES="" conda run -n genesis python -m pytest \
    tests/test_diff_forward_kinematics.py -v -n 0
```

기대 결과: 23 passed / 2 fail (`test_diff_fk_multistep_control_force[J4_free_rev-cpu]`, `[J5_chain3-cpu]`). cleanup 전후 동일.

---

## Follow-ups

1. `reset_grad` 의 `kernel_zero_acc_smooth_bw` 호출도 제거 가능 여부 — multi-horizon (`scene.reset(snapshot)` 후 다음 `loss.backward()`) 시나리오에서 leak 0 확인 후 정리.
2. `kernel_zero_acc_smooth_bw` 정의 자체 (`accessor.py:811-839`) 와 그 import — 위 follow-up 끝나면 함께 제거 후보.
3. `kernel_compute_qacc` (line 1184 의 wrapper kernel) — forward 에서도 호출 site 없음 (`func_compute_qacc` 만 사용). dead wrapper, 별도 cleanup 후보.
4. `kernel_solve_mass_step1_one_dof_bw` / `step2_reverse_bw` 의 docstring/주석 reference — 코드는 이미 없지만 주석에서 자주 언급됨. 정리 시 일관된 정정 필요.
