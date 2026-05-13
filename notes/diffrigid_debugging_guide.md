# Differentiability Debugging Guide — Genesis Rigid Body

이 문서는 Genesis rigid body solver 의 differentiability bug 를 디버깅할 때
참고하는 일반 가이드라인. 과거 세션들에서 반복적으로 빠진 함정과 그 해결
패턴을 정리.

---

## Manual backward 를 쓸지 결정하는 기준

**기본 원칙: autodiff 를 최대한 사용.** Manual backward 는 코드 유지보수가
훨씬 어렵고 (수식 변경 시 forward 와 backward 양쪽 모두 수정해야 함),
auto-AD 가 자동으로 새 chain 을 추적하지 못함.

Manual backward 가 정당화되는 경우는 다음 두 가지:

### Case 1: Autodiff 에 실제 버그가 있는 경우
Quadrants AD 가 silent drop, wrong gradient 또는 reverse_segments@68 error 를
일으키는 경우.

**과거 사례:**
- `func_solve_mass_entity` 의 cross-iteration same-buffer reads → silent drop
  on `mass_mat.grad` chain (→ `kernel_manual_compute_qacc_bw` via IFT)
- `kernel_compute_mass_matrix.grad` body 가 for-loop + straight-line statements
  mix 라 `reverse_segments@68 Invalid program input for autodiff` reject
- `update_cartesian_space.grad` 의 cross-link 가 single-kernel 일 때 silent
  attenuation (→ per-link split)

이런 경우는 가능하면 *minimal repro 를 Quadrants 팀에 보고* 하고, fix 가
들어오기 전까지 임시 우회로 manual backward 사용.

### Case 2: Autodiff 가능하지만 비용이 큰 경우
Python-side 로 loop 를 빼서 효율성을 올리는 경우. Auto-AD 가 큰 kernel 의
모든 intermediate 를 저장하느라 메모리/속도 부담이 큰 경우.

**과거 사례:**
- LDLT solve 의 cross-iteration dependency 를 Python-side 로 풀어 per-DOF
  Step 1 manual reverse 로 분해 (이후 IFT 로 더 단순화됨)
- FK 의 per-link split 으로 Jacobian rank-1 update 분리

---

## 디버깅 워크플로우

### Step 1: Detect — FD sweep 으로 비정상 감지

```python
# 권장 sweep template
for topology in TOPOLOGIES:
    for N in [1, 2, 4, 8, 16, 32]:
        for seed in seeds:  # 최소 5개, 권장 10개
            ana, fd = measure(...)
            rel = rel_err_t0(ana, fd, atol=1e-10)
```

**Pass criterion:**
- `|fd| < 1e-10` (FP64 floor) 인 entry 는 rel 계산에서 mask out
  (mask 안 하면 floor 가 inflated rel 만들어서 fake alarm)
- N=1: max rel < 1e-9 expected (FP64 floor)
- 1 < N ≤ 32: max rel < 0.1 (10%) 까지는 chaotic dynamics 로 인한 FD
  divergence 가능성. 그 이상이면 real bug.
- **같은 entry 가 여러 seed 에서 consistent 하게** 큰 rel 일 때 real bug.
  한 seed 만 큰 건 numerical fluke 가능성 있음.

### Step 2: Localize — per-DOF / per-step 으로 좁히기

Sweep 의 max rel 이 어디서 오는지 분해. 어떤 DOF? 어떤 t (substep)?
어떤 seed?

```python
# per-DOF dump template (notes/diag_j4_n2_perdof.py 참고)
for seed in seeds:
    ana, fd = measure(mjcf, n_dofs, N, seed)
    for t in range(N):
        for d in range(n_dofs):
            rel = abs(ana[t,d] - fd[t,d]) / max(abs(fd[t,d]), atol)
            if rel > threshold: flag(seed, t, d)
```

### Step 3: N=1 vs N≥2 비대칭 — 정보 수집 단계 (skip 금지)

이건 **정보 수집** 단계이지 *분기점이 아님*. 결과는 step 4 의 *dump 범위*
를 좁히는 데 사용. **어떤 결과든 step 4 로 진행해서 chain dump + manual
chain rule 검증부터 거쳐야 함**:

- **N=1 정확 + N≥2 fail** → 버그가 cross-substep state propagation 에 있을
  *가능성이 높음*. Step 4 에서 첫 번째 BW substep (t=N-1) 의 chain 은
  대조용으로, 두 번째 BW substep (t=N-2) 의 chain 을 집중 dump.
- **N=1 부터 fail** → single-step backward 로직 자체에 버그. Step 4 에서
  t=N-1 (= N=1 case) 의 chain 을 dump.
- **N=1 정확 + N=2 정확 + N≥4 fail** → 누적 numerical noise 가능성도 있지만
  진짜 버그라면 chain 의 비선형성 또는 cross-substep accumulation.

> ⚠️ **함정**: N=1 정확 + N≥2 fail 만 보고 *primal staleness 의심* 으로
> 직행하면 안 됨. 그건 *하나의 가능성*일 뿐이고, chain 의 어디서 잘못된
> 값이 도입되는지 모르는 상태에서는 가설 검증/반증의 사이클을 효율적으로
> 돌릴 수 없음. **step 4 의 chain dump 가 정확한 stage 를 알려준다 →
> 그 다음에 step 7 의 primal 검사가 의미 있음.**

### Step 4: Backward chain dump 후 manual chain rule 검증 (필수)

**Skip 불가**. 어떤 step 3 결과든 이걸 거쳐서 chain 의 어느 stage 에서
잘못된 값이 도입되는지 정확히 식별해야 함.

**🎯 최종 목표 (이걸 명시적으로 리포트해야 진행 가능)**:
> "이 chain 에서 *어떤 함수* (kernel.grad) 가 *어떤 .grad 필드* 에 *어떤
> 값* 을 잘못 쓰는지" 한 줄로 답할 수 있을 때까지 step 4 를 끝내지 말 것.

이걸 식별하지 않고 "primal 일거다", "cross-substep leak 일거다", "ordering
일거다" 같은 *추측* 으로 fix 시도하는 건 가이드 P3 의 함정. Fix 가 우연히
부분적으로 작동해도 nearby DOF 가 악화되거나 다른 seed/N 에서 regression
발생.

**Backward verification 방법 (필수 순서)**:

1. Bad entry (예: ctrl_force.grad[i]) 가 *어떤 stage 에서 처음 wrong*
   되는지 확인. 끝 stage 부터 거꾸로:
   - 최종 ctrl_force.grad — 이게 wrong
   - 그 직전 force.grad — wrong? right?
   - 그 직전 acc.grad — wrong? right?
   - ...
   각 stage 의 manual chain rule 결과 = kernel 출력 비교.

2. **manual chain rule 은 *kernel 이 실제로 본 primal* 을 사용** (가장 큰
   함정. 가정된 깨끗한 값으로 계산하면 false positive silent drop 처럼 보임).

3. 처음 *kernel 출력 ≠ manual chain rule* 인 stage 를 찾으면 그게 bug 의 source.

4. 그 stage 의 입력 .grad 도 검증해서 *입력이 wrong* 인지 *kernel 자체가
   wrong* 인지 구분.
   - 입력 wrong → 더 상류 stage 추적 (재귀)
   - kernel 자체 wrong → step 5 (manual backward) 로

> ⚠️ **함정**: 한 stage 의 chain rule 만 검증하고 그게 맞다고 "다른 곳이
> 문제겠지" 라고 단정하지 말 것. 여러 stage 를 *모두* 비교해서 *처음으로*
> 갈리는 곳을 찾아야 함.

Bad entry 에 기여하는 함수들을 backward chain 따라 역추적. 각 stage 에서
`.grad` field 를 dump (`GENESIS_DEBUG_GRAD=2`).

```
post-loss.backward → links_pos.grad
post-UCS.grad      → qpos.grad
post-begin_bw_sub  → qpos_next.grad (swap from qpos.grad)
post-step_2.grad   → qpos.grad, vel.grad, acc.grad
post-compute_qacc  → force.grad
post-fwd_dyn.grad  → ctrl_force.grad ← 우리가 보고 싶은 값
```

각 stage 의 입출력으로 manual chain rule 식을 numpy 로 작성해서 비교.

> ⚠️ **가장 큰 함정**: Manual chain rule 에 들어가는 forward primal 값은
> *kernel 이 실제로 보는 값* 이어야 함. 이론적 가정 (e.g., "rot0 은 identity
> 일 것이다") 으로 계산하면 false positive 의 silent drop 으로 보임.
>
> **올바른 방법**: kernel 직전에 Python 에서 primal 값을 dump 하고 그 값으로
> numpy chain 계산. 우리 `/tmp/numpy_verify.py` 패턴 참고.

**Exit 조건 도달 후**: 갈리는 stage 의 kernel 을 Step 5 로 (manual backward
로 대체 + 진단적 결과 관찰).

### Step 5: Manual backward 로 대체 + 결과 관찰 (진단적 시도)

Step 4 에서 식별된 buggy kernel 을 manual backward 로 대체. **목적은 해결
자체가 아니라 *해결되는지를 관찰* 해서 bug 의 정체를 분류** 하는 것:

- **manual 로 해결됨** → autodiff bug 확정. Step 6 으로 (minimal repro
  + Quadrants 팀 보고).
- **manual 로도 해결 안 됨** → autodiff 가 아니라 *입력 (forward primal
  또는 입력 .grad)* 이 잘못된 것. Step 7 로 (primal 검사).

**구현 절차**:

1. **수식 자체 사전 검증** (필수): manual kernel 작성 후 numpy verification.
   ```python
   # 1. kernel 의 실제 primal 을 Python 에서 dump (직전 monkey-patch)
   # 2. 같은 primal 로 numpy 에서 chain rule 계산
   # 3. kernel 출력과 비교
   diff = numpy_result - kernel_result
   assert np.abs(diff).max() < 1e-15  # FP64 floor
   ```
   이 검증 없으면 step 6 의 분기 ("baseline 과 같은 wrong vs 다른 wrong")
   가 무의미. Manual 수식 자체 bug 인지 진짜 primal 문제인지 구분 불가능.

2. **Wire 해서 production 적용**: auto-AD 의 `kernel.grad` 호출 자리에
   manual kernel 호출 삽입 (또는 대체).

3. **테스트 재실행** (sweep + per-DOF) 해서 rel error 변화 관찰.

### Step 6: 결과에 따른 분기

#### Case A: Manual backward 가 해결함 ✅ → **Autodiff bug 확정**

Bug 의 정체: Step 4 에서 식별한 kernel 의 *auto-AD reverse* 가 silent drop
또는 잘못된 chain rule 을 생성하는 Quadrants AD bug.

**다음 작업**:
1. **Minimal repro script 작성** — buggy kernel 만 standalone 으로 추출,
   같은 primal/input.grad 값으로 호출 → silent drop 재현.
   - 입력 primal 값을 Python 에서 dump (직전 monkey-patch).
   - Standalone 환경에서 같은 입력 + 같은 forward 수식 + Quadrants AD `.grad`
     로 재현.
   - Verify: standalone 환경에서도 manual 의 numpy 결과와 auto-AD 결과
     불일치 확인 → silent drop reproduced.
2. **Quadrants 팀에 보고** — minimal repro 첨부.
3. **Production code 유지** — manual backward 는 fix 가 들어오기 전까지
   임시 우회로 (Case 1 정당화).

#### Case B: Manual backward 가 해결 못함 ❌ → **입력이 잘못됨**

Bug 의 정체: Step 4 에서 식별한 kernel 의 *입력* 이 잘못된 값. Kernel 자체
(또는 manual replacement) 는 정확하지만 wrong input → wrong output.

**다음 작업**: Step 7 로 (primal / 입력 .grad 검사).

### Step 7: Forward primal / 입력 검사

Backward 시점의 primal 값을 kernel 직전에 Python 에서 dump하고 *예상값* 과
비교. 예상값은 BW chain 을 거슬러 올라가서 계산:

```python
# Python-side primal capture pattern
captured = {}
def wrapped_kernel(*args, **kwargs):
    captured['qpos_at_bw'] = qd_to_numpy(self._rigid_global_info.qpos, copy=True)
    captured['vel_at_bw'] = qd_to_numpy(self.dofs_state.vel, copy=True)
    orig_kernel(*args, **kwargs)
```

**Quadrants/zerocopy 환경 specific patterns 체크:**

1. **View aliasing** — 가장 자주 만나는 패턴.
   ```bash
   grep "qd_to_numpy\|qd_to_torch" | grep -v "copy=True"
   ```
   `copy=True` 없이 받은 numpy/torch array 는 underlying buffer 의 view.
   이후 같은 buffer 가 overwrite 되면 silent mutation.
   
   **과거 사례 (이 가이드의 트리거):** `RigidSolver.save_ckpt` 의
   `qd_to_numpy(_rigid_adjoint_cache.qpos)` 가 view 저장 → 다음 forward
   substep 의 `kernel_save_adjoint_cache` 가 같은 buffer overwrite →
   ckpt aliasing → BW 시 wrong primal load.

2. **State restoration order**:
   ```
   prepare_backward_substep    (cache → state, UCS forward)
   substep replay              (forward kernels in BW mode)
   copy_next_to_curr_no_check  (vel_next → vel, qpos_next → qpos)
   UCS.grad chain
   begin_backward_substep      (state restore to cache + grad swap)
   manual/auto BW kernels
   ```
   각 단계에서 primal field 가 어떤 timestamp 의 값을 들고 있는지 명확히.

3. **`is_backward=True` 의 forward 동작 차이**: `kernel_step_2_post_integrate`
   의 `if qd.static(not is_backward): func_copy_next_to_curr` 같이 BW mode
   에서는 in-place update 가 skip 되는 경우 있음. 이게 primal staleness
   를 만들 수 있음.

4. **Adjoint cache slot 의미**: `_rigid_adjoint_cache.qpos` 가
   `(substeps_local + 1, n_qs, n_envs)` shape 일 때, 각 slot 이 어느
   substep 의 시작/끝 state 를 담는지 명확히 trace.

---

## Common pitfalls (우리가 빠진 함정 모음)

### P1. "Silent drop" 진단의 false positive
"Manual chain 식과 kernel 결과가 다르다" 만 보고 autodiff 의 silent drop 으로
판단하면 위험. 실제로는 *서로 다른 primal* 로 계산한 같은 chain 일 수 있음.

대응: Step 4 에서 manual chain 은 *반드시* kernel 의 실제 primal 을
사용해서 계산.

### P2. Manual backward 검증 누락
Manual kernel 구현해놓고 검증 없이 production 적용. 만약 manual 자체에
버그가 있으면 Step 6 의 정량 분기 (동일 wrong vs 다른 wrong) 가 무의미해짐.

대응: numpy verification 필수.

### P3. N=1 vs N≥2 비대칭 무시 (또는 과도하게 신뢰)
N≥2 가 catastrophic 이면 single-step 로직에 집중하기 쉬움. 그러나 N=1 이
정확하면 single-step 무죄. 시간 낭비 방지.

대응: Step 3 분기를 가장 먼저 수행.

**역방향 함정**: N=1 정확 + N≥2 fail 이라고 *primal staleness 의심* 으로
**직행하지 말 것**. 그건 가능한 원인 중 하나일 뿐. Chain 의 어느 stage 에서
잘못된 값이 들어오는지 모르면 가설을 무한히 시도하게 됨 (UCS refresh,
cache copy, state 재계산 등). 반드시 step 4 (chain dump) 거쳐서 *어느
stage* 가 문제인지 식별 → 그 stage 의 입력 / primal 검사 → fix.

### P4. FD precision floor 와 real bug 혼동
FD 의 `(lp - lm) / (2*eps)` 가 FP64 precision 으로 inflate 되어 rel err
크게 나오면 fake bug 처럼 보임.

대응: Richardson FD 로 eps 변화 시 fd 값 수렴 확인. 여러 eps 에서
consistent 한 값이 true gradient.

### P5. Mask atol 빼먹기
`|fd| < atol` 인 entry 의 rel err 은 의미 없음. 마스킹 안 하면 dominant
가 되어 진짜 버그 가림.

대응: `mask = |fd| > 1e-10` 적용한 max rel 사용.

### P6. Dead-end `.grad` field 를 의심 대상으로 삼기
중간 stage 의 `.grad` dump 에서 `cinr_*`, `cdof_*`, `cd_*`, `cfrc_*` 같은
fields 에 residual 값이 보이면 leak 처럼 보임. 그러나 이런 field 들은 대부분
*forward 의 `Y = ... + X` 패턴의 X 쪽* 으로 downstream consumer 가 없음
(또는 매우 약함). 따라서 이런 `.grad` 잔여값이 ctrl_force.grad chain 에
실질적으로 영향을 줄 가능성은 낮음.

**대응**: 이런 field 들의 zeroing 을 시도해도 ctrl_force.grad 가 변하지
않으면 dead-end 확정. 디버깅 우선순위에서 후순위로. 진짜 leak 은 보통
`vel.grad`, `qpos.grad`, `acc.grad`, `force.grad`, `mass_mat.grad` 처럼
forward chain 의 **load-bearing** field 에 있음.

**과거 경험**: 다양한 link-state `.grad` 잔여값을 zeroing 시도했지만 J1/J4
multistep 결과 변화 없었음 (`notes/diag_j1_n2_substep_leak.txt` 참고).

### P7. `substep_pre_coupling_grad` 의 함수 순서 변경 금지
이 함수 안의 kernel 호출 순서는 *state semantics* 가 미묘하게 얽혀 있음.
순서를 바꾸면 forward primal 의 timestamp 가 어긋나서 silent wrong gradient
발생.

특히 주의할 페어:
- `kernel_copy_next_to_curr_no_check` ↔ UCS.grad chain — `qpos` 가 어느
  시점 값인지에 따라 UCS Jacobian 결과 달라짐
- `kernel_begin_backward_substep` ↔ manual/auto BW kernel — swap 과
  state restore 가 한 번에 일어남
- forward replay 의 `self.substep(f)` 호출 시점 — adjoint cache 가
  여기서 다시 save 되므로 위치 바꾸면 cache content 달라짐

**대응**:
1. 순서를 바꿔야 한다고 생각되면, **반드시** 변경 전후로 J1~J5 전체 sweep
   돌려서 regression 확인.
2. 새 kernel 을 *추가* 할 때는 기존 순서를 *변경하지 않고* 적절한 위치에
   삽입.
3. 만약 정말 바꿔야 한다면, 각 kernel 이 의존하는 forward primal 의
   timestamp 를 문서화하고 변경 후에도 일관되는지 검증.

---

## Reusable diagnostic infrastructure

| 파일 (위치) | 용도 |
|---|---|
| `notes/diag_all_topo_relerror_sweep.py` | Topology × N sweep, 다중 seed |
| `notes/diag_multistep_worst_case.py` | MJCF templates (J1~J5) |
| `notes/diag_j4_n2_perdof.py` | Per-DOF rel err breakdown 패턴 |
| `notes/diag_manual_bw_verify.py` | Manual backward 의 수식 검증 (numpy) |
| `/tmp/fd_richardson.py` (recreate) | Richardson FD eps 수렴 분석 |
| `/tmp/numpy_verify.py` (recreate) | kernel 실제 primal + numpy chain |
| `/tmp/check_cache.py` (recreate) | adjoint cache content inspection |
| `_debug_grad_dump` (`rigid_solver.py:1288`) | Stage 별 .grad max/norm + verbose per-element |

**Pattern: monkey-patch substep_pre_coupling_grad**

```python
orig = rs.RigidSolver.substep_pre_coupling_grad

def patched(self, f):
    pre_state = capture(self)
    orig(self, f)
    post_state = capture(self)
    log.append((f, pre_state, post_state))

rs.RigidSolver.substep_pre_coupling_grad = patched
```

**Pattern: monkey-patch specific kernel**

```python
import rigid_solver as rs
orig_kernel = rs.kernel_xxx
captured = []
def wrapped(*args, **kwargs):
    captured.append(capture(args[0]))  # args[0] is usually self/state
    orig_kernel(*args, **kwargs)
rs.kernel_xxx = wrapped
```

---

## Reference: past incidents

| 날짜 | 증상 | Root cause | Fix |
|---|---|---|---|
| 2026-04 | J4 N=1 silent drop on free quat | `func_solve_mass_entity` 의 cross-iter same-buffer reads (Quadrants AD bug) | Manual `kernel_manual_compute_qacc_bw` via IFT |
| 2026-05-11 | J1~J3 multistep grad leak | Kernel-side `.grad = 0.0` write silently dropped | Python-side `qd_zero_grad` 사용 |
| 2026-05-12 | J4/J5 multistep workaround | Quadrants AD chain 더 깊은 곳 | Manual backward (임시) |
| 2026-05-13 | J4 N≥2 catastrophic (이 문서의 트리거) | `save_ckpt` view aliasing | `qd_to_numpy(..., copy=True)` (3줄) |

---

## TL;DR (cheat sheet)

```
1. Detect: FD sweep, mask |fd|>atol, multi-seed consistency 확인
2. Localize: per-DOF + per-step dump → bad (seed, t, DOF) 식별
3. N=1 vs N≥2 정보 수집 (skip 금지 but 분기점 아님):
   - N=1 정확 + N≥2 fail → step 4 에서 t<N-1 BW chain 집중
   - N=1 부터 fail        → step 4 에서 t=N-1 BW chain 집중
   ⚠️ 이 결과로 step 4 를 SKIP 하면 안 됨. 어디가 잘못됐는지
      모르는 상태에서 primal 의심에 직행하면 가설 무한 시도 함정.
4. Chain dump + manual chain rule (필수):
   - 각 BW stage 의 .grad 값 dump (GENESIS_DEBUG_GRAD=2)
   - kernel 의 실제 primal 을 Python 에서 dump 해서 numpy chain 계산
   - 어느 stage 에서 manual vs kernel 결과가 처음 갈리는지 식별
   - 🎯 EXIT 조건: "kernel X 가 .grad Y 에 wrong 값 Z 를 쓴다" 한 줄 답.
5. 갈리는 stage 의 kernel 검토:
   - 코드 자체에 오류? → fix
   - 수식 정상이면 → manual backward 작성 + numpy verification (수식 검증)
                    → production 적용 → 결과 관찰 (진단적 시도)
6. 결과 분기:
   - manual 로 해결됨 → autodiff bug 확정
     → minimal repro 작성 + Quadrants 팀 보고
     → manual 은 임시 우회로 유지
   - manual 로도 해결 안됨 → 입력 (forward primal 또는 입력 .grad) 문제
     → Step 7
7. Forward primal / 입력 검사:
   - 문제 stage 의 kernel 직전 primal / 입력 .grad 를 Python 에서 dump
   - 예상값과 비교 (chain 거슬러 올라가서 계산)
   - 다르면: view aliasing / state restore / is_backward 동작 등 확인
```
