# Diffrigid: `func_solve_mass` Manual Backward — Handoff

날짜: 2026-05-12  
브랜치: `20260503_diff_rigid_demo`  
관련 commit: `9fc84540` (J4 silent-drop diagnostics)

---

## TL;DR

J4/J5 multi-step `control_dofs_force` gradient mismatch (xfail 의 root cause)
는 *Quadrants AD silent drop* 도 *FK chain rule 결함* 도 아니다.

**진짜 root cause**: `func_solve_mass_entity` (forward_dynamics.py:660) 의
*Step 1 BW skip* 가 `self.substep(f)` (backward path 의 forward 재실행) 안에서
`acc = 0` 을 만들고, 그 결과 `vel_next = 0`, `qpos_next = identity` 로 reset.
그 *잘못된 forward primal* 위에서 모든 backward chain (FK reverse 포함) 이
계산되어 `∂R/∂qy` 등이 *zero-perturb form* 으로 나옴 → silent drop 처럼 보임.

**Fix 방향 (다음 세션)**: `func_solve_mass` 의 backward 를 **manual 으로** 구현.
Step 1 BW skip 제거 + `kernel_compute_qacc.grad` + 외부 Phase B (`kernel_solve_mass_step1_one_dof_bw` + `.grad`, `kernel_solve_mass_step2_reverse_bw`) 를 **단일 manual kernel** 로 통합.

---

## 진단 과정 요약 (이번 세션)

### 1. Baseline 측정

`notes/diag_multistep_worst_case.py` (5 토폴로지 × N∈{4,16,32} × 3 seeds):

| 토폴로지 | N=4 max\|diff\| | N=16 | N=32 |
|---|---|---|---|
| J1 freejoint | 5.4e-18 | 1.5e-16 | 2.4e-15 |
| J2 revolute | 0 | 0 | 0 |
| J3 prismatic | 3.4e-19 | 3.9e-17 | 1.1e-15 |
| **J4 free+rev** | **1.8e-7** | **7.5e-6** | **5.6e-5** |
| **J5 chain3** | **1.7e-7** | **4.8e-7** | **1.2e-6** |

J1~J3 (single-DOF entity) 정상. J4/J5 (multi-DOF) 만 mismatch. N 따라 거의 선형 누적.

저장: `notes/diag_multistep_worst_case_baseline.txt`.

### 2. J4 N=1 dump 분석

`notes/diag_j4_n1_grad_dump_full.py` 으로 backward 의 각 stage 마다 모든 forward field 의 `.grad` 전체 값 dump.

seed=1001 N=1 결과 (`notes/diag_j4_n1_mismatch.txt`):
```
              ana          fd            diff
d=0 root_x  +2.667e-05  +2.667e-05  +1.3e-13    OK
d=4 root_wy -1.022e-08  +1.090e-08  -2.1e-08    부호 반대 (rel ≈ -1.94)
d=5 root_wz -2.159e-09  +2.158e-09  -4.3e-09    부호 반대 (rel = -2.0)
```

**관찰**: `rel ≈ ±2` — `ana ≈ -fd`. 단순 over-counting 아님, chain 의 어떤 부호가 뒤집힘.

### 3. Backward chain stage 별 검증

`compute_qacc.grad` (`force.grad = mass^-1·acc.grad`) ✅ 수작업 검증 통과.
`step_2.grad` (integrator backward) ✅ 수작업 검증 통과.
`update_cartesian_space.grad` (FK reverse): dump 에서 `out_grad[0]·∂out[0]/∂qy` chain 누락 확인.

### 4. `qd_transform_by_quat` standalone

`notes/quadrants_repros/case_3_transform_by_quat.py`: AD, FD, hand-derived 모두 일치 (max|diff|=0). **함수 자체는 정확**.

minimal repros cases 4-8 (cross-index write, parent_pos add, qd.func boundary, double-write+branch, Vec3 fields) — 모두 정상. **silent drop trigger 가 minimal repro 으로 분리 안 됨**.

### 5. Manual backward kernel for `update_cartesian_space` 시도

`genesis/engine/solvers/rigid/abd/manual_bw.py` 작성. `d_transform_by_quat__dq`, `d_quat_mul__dlhs`, `d_quat_mul__drhs`, `d_rotvec_to_quat__drotvec` 손-유도 — `notes/diag_manual_bw_verify.py` 으로 Quadrants AD 와 일치 확인 (max|diff|=0).

J4 link-by-link manual backward (`kernel_manual_uc_bw_one_link`) 구현 후 rigid_solver.py 의 .grad 호출 교체. 결과: **mismatch 그대로** (silent drop 값 동일).

### 6. Forward primal 결함 발견 (KEY)

`notes/diag_j4_n1_grad_dump_full.py` 의 dump 에 `_dbg` 추가:

```
BEFORE prepare_backward_substep:  qpos = post-integrate (qy ≈ -1.8e-4)
AFTER  prepare_backward_substep:  qpos = identity, qpos_next = post-integrate
AFTER  self.substep(f):           qpos_next = IDENTITY (!!!)
                                  acc = 0
                                  vel_next = 0
                                  force = u (정상)
```

**`self.substep(f)` 가 backward 모드 에서 `acc = 0` 으로 인해 `qpos_next` 를 identity 으로 reset**. 그 이후 `kernel_copy_next_to_curr_no_check` (line 1464) 가 `qpos = qpos_next = identity` 으로 copy.

결과: 모든 backward chain 이 *qpos = identity* primal 으로 계산 → `∂R/∂qy = -2·qy = 0` → 모든 quat-derivative chain term = 0 → silent drop 처럼 보임.

### 7. Root cause 추가 추적

`acc = 0` 의 source: `func_compute_qacc` 의 `func_solve_mass` 의 **Step 1 BW skip**:

```python
# forward_dynamics.py:694
if qd.static(not BW):    # BW 모드에서 Step 1 (L^T·w=y) skip
    for i_d_ in range(n_dofs):
        ...
```

**Step 1 BW skip 의 원래 이유** (line 680-693 주석):
- Step 1 forward 식: `out[i_d] = vec[i_d] - sum_{j>i_d} L[j,i_d] · out[j_d]`
- *cross-iter same-buffer read* (case 2 `step_bug` 패턴)
- Quadrants AD 의 *backward* 가 이 chain 을 silent drop
- 우회: 외부 `kernel_solve_mass_step1_one_dof_bw` (Phase B Python loop, case 2 `step_ok` 패턴 으로 cross-launch 분할)
- BW=True 모드 에서 Step 1 skip = *Quadrants AD 가 그 chain 을 trace 하지 않게* + 외부 Phase B 가 backward 따로 처리

**Trade-off**:
- BW skip 활성 (현재): 외부 Phase B chain 정확. **forward primal 깨짐** (acc/vel_next/qpos_next = 0).
- BW skip 비활성 (시도): forward primal 정상화. **Quadrants AD silent drop + Phase B 중복 → chain 깨짐 (mismatch 1e-5 악화)**.

### 8. 시도한 fix 들 (모두 부분 성공 또는 실패)

| Fix | 결과 |
|---|---|
| `kernel_copy_next_to_curr_no_check` 를 `self.substep(f)` *전* 으로 이동 | silent drop (-fd) → over-count (+2·fd). 부호 맞아짐. 다만 step_2.grad 의 pre-integrate qpos 의도 깨뜨림. |
| 위 + initial-UCS section manual 비활성 | 변화 없음 (N=1 에선 영향 없음). |
| Step 1 BW skip 제거 + 다른 변경 revert | mismatch 1e-5 (악화). Quadrants AD silent drop + 외부 Phase B 중복. |
| `func_integrate` 의 vel_next/qpos_next write 를 BW 가드 | 모든 grad = 0 (Quadrants AD trace 가 no-op 의 reverse). |

**결론**: 단순한 fix 안 됨. 진정한 fix 는 backward path 의 *manual override*.

---

## 다음 세션 계획: `func_solve_mass` Manual Backward

### 작업 범위 (Task #25-28)

1. **`func_solve_mass_entity` 의 Step 1 BW skip 제거** (forward_dynamics.py:694)
   - `if qd.static(not BW):` guard 제거. Step 1 forward 식 unconditional 실행.
   - 결과: BW=True 모드 에서도 `out (= acc_smooth)` 정상 계산 → `acc`, `vel_next`, `qpos_next` 정상.
   - Quadrants AD 의 trace 가 *cross-iter same-buffer chain* 을 trace 하지만 — 우리는 그 backward 안 호출 (= Quadrants silent drop 영향 없음).

2. **Manual kernel `kernel_manual_compute_qacc_bw` 작성** (manual_bw.py 에 추가)
   - Input: `acc.grad`
   - Output: `force.grad = M⁻¹ · acc.grad` (forward Step 1/2/3 식 그대로, I/O 만 다름)
   - Output: `mass_mat.grad += -force.grad ⊗ acc` (IFT)
   - 즉 *전체 manual chain* — 외부 Phase B 와 통합.

3. **rigid_solver.py 의 `substep_pre_coupling_grad` 수정**:
   - `kernel_compute_qacc.grad(...)` 호출 *제거*.
   - 외부 Phase B 우회들 *제거*:
     - `kernel_solve_mass_step1_one_dof_bw` (forward Python loop)
     - `kernel_solve_mass_step1_one_dof_bw.grad` (backward Python loop)
     - `kernel_solve_mass_step2_reverse_bw`
   - 대신 `kernel_manual_compute_qacc_bw(...)` 호출.

4. **검증**: J4 N=1 mismatch → 이상적으로 `<1e-10`. 그 후 full sweep + J1~J3 회귀 확인.

### Manual kernel 의 forward 식 derivation

`func_compute_qacc` forward:
```
acc_smooth = solve(M, force)        # M·acc_smooth = force, 즉 acc_smooth = M⁻¹·force
acc[d] = acc_smooth[d]              # identity copy
```

Backward (chain rule + IFT):
```
acc_smooth.grad += acc.grad         # copy
force.grad += M⁻¹·acc_smooth.grad   # M symmetric → M⁻ᵀ = M⁻¹
mass_mat.grad += -force.grad ⊗ acc_smooth   # IFT: ∂(M⁻¹·y)/∂M_ij = -(M⁻¹·y) ⊗ (M⁻¹·y) 의 component
```

Implementation 주의:
- `force.grad = M⁻¹·acc_smooth.grad` 는 forward 의 Step 1/2/3 식 그대로 (input/output 만 force/acc_smooth ↔ acc_smooth.grad/force.grad).
- forward primal 의 `mass_mat_L`, `mass_mat_D_inv` 사용 (현재 forward primal 정상).
- `mass_mat.grad` rank-1 outer product. n_dofs × n_dofs entries.

### 검증 기준

- `notes/diag_j4_n1_mismatch.py`: max|diff| 가 베이스라인 (2e-8) 보다 줄어야. 이상적으로 < 1e-10 (FP64 floor).
- `notes/diag_multistep_worst_case.py` (J4 N=32): max|diff| 5.6e-5 → 가능한 한 작게.
- `pytest tests/test_diff_forward_kinematics.py` J4/J5 multistep xfail → XPASS 가 목표.
- J1, J2, J3 single-DOF 회귀 없어야 (현재 0 또는 1e-15).

### 위험 / 주의

- **Forward primal value 변화**: Step 1 BW skip 제거 후 `out` 가 정상 값 → 후속 forward dynamics chain (mass_mat update 등) 에 영향. 다른 backward chain (`kernel_forward_dynamics_without_qacc.grad`) 가 *우리 manual 의 결과* 위에 계산.
- **mass_mat_L_bw / mass_mat_D_inv 의 사용**: 기존 Phase B (factor_mass stage_a/b/c_bw) 는 *유지*. mass_mat → mass_mat_L 의 backward chain 처리 — 우리 manual 의 mass_mat.grad seed 가 그 chain 의 input 이 됨.
- **multistep 영향**: N=1 에서 통과해도 N>1 에서 *substep 간 carry-over leak* 가 별도 issue 일 수 있음 (메모리 노트의 cdof/cinr/cfrc 누락).

---

## 진단 도구들 (notes/)

이번 세션에서 만든 진단 인프라:

| 파일 | 용도 |
|---|---|
| `notes/diag_j4_grad_dump.py` | J4 multistep backward stage 별 `_debug_grad_dump` (max/norm) |
| `notes/diag_j4_n1_grad_dump_full.py` | N=1 backward 각 stage 의 모든 forward field `.grad` 전체 per-element dump. `_dbg()` 가드 추가됨. monkey-patch `_full_dump` |
| `notes/diag_j4_n1_mismatch.py` | J4 N=1 ana vs FD per-DOF 비교 (3 seeds) |
| `notes/diag_j4_n1_mass_inv_verify.py` | M⁻¹ 손-검증 — mass_mat 의 LDLT factor 가 *transpose convention* 인지 확인 |
| `notes/diag_j4_n1_fd_eps_sweep.py` | FD eps sweep (1e-3 ~ 1e-10) — convergence 확인 |
| `notes/diag_j4_n1_forward_perturb.py` | ctrl_force ± eps → links_pos 변화 측정 (forward perturb) |
| `notes/diag_manual_bw_verify.py` | manual_bw.py 의 손-유도 함수들이 Quadrants AD 와 일치하는지 검증 |
| `notes/parse_j4_grad_dump.py` | dump txt → stage-by-stage magnitude table |
| `notes/format_j4_n1_full_dump.py` | full dump 의 zero field collapse + nonzero magnitude 정렬 |

## 검증 시 활용할 baseline 결과들

`notes/diag_j4_n1_mismatch.txt`: J4 N=1 (3 seeds) ana vs FD 표.  
`notes/diag_j4_n1_grad_dump_full_clean.txt`: 346 줄, backward 각 stage 의 모든 field `.grad` 값.  
`notes/diag_multistep_worst_case_baseline.txt`: J1~J5 × N∈{4,16,32} × 3 seeds 베이스라인.

---

## 임시 코드 변경 상태 (revert 됐는지 확인 필요)

다음 세션 시작 시 `git diff genesis/` 로 확인:
- `genesis/engine/solvers/rigid/rigid_solver.py`: manual_bw import + 두 곳의 manual kernel 호출 + initial-UCS section 의 비활성 코멘트 — *현재 manual 호출은 revert 됨 (기존 `kernel_update_cartesian_space_one_link.grad` 호출 복원)*. 단 *import 와 manual_bw.py 파일은 남아있음*.
- `genesis/engine/solvers/rigid/abd/forward_dynamics.py`:
  - `pos += vel * dt` (line 1584) — 이전 `pos = pos + vel * dt` 에서 변경. 사용자 지시: "+= 사용해야 함" (Quadrants AD reverse-mode 의 in-place 누적). **이 변경은 유지**.
- `genesis/engine/solvers/rigid/abd/manual_bw.py`: 새 파일, manual chain rule 함수 4개 + `kernel_manual_uc_bw_one_link`. **유지**.

**확인할 것**:
```bash
git diff genesis/engine/solvers/rigid/rigid_solver.py
git diff genesis/engine/solvers/rigid/abd/forward_dynamics.py
```

---

## 메모리에 저장된 관련 사실들 (MEMORY.md)

- `feedback_diffrigid_workaround_to_manual_bw.md`: "Quadrants AD 한계 우회는 임시 단계. 정확성 확보 후 manual backward 로 옮겨 forward 비용 회수."
- `project_quadrants_adstack_silent_failure.md`: adstack opt-in 플래그 + edge case silent drop.
- `feedback_diffrigid_phase_b.md`: LDLT backward fix 방법론.
- `feedback_multistep_grad_leak.md`: J4/J5 silent-AD chain loss xfail.

## 사용자 명시 가이드 (이번 세션 도중)

- Quadrants 의 `pos += vel * dt` 형식 강조 — Genesis 자체 코드에서도 `+=` 유지 (in-place 누적).
- "BW mode 에 따라 연산 누락 시키는 건 좋은 해결책 아님" — Step 1 BW skip 이 이전 다른 bug 의 fix 였지만 *substep 의 무결성을 훼손* 한다는 지적.
- 다음 작업: `func_solve_mass` 의 backward 를 manual 으로.
- 응답은 한국어 + 코드 주석/문자열은 영어.

---

## 첫 단계 코드 위치

- Step 1 BW skip 제거: `genesis/engine/solvers/rigid/abd/forward_dynamics.py:660-722` (`func_solve_mass_entity`)
- Manual kernel 추가: `genesis/engine/solvers/rigid/abd/manual_bw.py` (기존 update_cartesian_space manual kernel 옆에 추가)
- 호출 site 수정: `genesis/engine/solvers/rigid/rigid_solver.py:1370-1465` 부근 (`substep_pre_coupling_grad`)
  - 삭제 대상: factor_mass Stage A/B/C 사이의 Phase B Python loop (line 1438-1448 — Stage B 가 그건데 이건 *mass_mat factorization* 이라 *유지*. 별도 Phase B 가 다른 곳에 있는지 확인 필요).
  - 정확한 Phase B 위치: `grep -n "kernel_solve_mass_step1_one_dof_bw\|kernel_solve_mass_step2_reverse_bw" genesis/engine/solvers/rigid/rigid_solver.py`
