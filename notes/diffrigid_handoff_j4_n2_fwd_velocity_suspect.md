# Diffrigid Handoff — J4 N=2 Step 4 결론 + Step 5 진입

날짜: 2026-05-13
브랜치: `20260512_diff_rigid_demo`
참고 commit: `b6591496` (manual UC backward wired)

---

## Step 1~3 결과

### Step 1 (Detect) — Quick sweep N=1,2,4 × 5 seeds

| topology | N=1 | N=2 | N=4 |
|---|---|---|---|
| J1_free | 4e-11 | 1.8e-3 | 7e-3 |
| J2_revolute | 0 | 0 | 0 |
| J3_prismatic | 2e-12 | 3e-12 | 2e-12 |
| **J4_free_rev** | **7e-3** | **6.97** | **7.35** |
| J5_chain3 | 2e-2 | 3e-3 | 7e-4 |

**Winner: J4 N=2 max rel = 6.97**

### Step 2 (Localize)
- step t=1 (FIRST BW): FP64 floor ✅
- step t=0 (SECOND BW): catastrophic on root_y/z/wx/wy/arm_rev. root_x/wz 정상.

### Step 3
N=1 정확 + N≥2 fail → cross-substep state propagation 문제.

---

## Step 4 — Chain dump + numpy 검증 (완료)

### 검증 1: fvol.grad (post-fwd_velocity.grad, stage 11)
- numpy chain rule 작성 (`/tmp/diag_fvol_verify_v2.py`)
- 6 calls (1st substep mid + 2nd substep mid + 2nd substep initial) 모두 검증
- **max|np-k| = 2.6e-23 (FP64 floor)** ✅ → **fvol.grad 무죄**

### 검증 2: forward primal N=1 vs N=2 second substep (`/tmp/diag_primal_n1_vs_n2.py`)
- mid-SPC: vel/cd_v/cd_a 가 rel 1e-7~1e-8 차이 (소량, FP64 보다 큼)
- INITIAL: 모두 0 차이
- ana[t=0] 의 700% rel err 를 설명하기엔 너무 작은 차이 → 부차

### 검증 3: cross-substep state zero-out gateway (`/tmp/diag_zero_xsubstep_state.py`)

baseline max rel = 6.97. Zero-out 결과:

| Zero set | max rel | 변화 |
|---|---|---|
| ALL (cd_*+cdof_*+cdofd_*) | 5.08 | 일부 entry 개선, 일부 악화 (no monotonic fix) |
| cd_v + cd_a | 5.72 | partial |
| cdof_v + cdof_a | 6.97 | almost no change |
| cdofd_v + cdofd_a | 7.62 | slight regression |

→ **load-bearing + wrong** (P6 dead-end 아님, 정확한 chain rule output 필요).

### Step 4 한 줄 답 (가이드 exit 조건):
> **prev BW substep 의 `func_update_force.grad` (= `inertial_mul.grad` + `motion_cross_force.grad`) 가 `cfrc_*.grad → cd_v/cd_a.grad` chain 에서 silent drop 또는 wrong 값을 쓴다. ctrl_force.grad path 는 정확 (ana[t=1] FP64 floor), cfrc_*.grad 도 정확할 가능성, 그러나 cfrc → (inertial_mul + motion_cross_force) → cd_v/cd_a chain 의 reverse 가 wrong.**

`kernel_forward_dynamics_without_qacc` 내부 sub-funcs (line 1366~):
- `func_compute_mass_matrix` (mass_mat ← cdof_*, cinr_*, crb_*, links_state)
- `func_torque_and_passive_force` (ctrl_force, qf_passive)
- `func_update_acc`
- **`func_update_force`** (cfrc_vel/ang ← inertial_mul(cinr, **cd_v/cd_a**) + motion_cross_force(**cd_a/cd_v**, f2))  ← **의심 1순위**
- `func_bias_force` (qf_bias ← cdof_* · cfrc_*; cd_v/cd_a 직접 사용 안 함)

**`func_bias_force` 는 cd_*/cd_a 를 직접 사용 안 함** — 처음 가설 dismiss.
`func_update_force` 의 forward 식 (line 1030-1076):
```
f1_ang, f1_vel = inertial_mul(cinr_pos, cinr_inertial, cinr_mass, cdd_v, cdd_a)
f2_ang, f2_vel = inertial_mul(cinr_pos, cinr_inertial, cinr_mass, cd_v, cd_a)
f3_ang, f3_vel = motion_cross_force(cd_a, cd_v, f2_ang, f2_vel)
cfrc_vel = f1_vel + f3_vel + cfrc_applied_vel + cfrc_coupling_vel
cfrc_ang = f1_ang + f3_ang + cfrc_applied_ang + cfrc_coupling_ang
# Then propagate cfrc upward in tree: cfrc[parent] += cfrc[i_l]
```

`cd_v/cd_a.grad` 는 reverse 의:
- `inertial_mul.grad` (f2 chain) → cd_v.grad, cd_a.grad
- `motion_cross_force.grad` (f3 chain) → cd_a.grad, cd_v.grad

---

## Step 5 — Manual replacement (시작)

### 다음 구체적 액션
1. **`inertial_mul` 과 `motion_cross_force` 의 forward 식 읽기** (`genesis/utils/geom.py`)
2. **numpy chain rule 작성**:
   - `cfrc_vel/ang.grad` 가 input (정확 가정).
   - Forward primal: `cinr_pos, cinr_inertial, cinr_mass, cd_vel, cd_ang` (kernel 직전 캡처)
   - Output: `cd_vel.grad, cd_ang.grad` (per-link, 2개)
3. **prev BW substep 의 fwd_dyn.grad call 직전/직후 캡처**: cfrc_*.grad (입력) + cd_v/cd_a.grad (출력)
4. **numpy chain rule 적용** → manual cd_v/cd_a.grad 계산
5. **kernel output 과 비교** (FP64 floor 일치 여부)
6. **결과**:
   - 일치 → func_update_force 의 cd 부분 OK, 다른 출력 path 의심 (cdof_*, cdofd_*)
   - 불일치 → **autodiff bug 확정 (Case A)** → manual replacement 의 단위는 func_update_force.grad 전체 또는 inertial_mul/motion_cross_force만.

### 의심 우선순위 (수정)
- **`func_update_force.grad` (inertial_mul + motion_cross_force)**: cd_v/cd_a.grad 의 유일한 source. **1순위**.
- `func_compute_mass_matrix.grad`: cdof_*.grad chain 의 source.
- `func_bias_force.grad`: cdof_*.grad 와 cinr_*.grad chain. cd_v/cd_a 직접 X.
- `func_torque_and_passive_force.grad`: ctrl_force.grad 정확하므로 *이 sub-func 는 OK* (chain 통과 검증됨).

---

## 검증 인프라

| 파일 | 용도 |
|---|---|
| `/tmp/diag_fvol_verify_v2.py` | fvol.grad numpy chain rule 검증 (완료, OK) |
| `/tmp/diag_primal_n1_vs_n2.py` | forward primal N=1 vs N=2 비교 (완료, 부차) |
| `/tmp/diag_zero_xsubstep_state.py` | cross-substep state zero gateway (완료, load-bearing wrong 확정) |
| `/tmp/parse_dump.py` | dump → 압축 stage table |
| `/tmp/j4_n2_dump.txt` | raw GENESIS_DEBUG_GRAD=2 dump |

### 다음 작성할 진단 (Step 5 액션)
- `/tmp/diag_bias_force_verify.py` (또는 fwd_dyn 전체 verify) — input force.grad + primal 캡처 → numpy chain rule → cd_v/cd_a/cdof_*/cdofd_*.grad 비교

---

## 가이드 활용 self-check
- ✅ Step 1: Detect 완료 (FD sweep, J4 N=2 winner)
- ✅ Step 2: Localize 완료 (per-DOF: root_y/z/wx/wy/arm_rev catastrophic on t=0 BW)
- ✅ Step 3: N=1 vs N≥2 패턴 — cross-substep propagation
- ✅ Step 4: chain dump + numpy verification
  - fvol.grad 검증 완료 (무죄, FP64 floor) — `/tmp/diag_fvol_verify_v2.py`
  - forward primal N=1 vs N=2 sub-1e-7 차이 (부차) — `/tmp/diag_primal_n1_vs_n2.py`
  - Cross-substep state zero gateway: load-bearing wrong 확정 — `/tmp/diag_zero_xsubstep_state.py`
  - **Exit 한 줄 답**: prev BW substep 의 fwd_dyn.grad 가 cross-substep state (cd_v/cd_a/cdof_*/cdofd_*/cinr_*).grad 를 wrong 으로 쓴다.
- 🚧 Step 5: manual chain rule 작성 + 검증 (시도 1 결과 정리)
  - 시도: func_update_force.grad isolated 검증 (`/tmp/diag_update_force_grad_verify.py`)
  - **결과: 검증 단위 잘못됨** — cfrc_*.grad 는 fwd_dyn.grad 의 *내부 중간 state*, isolated input 아님. fwd_dyn.grad 전체 chain rule 을 한 번에 작성해야 valid 비교.
  - 다음 시도 방향:
    - (a) **fwd_dyn.grad 전체 chain rule numpy 작성** — sub-funcs 5개 (compute_mass_matrix, torque_and_passive_force, update_acc, update_force, bias_force) 의 reverse 를 한 번에. 크지만 정확.
    - (b) **manual replacement 직접 시도** (Step 5 본단): `kernel_manual_fwd_dyn_bw` 를 작성해서 wire 하고 ana[t=0] 변화 관찰. 빠르지만 chain rule 구현 부담 동일.
    - (c) **더 fine 한 zero-out 분석** (cinr_*.grad 영향 등) 으로 wrong 의 source 를 더 좁힘. 빠른 sanity.
- ⏸ Step 6: 결과 분기
- ⏸ Step 7: forward primal 검사 (이미 한 차례 거침 — sub-1e-7 차이로 부차 확인)

## 다음 세션 진행 옵션 (priority)
1. **Option (c) — fine-grained zero-out 분석**: cinr_inertial.grad, crb_*.grad 등 추가 zero-out 시도. ana[t=0] 와 fd 의 일치도 가까워지는 set 식별. wrong 의 source kernel 단위 더 좁힐 단서.
2. **Option (b) — manual_fwd_dyn_bw 작성 시작**: 한 sub-func 씩 manual reverse 추가, 매 추가마다 ana[t=0] 측정. 단계적 wrong source 분리.
3. **Option (a) — 전체 numpy 검증**: 가이드 P1 정확성 보장. 가장 안전.

---

## Fine zero-out 결과 (`/tmp/diag_fine_zero.py`, 자율 진행 후 완료)

| Field | rel after zero | Δ vs baseline (6.97) |
|---|---|---|
| **`links_state.cinr_pos`** | **4.31** | **-2.66** ← largest single-field improvement |
| `links_state.cd_ang` | 5.04 | -1.93 |
| `links_state.cd_vel` | 6.29 | -0.68 |
| `dofs_state.cdofd_vel` | 7.62 | **+0.64** (worsen) |
| `cdd_*`, `cinr_mass`, `crb_*`, `cfrc_*`, `qf_*` | 6.97 | 0 (변화 없음) |

→ `cinr_pos.grad` 가 가장 load-bearing wrong source. cinr_mass/cinr_inertial/crb_* 등은 거의 dead-end (가이드 P6).

## Step 4 한 줄 답 v3 (precise)
> **`func_update_force.grad` 의 `inertial_mul.grad` 가 cinr_pos.grad 에 wrong values 를 쓴다.**
> cinr_pos.grad cross-substep 운반 → next BW 의 COM_links.grad → links_state.{pos,quat}.grad → manual UC.grad → ctrl_force.grad[t=0] wrong.

## 다음 세션 액션 (Step 5)
1. **`inertial_mul.grad` 의 numpy chain rule 작성 + 검증**:
   - fwd_dyn.grad 전체를 numpy 로 가는 대신 *cinr_pos.grad output* 부분만 isolate
   - 단, isolated input (f2_*.grad) 의 chain 도 fwd_dyn 내부 → fwd_dyn.grad call 직후 capture 필요
   - 또는 *manual replacement* (단일 sub-func)
2. **manual `kernel_update_force_bw`** 작성 (가이드 P9 따라 *모든* joint type/branch faithful)
3. Quadrants 팀 보고용 minimal repro: inertial_mul.grad 의 silent drop (cinr_pos.grad 부분) 재현 스크립트

---

## Step 5 sub-1 결과 (자율 진행, `/tmp/diag_inertial_mul_standalone.py`)

**Test 1 (standalone Quadrants kernel for `inertial_mul`)**:
- kernel.grad = numpy chain rule, max|k-n| = 0 또는 FP64 2e-25 (정확)
- **inertial_mul.grad 의 reverse 자체는 Quadrants AD 에서 정확**

**Test 2 (J4 in-context)**:
- 1st substep fwd_dyn.grad 가 cinr_pos.grad 를 Δ = [-2.6e-8, 5.5e-5, 5.9e-5] (chassis), [-7.2e-8, 5.5e-5, 1.5e-4] (arm) 으로 변경
- max | post fwd_dyn.grad | = 1.46e-4 (= stage 8 의 dump 값과 일치)

→ **결론: wrong source 는 inertial_mul.grad 자체가 아니라 fwd_dyn 의 context (sub-func 간 .grad field 공유 / cross-iteration writes / atomic_add 같은 interaction)**.

이는 이전 UCS per-link split 과 같은 카테고리: 같은 kernel 안의 multi-sub-func reverse 가 cross-iteration .grad field 를 silent drop. **Fix candidate = kernel splitting** (fwd_dyn 의 sub-funcs 를 별도 kernel 로 isolate).

## Step 5 sub-2 (다음 액션)
1. **`kernel_update_force_split`** 작성 — fwd_dyn 의 update_force 를 별도 kernel 로 split. 다른 sub-func 와 분리해서 cinr_pos.grad atomic_add silent drop 가능성 분리.
2. 또는 **`kernel_forward_dynamics_without_qacc_split`** — sub-func 5개 각각 별도 kernel.
3. ana[t=0] 변화 측정 → 효과 확인.

만약 (1) 또는 (2) 가 효과 → Quadrants AD 의 kernel-internal silent drop 확정 → minimal repro + 보고.
효과 없으면 → 다른 가설 (compute_mass_matrix.grad 의 cinr_pos.grad chain 도 의심).

