# Diffrigid Handoff — J4 N=2 compute_mass_matrix Stage Dump

날짜: 2026-05-13
브랜치: `20260512_diff_rigid_demo`
참고 commit: `a01911e8` (6 sub-block split)

---

## 진행 흐름 (Step 1-5)

| Step | 결과 |
|---|---|
| 1 Detect | J4 N=2 max rel = 6.97 (worst topology) |
| 2 Localize | step t=0 BW substep 의 root_y/z/wx/wy/arm_rev catastrophic |
| 3 N=1 OK + N≥2 fail | cross-substep state propagation |
| 4 chain dump | cinr_pos.grad 가 load-bearing wrong (fine-zero 분석) |
| 5 sub-1 split fwd_dyn | 효과 없음 (functional only) |
| 5 sub-2 manual update_force_bw | 효과 없음 (Quadrants auto-AD 와 동일) |
| 5 sub-3 skip compute_mass_matrix.grad | **-50% (6.97 → 3.48)**: compute_mass_matrix 가 wrong source 의 절반 |
| 5 sub-4 6 sub-block split | functional OK, sub-block ablation 모두 동일 (sequential chain) |
| 5 sub-5 stage dump | 어느 sub-block 이 wrong contribution 만드는지 dump |

---

## Stage Dump 결과 (2nd BW substep, seed=1000)

Sub-kernel.grad 호출 직후 chain 상태 (`/tmp/diag_mm_stage_dump.py`):

| Stage | mass_mat | f_ang | f_vel | crb_pos | crb_I | cinr_pos | cdof_vel | cdof_ang |
|---|---|---|---|---|---|---|---|---|
| pre (IFT seed) | 2.87e-4 | 0 | 0 | 0 | 0 | 7.45e-8 | 1.31e-8 | 2.21e-9 |
| post impint_corr | same | 0 | 0 | 0 | 0 | same | same | same |
| post armature | same | 0 | 0 | 0 | 0 | same | same | same |
| **post assemble** | 0 | 8.1e-8 | 2.87e-4 | 0 | 0 | same | **2.69e-5** | **3.15e-5** |
| **post compute_f** | 0 | 0 | 0 | **1.39e-4** | 8.1e-8 | same | **4.30e-4** (16×↑) | 3.15e-5 |
| post crb_aggregate | 0 | 0 | 0 | same | 7.35e-8 | same | same | same |
| post crb_initialize | 0 | 0 | 0 | 0 | 0 | **1.39e-4** | same | same |

**관찰**:
- assemble 이 cdof_vel/ang.grad 의 initial contribution 만듦 (2.69e-5 / 3.15e-5)
- compute_f 가 cdof_vel.grad 를 **16배 증폭** (2.69e-5 → 4.30e-4) via inertial_mul reverse
- crb_aggregate / crb_initialize 는 simple chain (tree add reverse + copy reverse)
- cinr_pos.grad 의 *기존 값* 7.45e-8 (pre-mm-reverse 시점) — fwd_dyn 의 *다른* sub-func (update_force) 가 만든 contribution
- cinr_pos.grad 의 *최종 값* 1.39e-4 — compute_mass_matrix 가 추가한 contribution

---

## 결정적 의문

inertial_mul standalone 검증 OK (`/tmp/diag_inertial_mul_standalone.py`):
- standalone Quadrants kernel = numpy chain rule (FP64 floor)
- → compute_f.grad 의 chain rule (= inertial_mul reverse) **자체는 정확**

따라서 wrong source 후보:
1. **`assemble.grad`** 의 chain rule 이 wrong (`mass_mat → f, cdof` 의 reverse 부정확)
2. **`manual_compute_qacc_bw`** 의 IFT seed mass_mat.grad 가 wrong (cross-substep state 영향)
3. **Forward primal staleness** — compute_f 가 사용하는 crb_pos / cdof_vel 등이 *step t=0 의 시점*이 아닌 *step t=1 시점*의 값

---

## 다음 액션 (Step 5 sub-6)

자율 진행 옵션:

### Option A: `mm_assemble.grad` numpy chain rule 검증
- mass_mat.grad → f_*.grad + cdof_*.grad 의 reverse 식 작성
- 직접: f.dot(cdof) 의 chain rule + symmetric copy reverse
- pre + post snapshot 비교 (fvol.grad 와 같은 패턴)

### Option B: `manual_compute_qacc_bw` 의 mass_mat.grad seed 검증
- IFT 식: mass_mat.grad = -force_contrib ⊗ acc_smooth (또는 비슷)
- 2nd BW substep 시점의 input (acc.grad, force, qpos 등) 으로부터 numpy 로 mass_mat.grad 재계산
- kernel output 과 비교 → IFT 자체의 cross-substep 정확성 검증

### Option C: Forward primal staleness 검증
- 2nd BW substep 시점의 crb_pos / cdof_vel 의 값 dump
- N=1 의 동일 stage primal 과 비교 (이전 fvol.grad 검증과 같은 패턴 — sub-1e-7 차이로 부차였음, 그러나 compute_f 는 다를 수 있음)

추천 진행: **A → B → C 순서**. assemble.grad 가 chain 의 첫 번째 mass_mat → f chain 시작 stage 이므로 numpy 검증이 가장 적은 작업.

---

## 검증 인프라

| 파일 | 용도 |
|---|---|
| `/tmp/diag_mm_stage_dump.py` | 6 sub-kernel.grad chain stage dump |
| `/tmp/diag_mm_subblock_skip.py` | sub-block ablation (sequential 확인) |
| `/tmp/diag_inertial_mul_standalone.py` | inertial_mul standalone Quadrants vs numpy |
| `/tmp/diag_fine_zero.py` | cross-substep field 별 zero out |
| `notes/diffrigid_handoff_j4_n2_fwd_velocity_suspect.md` | 이전 단계 핸드오프 |

---

## 가이드 self-check
- ✅ Step 1-3
- ✅ Step 4: chain dump 완료. exit 한 줄 답:
  > **`func_compute_mass_matrix.grad` 가 cinr_pos.grad 의 wrong contribution 50% 를 만든다 (compute_mass_matrix 의 어느 sub-block 인지 sub-stage dump 진행 중)**
- 🚧 Step 5: split + manual update_force + 6-sub-block split 완료. manual replace 효과 확정 안 됨 (sub-block sequential chain). 다음: assemble.grad numpy 검증 또는 IFT seed 검증.
