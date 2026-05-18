# Diffrigid J4 N=2 — 검증 완료 후보 종합

날짜: 2026-05-13
커밋: `10694069`

---

## 검증된 모든 chain rule + primal + FD 정확. 그러나 ana 여전히 wrong.

| 검증 | 결과 |
|---|---|
| **#1 IFT seed (manual_compute_qacc_bw, current substep)** | ✅ numpy vs kernel FP64 floor |
| **#2 Forward primal at qacc_bw call** (N=1 vs N=2) | ✅ mass_mat / L / D_inv 일치 |
| **#3a step_2.grad chain rule** (hand-derived) | ✅ vel.grad / acc.grad / qpos.grad 식 일치 |
| **#3b COM_links.grad skip** | -50% (compute_mm.grad skip 과 같은 chain 끊김) |
| **#4 FD 자체** (Richardson eps sweep) | ✅ eps=1e-3 ~ 1e-6 모두 5+ 자리 일치 |
| inertial_mul standalone | ✅ numpy vs Quadrants AD 일치 |
| update_force manual vs auto-AD | ✅ 동일 ana |
| mm_assemble manual vs auto-AD | ✅ 동일 ana |
| mm_crb_aggregate manual vs auto-AD | ✅ 동일 ana |
| fvol.grad numpy verification (이전 단계) | ✅ FP64 floor 일치 |

---

## ana[t=0] vs FD[t=0] (J4 N=2 seed=1000)

| DOF | ana | FD | 비율 |
|---|---|---|---|
| root_x | 5.33e-5 | 5.33e-5 | ✅ 1.0 |
| root_y | -9.53e-9 | -2.06e-8 | **0.46x** wrong |
| root_z | 6.52e-8 | 4.88e-8 | **1.33x** wrong |
| root_wx | -3.2e-11 | -5.9e-12 | (FP64 floor) |
| root_wy | -1.37e-8 | -1.71e-9 | **8x** wrong |
| root_wz | 8.87e-9 | 8.88e-9 | ✅ 1.0 |
| arm_rev | 1.50e-8 | 1.80e-8 | **0.83x** wrong |

---

## 핵심 모순

- 모든 검증된 chain rule 식 정확.
- Forward primal 정확.
- FD 정확.
- ⟹ ana 도 정확해야. 그러나 wrong by 0.46~8x.

## 남은 가능성 (모두 검증 필요)

1. **Manual chain rule 식 자체가 wrong + Quadrants AD 도 같은 오류식** — numpy 와 Quadrants AD 양쪽이 같은 wrong 식 적용. 검증: PyTorch autograd 같은 완전히 다른 reference 와 비교.

2. **`motion_cross_force` standalone 검증 안 됨** — inertial_mul 만 검증. 단 standalone 정확이라도 (1) 의 모순 해소 못 함.

3. **Cross-substep state 의 *시점별* primal staleness** — N=1 vs N=2 비교는 한 시점만. BW chain 안의 *여러 stage* 에서 primal 이 서로 다른 timestamp 가질 수 있음.

4. **prev BW substep 의 chain output 정확성** — chain rule 정확 + input 정확 = output 정확. 그러나 ana[t=1] 정확이 *모든 output 정확*을 보장 안 함 (ctrl_force.grad path 만 보장).

## 다음 진단 후보 (자율 진행 시)

### Option E: PyTorch autograd reference 비교
- 같은 J4 step t=0 forward 식을 PyTorch 로 작성 + autograd → reference
- ana[t=0] vs PyTorch reference vs FD 의 3-way 비교
- Manual 식 자체의 정확성 검증

### Option F: motion_cross_force standalone
- inertial_mul 와 같은 패턴
- 작업 작음

### Option G: 시점별 primal staleness deep dive
- BW chain 안의 각 sub-kernel.grad 시점의 primal 캡처
- 다른 시점에서 다른 timestamp 의 값 사용 여부 확인

### Option H: 1st BW substep 의 cross-substep .grad output 검증
- ana[t=1] 정확 ≠ 1st substep 의 *모든 output* 정확
- chain rule 정확하면 output 도 정확이지만, 만약 chain rule 자체가 wrong (= Option 1) 이면 output wrong

---

## 검증 인프라

| 파일 | 용도 |
|---|---|
| `/tmp/diag_ift_seed_verify.py` | IFT seed numpy vs kernel |
| `/tmp/diag_qacc_bw_primal_n1_vs_n2.py` | Forward primal N=1 vs N=2 |
| `/tmp/diag_com_links_skip.py` | COM_links.grad skip ablation |
| `/tmp/diag_richardson_fd.py` | Richardson FD eps sweep |
| `/tmp/diag_inertial_mul_standalone.py` | inertial_mul standalone |
| `/tmp/diag_fine_zero.py` | cross-substep field zero out |
