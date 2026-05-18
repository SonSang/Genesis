# Diffrigid J4 N=2 — 후보 #1, #2 검증 결과

날짜: 2026-05-13
커밋: `10694069` (manual mm_assemble + mm_crb_aggregate)

---

## Step 5 종합 검증 결과

### 5개 sub-func chain rule 정확성 (모두 정확)

| Sub-func | 방법 | 결과 |
|---|---|---|
| `inertial_mul` | standalone Quadrants vs numpy | max diff 2e-25 (FP64 floor) ✅ |
| `update_force` | manual_update_force_bw vs auto-AD | identical ana[t=0] ✅ |
| `mm_assemble` | manual_mm_assemble_bw vs auto-AD | identical ana[t=0] ✅ |
| `mm_crb_aggregate` | manual_mm_crb_aggregate_bw vs auto-AD | identical ana[t=0] ✅ |
| `kernel_manual_compute_qacc_bw` (IFT) | numpy IFT vs manual code | max diff 1e-22 ✅ |

### 후보 #1 (IFT seed)
- 검증: 2nd BW substep 의 manual_compute_qacc_bw 의 input + post 캡처 후 numpy IFT 식 재계산
- 결과: `force.grad max diff 1.3e-23`, `mass_mat.grad max diff 1.1e-22` ✅ FP64 floor
- → **IFT 식 자체는 정확**.

### 후보 #2 (Forward primal staleness)
- 검증: N=1 vs N=2 (2nd BW substep) 의 manual_compute_qacc_bw 직전 forward primal 비교
- 결과:
  | Field | |Δ| | rel |
  |---|---|---|
  | mass_mat, mass_mat_L, mass_mat_D_inv | 0 | 0 ✅ |
  | qpos, vel | 0 | 0 ✅ |
  | acc_smooth, acc, force | sub-1e-7 | sub-1e-7 |
- LDLT 의 forward primal (mass_mat, mass_mat_L, mass_mat_D_inv) 정확히 일치 (staleness 없음).
- acc_smooth / acc / force 작은 차이 — ana[t=0] 700% 변동 설명 불가.
- → **Forward primal staleness 도 wrong source 아님**.

---

## 후보 #3 진입 — `step_2.grad` output `acc.grad`

manual_compute_qacc_bw 의 input `acc.grad` (pre-qacc_bw 시점 max = 7.999e-05) 가 step_2.grad 의 output. 이게 정확하지 않다면 IFT seed wrong → mass_mat.grad wrong → downstream chain (assemble, compute_f) 모두 wrong (chain rule 자체는 정확하므로 wrong input → wrong output).

step_2 의 forward 구조:
- `update_acc` (auto-AD)
- `implicit_damping` (auto-AD)
- `func_integrate` (이미 manual via Option B, `kernel_manual_step_2_integrate_bw`)
- 그 외 kernel_step_2.grad 안의 다른 부분 (auto-AD)

reverse 의 chain (step_2.grad 의 input qpos.grad/vel.grad → output acc.grad):
- vel = vel_old + acc * dt → reverse: acc.grad += dt * vel.grad
- 이게 `func_integrate` 의 reverse (manual_step_2_integrate_bw 가 처리)

검증 옵션:
A. `kernel_manual_step_2_integrate_bw` 자체의 정확성 검증 (numpy vs kernel)
B. `kernel_step_2.grad` 의 나머지 (update_acc, implicit_damping 등) 검증
C. stage dump 의 stage 14 (post-begin_bw_sub) → stage 15 (post-step_2.grad) 의 acc.grad 변화 검증
D. step_2 의 input 인 `qpos.grad`, `vel_next.grad` 의 정확성 검증

가장 빠른 단축: **D — step_2.grad input 의 정확성 검증**. step_2.grad 가 정확한 chain rule 을 적용하더라도 wrong input → wrong output.

stage 14 의 vel_next.grad / qpos.grad (= begin_bw_sub 후의 swap 결과). 이게 wrong 인지.

vel_next.grad 의 source = 이전 stage 의 vel.grad (begin_bw_sub 가 swap). 그 vel.grad 는 stage 11 의 fvol.grad output (검증된 정확).

근데 stage 11 의 fvol.grad output 검증은 *그 chain rule* 의 정확성 — input cd_*/cdofd_*/cdof_*.grad 이 정확하다고 *가정* 했을 때. 만약 input 자체가 wrong 이면 fvol.grad 의 output 도 wrong.

fvol.grad input 의 source = 이전 BW substep 의 fwd_dyn.grad output. 이미 sub-block 검증에서 chain rule 자체 정확 확정. 따라서 fwd_dyn.grad 의 output cd_*/cdof_*/cdofd_*.grad 가 정확하다고 가정.

근데 그 chain 의 정확성도 검증 안 됨. 다음 가설 chain:

prev BW substep:
1. mass_mat.grad seed (IFT, 정확) → assemble.grad (정확) → compute_f.grad (정확) → cdof_*.grad, crb_*.grad
2. crb_aggregate.grad (정확) → cinr_*.grad
3. crb_initialize.grad (단순 copy reverse) → cinr_*.grad 누적

prev BW 의 mass_mat.grad seed 가 정확하다면 모든 downstream 도 정확. 즉 fwd_dyn.grad 의 output 도 정확.

prev BW 의 mass_mat.grad seed 의 정확성? = prev BW 의 acc.grad / acc_smooth.grad input 의 정확성. prev BW 는 1st substep (step t=1 의 reverse). acc.grad input = step_2.grad output. ana[t=1] 정확하므로 1st substep 의 chain 정확. → 1st substep 의 acc.grad / mass_mat.grad / cd_*/cdof_*/cdofd_*.grad 모두 정확.

그 출력 (cross-substep state) 이 2nd substep 의 input 으로 운반 → 2nd substep 의 input 정확.

근데 2nd substep 에서 ana[t=0] wrong. 그러면 2nd substep 의 chain rule 또는 forward primal 어딘가 wrong.

이미 검증된 부분: chain rule 정확, forward primal 거의 정확.

미검증: kernel_step_2.grad 의 chain rule 정확성. *2nd substep 의* step_2.grad 가 wrong acc.grad 생성 가능성.

근데 1st substep 에서도 같은 step_2.grad. 차이 = 1st 와 2nd 의 input 이 다름. step_2 의 chain rule 자체는 deterministic.

step_2.grad 의 chain rule 이 *모든 input 에서 정확* 이라면 ana[t=0] = fd. wrong → chain rule wrong 또는 forward primal wrong.

이건 후보 #3 의 진입점. 큰 작업.

---

## 다음 액션 (Step 5 sub-7)

자율 진행 다음 단계: step_2.grad 의 chain rule 검증.

1. **stage dump 확장**: stage 14 → stage 15 의 acc.grad 생성 stage 의 dump
2. **manual_step_2_integrate_bw 의 standalone 검증**: numpy chain rule 작성 + 비교
3. **kernel_step_2.grad 의 나머지 (auto-AD)**: 어느 부분이 acc.grad chain 에 contribute?

step_2 의 구조 파악 필요. 큰 작업이지만 후보 #3 의 핵심.
