# Quadrants AD 의 cross-iter same-buffer 한계 — 쉬운 설명

`tests/test_quadrants_self_ref_ad.py` 에 박제된 두 패턴이 왜 한쪽은 통과하고 한쪽은 실패하는지를 정리. J4/J5 (`tests/test_diff_forward_kinematics.py`) 가 깨지는 근본 원인이 이 한계임.

---

## 1. 두 패턴

### Pattern A — Step 2 (PASS)
```python
for i in range(n):
    out_bw[0, i] = vec[i]              # loop 1: out_bw[0] 만 쓴다
for i in range(n):
    out_bw[1, i] = out_bw[0, i] * 2.0  # loop 2: out_bw[0] 읽고 out_bw[1] 쓴다
```
**특징**: 읽기와 쓰기가 서로 다른 loop 에 있음. loop 1 이 완전히 끝난 뒤에 loop 2 가 시작.

### Pattern B — Step 1 (FAIL)
```python
for i_d_ in range(n):
    i_d = n - i_d_ - 1
    out_bw[0, i_d] = vec[i_d]                                           # (a) 자기 슬롯 seed
    for j_d in range(i_d + 1, n):
        out_bw[0, i_d] = out_bw[0, i_d] - L[j_d, i_d] * out_bw[0, j_d]  # (b) 같은 버퍼 다른 슬롯 읽음
```
**특징**: `i_d` 가 descending (예: 3→2→1→0) 으로 돌면서, 매 iter 마다 **이전 iter 에서 채워진 같은 버퍼의 다른 슬롯** (`out_bw[0, j_d]`, `j_d > i_d`) 을 읽음.

---

## 2. 왜 Pattern B 만 실패하나

### Forward operation 의 정확한 reverse adjoint
```
out_bw[0, i_d]_new = out_bw[0, i_d]_old - L[j_d, i_d] · out_bw[0, j_d]
```
이 한 줄의 reverse-mode adjoint 는 두 항이 필요:

| # | adjoint contribution | Quadrants 동작 |
|---|---|---|
| 1 | `grad[out_bw[0, i_d]_old] += grad[out_bw[0, i_d]_new]` | ✅ 잡음 |
| 2 | `grad[out_bw[0, j_d]] += -L[j_d, i_d] · grad[out_bw[0, i_d]_new]` | ❌ 놓침 |

(2) 가 같은 field 의 다른 인덱스 (= 다른 outer iter 의 출력) 로 흘러야 하는 cross-iter adjoint contribution. Quadrants adstack 이 이 path 를 tape 에 박지 않음.

### Pattern A 에서는 왜 문제가 없나
A 는 `out_bw[0]` 의 read 와 write 가 **서로 다른 loop** 에 있어서, 두 loop 사이에 dataflow boundary 가 있음. Quadrants 는 이런 명시적인 read-after-write 는 정상적으로 추적함. 같은 field 의 cross-index 의존성이라도 "loop 가 완전히 끝난 뒤 새 loop" 라면 OK.

B 는 read 와 write 가 **같은 outer loop 의 다른 iter** 끼리 얽혀있음. Quadrants 가 이 indirect dependency 의 reverse path 를 못 만듦.

---

## 3. 숫자로 확인

테스트 셋업:
```
vec = [1, 2, 3, 4]
L 의 strict lower triangle:
  L[1,0]=0.1
  L[2,0]=0.2, L[2,1]=0.3
  L[3,0]=0.4, L[3,1]=0.5, L[3,2]=0.6
seed: grad[out_bw[0, :]] = [1, 1, 1, 1]    (즉 loss = sum(w_i))
```

forward 는 `L^T w = vec` 의 backward substitution → `w = L^(-T) vec`.

수학적으로 정확한 gradient:
```
loss = sum(w) = 1^T · L^(-T) · vec
d(loss)/d(vec) = L^(-1) · [1,1,1,1]^T
```

`L x = [1,1,1,1]` 손풀이:
```
x[0] = 1
x[1] = 1 - 0.1·1                       =  0.9
x[2] = 1 - 0.2·1 - 0.3·0.9             =  0.53
x[3] = 1 - 0.4·1 - 0.5·0.9 - 0.6·0.53  = -0.168
```

| 결과 | vec.grad |
|---|---|
| **FD** (= 정답) | `[1.0,  0.9,  0.53, -0.168]` |
| **Quadrants analytical** | `[1,    1,    1,     1]` |

Quadrants 값을 보면 **모든 L 항의 기여가 0 으로 사라졌음**. 남은 건 `out_bw[0, i_d] = vec[i_d]` 의 trivial chain (derivative = 1) 뿐. → "같은 outer loop 안에서 옆 slot 으로 adjoint 가 전혀 흐르지 않는다" 의 직접 증거.

---

## 4. 이게 왜 LDLT solve 를 깨뜨리나

`func_solve_mass_entity` (`genesis/engine/solvers/rigid/abd/forward_dynamics.py:660`) 는 `M @ x = vec` 를 LDLT (M = L D L^T) 로 푸는데 3 단계:

| Step | 무엇을 푸나 | 패턴 |
|---|---|---|
| 1 | `L^T w = vec` (backward sub) | **Pattern B** — cross-iter same-buffer |
| 2 | `z = D^{-1} w` | Pattern A (slot 0 → slot 1, 분리 loop) |
| 3 | `L x = z` (forward sub) | inner loop 가 `out[j_d]` (다른 버퍼) 읽음 — 안전 |

Step 1 에서 Pattern B 가 발화 → adjoint chain 이 거기서 끊김 → `acc_smooth_bw[0].grad = 0` 으로 굳음 → `force.grad` 까지 못 내려감 → J4 의 child revolute joint qvel gradient (`v.grad[6]`) 가 analytical 0, FD non-zero 로 어긋남.

### J1~J3 가 안 걸린 이유
- **J1** (freejoint, 6 dof): mass matrix 가 거의 block-diagonal. L 의 strict lower triangle 이 trivial → cross-iter 항이 어차피 0.
- **J2/J3** (1 dof): mass matrix 가 1×1 → Step 1 의 inner `for j_d in range(i_d+1, entity_dof_end)` 가 empty range. **Pattern B 가 실행조차 안 됨**.
- **J4** (freejoint + revolute child, 7 dof): root-child mass coupling 으로 L 의 off-diagonal 이 처음으로 non-trivial → Pattern B 가 처음으로 stress test 됨 → 깨짐.

---

## 5. 우회 전략 (Phase B)

가설: **outer loop 를 Python 측 sequential kernel 호출로 펼치면 cross-iter dependency 가 "kernel 간 dependency" 로 변환되어 Quadrants 가 추적 가능해짐**. UCS (`kernel_update_cartesian_space`) 에서 이미 per-link split 으로 동일 우회를 적용한 전례 있음.

구체:
```python
# 현재: 한 kernel 이 outer loop 다 도는 구조
kernel_solve_mass(...)   # 안에서 for i_d_ in range(n_dofs): ...

# 변경: 한 dof 만 처리하는 kernel + Python sequential 호출
for i_d_offset in range(max_n_dofs):       # Python loop
    kernel_solve_mass_step1_one_dof(i_d_offset, ...)
for i_d_offset in range(max_n_dofs):
    kernel_solve_mass_step3_one_dof(i_d_offset, ...)
```

Trade-off:
- ➕ Cross-iter dependency 가 kernel 경계로 옮겨가서 Quadrants 가 정상 backward 함
- ➖ Kernel launch overhead 가 `n_dofs` 배. 일반 로봇 (dof < 30) 은 감당 가능하지만 perf 측정 필요

---

## 6. 회귀 가드

`tests/test_quadrants_self_ref_ad.py::test_quadrants_cross_iter_same_buffer_ad` 는 `xfail(strict=True)` 로 박제됨. Quadrants AD 자체가 고쳐지거나 우회 전략이 진화해서 이 테스트가 통과하면 → **XPASS → CI 실패** → "xfail 마커 떼고 cleanup 해라" 의 자동 알람 역할.

---

## 7. 실제 적용 결과 (Phase B, 2026-05-11)

위 우회 전략을 적용한 결과 J4 는 통과했지만 두 가지 **추가** 문제를 더 발견했고, 이 셋이 모두 풀려야 J1~J5 가 통과함.

### 7.1 (예상치 못한) Step 2 reverse 가 발화 안 함

per-DOF kernel split 만으로는 부족. 진단 결과:
- `kernel_compute_qacc.grad` 가 `acc.grad → acc_smooth_bw[1].grad` 까지는 정상 propagate (Step 3 reverse 동작)
- 그러나 **trivial Step 2 reverse `grad[acc_smooth_bw[0, i_d]] += D_inv[i_d] * grad[acc_smooth_bw[1, i_d]]` 를 silently drop**
- 원본 Step 1 BW (broken cross-iter) 가 살아있어도 같은 양상 → Step 1 skip 이 원인 아님

격리 reverse-mode 의 같은 패턴 (`out[1] = out[0] * c`) 은 정상 작동 (`test_quadrants_two_slot_self_ref_ad`). 즉 `kernel_compute_qacc.grad` 의 tape 구조 자체에 뭔가 문제가 있음. 의심점:
- 중첩된 `@qd.func` 경계 (`func_compute_qacc → func_solve_mass → func_solve_mass_entity`)
- `func_solve_mass_entity` 의 `if rigid_global_info.mass_mat_mask[i_e, i_b]` 동적 분기
- `func_compute_qacc` 안에 top-level `ndrange` 가 두 개 있는 구조

**해결**: Step 2 reverse 를 외부에서 manual kernel 로 작성 (`kernel_solve_mass_step2_reverse_bw`). `grad[acc_smooth_bw[0]] += D_inv * grad[acc_smooth_bw[1]]` 한 줄짜리.

### 7.2 per-link split bound `_MAX_LINKS = 2` 가 J5 (3 links) 에 부족

기존 작업에서 `kernel_forward_velocity_one_link` / `kernel_update_cartesian_space_one_link` 의 Python loop bound 가 `_MAX_LINKS = 2` 하드코딩이었음. J4 (2 links) 에는 충분했지만 J5 (chain3, 3 links) 에서 마지막 링크의 backward 가 누락.

**해결**: build time 에 `self._max_n_links_across_entities = max(e.n_links for e in self.entities)` 계산해서 동적 bound 로 교체.

### 7.3 Backward 전체 ordering (현재 코드)

`substep_pre_coupling_grad` 의 BW path:
```python
# ... constraint solver / FV / UCS backward (per-link split) ...

# Phase B core (LDLT backward):
for k in range(max_n_dofs):
    kernel_solve_mass_step1_one_dof_bw(k)          # forward primal cache
kernel_compute_qacc.grad(...)                       # acc.grad → acc_smooth_bw[1].grad
kernel_solve_mass_step2_reverse_bw(...)             # acc_smooth_bw[1].grad → acc_smooth_bw[0].grad
for k in reversed(range(max_n_dofs)):
    kernel_solve_mass_step1_one_dof_bw.grad(k)     # acc_smooth_bw[0].grad → force.grad
```

### 7.4 결과
- `tests/test_diff_forward_kinematics.py` — J1~J5 모두 통과 (11 tests, fp32/fp64 × single/batched matrix)
- `tests/test_quadrants_self_ref_ad.py` — 의도된 xfail 유지 (Quadrants AD 본체 한계)
