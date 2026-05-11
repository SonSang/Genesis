# Differentiable Rigid Simulation — Work-in-Progress Handoff

작성일: 2026-05-11  
브랜치: `20260503_diff_rigid_demo`  
관련 stash 커밋: `94da82fa stash`

다른 스테이션의 에이전트가 이 작업을 이어받기 위한 인수인계 문서. 목적은 Genesis 의 rigid body backward pass 를 일반 multi-link MJCF 시나리오에서 FD 와 일치하도록 만들어 cartpole / pusher / ant / walker 같은 RL diff 데모를 구동 가능하게 하는 것.

---

## 1. 큰 그림

### 목표
`tests/test_diff_forward_kinematics.py` 의 5 개 토폴로지 (J1~J5) 가 모두 FD 와 일치하는 backward gradient 를 내도록 만든다. 이 5 개는 점진적 stress test 로 설계됨:

| 토폴로지 | MJCF | dof | 특징 |
| --- | --- | --- | --- |
| J1 | `MJCF_FREE` | 6 | freejoint only (1 link) |
| J2 | `MJCF_REVOLUTE` | 1 | fixed-base + revolute (1 link) |
| J3 | `MJCF_PRISMATIC` | 1 | fixed-base + prismatic (1 link) |
| J4 | `MJCF_FREE_REV` | 7 | freejoint + revolute child (2 links) |
| J5 | `MJCF_REV_CHAIN3` | 3 | revolute chain 3 links |

### 현재 통과 상태
- **J1~J5 모두 통과** (Phase B 완료, 2026-05-11). fp32/fp64 × single/batched 11 tests.

### 이미 진행된 fix (커밋된 것 — `git log`)
1. `1bef38ba improve qd_rotvec_to_quat for stable backward pass` — `genesis/utils/geom.py` 의 `qd_rotvec_to_quat` 를 branch-free `sqrt(thetasq + eps**2)` 형태로 재작성. J2 catastrophic chain break 의 직접 원인이었음.
2. `ca45a64f now pass J1 + J2, using fp32,64 and single/multi batch env` — 테스트 파라미터화 (precision × n_envs 매트릭스 확장), J1/J2 통과.
3. `81acb7d5 fix backward pass bug for prismatic joint` — `func_forward_kinematics_entity` 의 PRISMATIC branch 에서 `quat_bw` cache slot 을 채우지 않아 J3 가 NaN 났던 것 수정. `quat = W(links_state.quat_bw, next_I, quat, BW)` 한 줄 추가.

### Phase B (이번 세션, 미커밋 — working tree)
J4/J5 통과를 위해 `kernel_compute_qacc.grad` 의 두 가지 silent failure 를 외부 orchestration 으로 우회:
- per-DOF Step 1 BW kernel split (`kernel_solve_mass_step1_one_dof_bw`)
- manual Step 2 reverse kernel (`kernel_solve_mass_step2_reverse_bw`)
- per-link split bound `_MAX_LINKS` 를 build-time `self._max_n_links_across_entities` 로 일반화 (J5 3-link chain 대응)

자세한 내용은 `notes/quadrants_cross_iter_ad_limitation.md` §7 참조.

---

## 2. 현재 막힌 지점 — `func_solve_mass_entity` backward 깨짐

J4 backward 에서 `v.grad[6]` (child revolute qvel) 가 analytical=0 인데 FD=+5.94e-5 (안정). `v.grad[3]` (freejoint qvel x) 도 analytical=-3.77e-5 vs FD=-2.56e-5 로 1.5× 어긋남.

`GENESIS_DEBUG_GRAD=1 python /tmp/probe_j4_grad_trace.py` (§4 의 probe) 로 단계별 grad 트레이스하면 다음 지점에서 끊김:

```
after step_2.grad:          acc.grad = 5.67e-4,  acc_smooth_bw.grad = 0,        force.grad = 0
after compute_qacc.grad:    acc.grad = 0,        acc_smooth_bw[1].grad = 4.97e-4, acc_smooth_bw[0].grad = 0, force.grad = 0   ← 단절
after fwd_dynamics_without_qacc.grad:  같음 — cd_vel/cdd_vel/cfrc 전부 0
```

→ **`kernel_compute_qacc.grad` 가 `acc_smooth_bw[1].grad` 를 `acc_smooth_bw[0].grad` 로 전달하지 않음.** 즉 LDLT solve 의 backward chain 이 Step 3 까지만 흐르고 Step 2/1 에서 멈춤.

### 원인 — Quadrants AD 의 cross-iter same-buffer dependency 한계

`genesis/engine/solvers/rigid/abd/forward_dynamics.py:660-714 func_solve_mass_entity` 의 BW-mode forward 패턴:

```python
# Step 1: Solve w st. L^T @ w = y   (line 678-696)
for i_d_ in range(n_dofs):
    i_d = entity_dof_end - i_d_ - 1
    ...
    if qd.static(BW):
        out_bw[0, i_d, i_b] = vec[i_d, i_b]
    for j_d in range(i_d + 1, entity_dof_end):
        if qd.static(BW):
            out_bw[0, i_d, i_b] = (
                out_bw[0, i_d, i_b] - L[j_d, i_d] * out_bw[0, j_d, i_b]   # ← cross-iter read
            )

# Step 2: z = D^{-1} w   (line 698-703)
for i_d in range(entity_dof_start, entity_dof_end):
    if qd.static(BW):
        out_bw[1, i_d, i_b] = out_bw[0, i_d, i_b] * D_inv[i_d, i_b]

# Step 3: Solve x st. L @ x = z   (line 705-714)
for i_d in range(entity_dof_start, entity_dof_end):
    ...
    if qd.static(BW):
        curr_out = out_bw[1, i_d, i_b]
    for j_d in range(entity_dof_start, i_d):
        curr_out = curr_out - L[i_d, j_d] * out[j_d, i_b]   # ← cross-iter read
    out[i_d, i_b] = curr_out
```

격리 결과 (§4 의 `probe_quadrants_step1_pattern.py`): 단순 self-ref (`buf[1,i] = buf[0,i] * c`) 는 작동하지만, **outer loop 가 같은 buffer 의 cross-iter 위치를 read 하는 패턴은 Quadrants AD 가 chain rule 을 누락함**. seed grad 가 모든 vec[i] 에 그대로 transfer 되고 (`vec.grad = [1, 1, 1, 1]`) 그 뒤의 cross-iter chain (`vec.grad = [1.0, 0.9, 0.53, -0.168]`) 이 빠짐.

이건 `update_cartesian_space.grad` 에서 cross-link adjoint attenuation 으로 봤던 것과 **정확히 동일한 Quadrants AD 한계**. 거기서는 per-link kernel split 으로 우회했고 부분 복구 됨 (qpos.grad 살아남, vel.grad 일부 남음).

### J1~J3 가 안 걸린 이유
- J1 (freejoint, 6 dof): mass matrix 가 essentially block-diagonal, off-diagonal L 항이 trivial.
- J2/J3 (1 dof): mass matrix 가 1×1 → LDLT solve 가 단순 division, cross-iter loop body 없음.
- **J4 (7 dof, freejoint + revolute child)**: mass matrix 가 처음으로 non-diagonal 이라 L 의 off-diagonal 항이 활성화. LDLT solve 의 cross-iter chain 이 처음으로 stress test 됨 → 거기서 깨짐.

---

## 3. Stash 커밋 `94da82fa` 의 변경 사항

이 커밋에 들어있는 변경을 카테고리별로 정리. 각 항목은 "검증된 효과" 와 "유지/되돌리기 판단" 을 명시.

### 3.1 `genesis/engine/solvers/rigid/abd/forward_kinematics.py`

#### (a) `func_forward_kinematics_entity` — prismatic / fixed joint cache write 보완
PRISMATIC 과 FIXED branch 에 `quat = W(links_state.quat_bw, next_I, quat, BW)` 와 `pos = W(...)` 추가. J3 의 NaN 원인이었던 cache slot 누락 패치. **유지.**

#### (b) `func_forward_kinematics_entity` line 595 부근 — free joint quat normalize 주석 처리
`quat_ = quat_ / quat_.norm()` 비활성화. backward 에서 tangent-space projection 으로 인한 grad signal loss 방지 시도. **효과 확인됨, 유지.**

#### (c) `func_forward_kinematics_entity_one_link` (line 661 부근, new) + `kernel_update_cartesian_space_one_link` (line 817 부근, new)
single-link 단위로 outer link loop 의 한 iteration 을 실행하는 함수/커널. cross-link adjoint attenuation 우회용 split-kernel. **효과 확인됨 (J4 qpos.grad 복구), 유지.**

#### (d) `func_forward_velocity_entity_one_link` + `kernel_forward_velocity_one_link` (new)
같은 패턴의 split-kernel 을 `forward_velocity` 에도 적용. **효과 없음 — Coriolis chain 의 단절은 forward_velocity 가 아니라 더 깊은 LDLT solve 에 있었음.** 정리할 때 제거 또는 향후 분리 PR 로 옮길 가능성 있음. 현재는 유지 (회귀 방어 역할).

### 3.2 `genesis/engine/solvers/rigid/abd/diff.py`

#### `kernel_copy_next_to_curr_no_check` (new, line 350 부근)
`qpos := qpos_next`, `vel := vel_next` 를 backward substep 진입 직전에 unguarded 로 복사. `kernel_prepare_backward_substep` 이 pre-integrate state 로 되돌려놓는 게 post-integrate FK backward 를 망가뜨려서, 이걸 다시 post-integrate 로 끌어올리는 용도. J4 freejoint quat w-component grad 복구의 핵심. **유지.**

### 3.3 `genesis/engine/solvers/rigid/rigid_solver.py`

#### (a) `_debug_grad_dump(tag)` (line 1255)
`GENESIS_DEBUG_GRAD=1` 환경 변수로 켜는 grad dump 헬퍼. level=2 로는 verbose 전체 벡터 출력. dumping fields:  
`rigid_global_info.qpos, dofs_state.{vel,pos,acc,acc_smooth,acc_smooth_bw,force,qf_bias,qf_smooth,qf_passive,qf_applied,cdofd_vel,cdofd_ang}, links_state.{pos,quat,pos_bw,quat_bw,cd_vel,cd_ang,cdd_vel,cdd_ang,cfrc_vel,cfrc_ang}, joints_state.{xanchor,xaxis}`.  
**유지** — 디버깅의 1차 도구. `MEMORY.md` 의 [Backward-pass grad trace methodology](../home/sanghyun/.claude/projects/-home-sanghyun-Projects-Genesis/memory/feedback_backward_grad_trace.md) 메모와 연계됨.

#### (b) `substep_pre_coupling_grad` (line 1318) — `_MAX_LINKS = 2` hard-coded split-kernel sequences
- post-integrate `kernel_forward_velocity.grad` → split-per-link 두 번
- post-integrate `kernel_update_cartesian_space.grad` → split-per-link 두 번
- initial-substep `if cur_substep_global == 0` 블록도 같은 패턴으로 split
- 단계별 `_debug_grad_dump` 호출 5 개 (entry / post-fwd_velocity / post-UCS / begin_BW / step_2 / compute_qacc / fwd_dynamics_without_qacc / initial-end)

**부분 유지** — UCS split 은 J4 qpos.grad 살리는 핵심이라 유지. FV split 은 효과 없음 (위 §3.1.d 참조). `_MAX_LINKS = 2` 는 J4 전용 하드코딩이라 **J5 (3 links) 로 가려면 entity 별 max n_links 로 일반화 필요**.

### 3.4 `genesis/utils/geom.py`
- `qd_transform_quat_by_quat`: `.normalized()` 제거 (backward tangent-space projection 방지)
- `qd_transform_by_quat`: `/ (q_ww + q_xx + q_yy + q_zz)` 제거 (backward signal 보존)

둘 다 **유지** — J4 qpos.grad 복구의 일부.

---

## 4. 재현 가이드

### 4.1 환경
```bash
conda activate genesis   # /home/sanghyun/miniconda3/envs/genesis
# torch 2.11.0+cu130, genesis 0.4.6, quadrants 0.7.7
```

⚠️ `uv run python ...` 절대 사용 금지 — `uv run` 이 venv 를 재구성하면서 torch 를 날림. 항상 conda env 의 `python` 직접 호출.

### 4.2 메인 회귀 테스트
```bash
python -m pytest tests/test_diff_forward_kinematics.py -v -x -p no:xdist
# J1, J2, J3 PASS / J4, J5 FAIL (현재 v.grad chain 단절)
```

### 4.3 J4 grad chain trace (가장 핵심)
재현 스크립트는 `/tmp/probe_j4_grad_trace.py` 에 있음 (사라졌으면 §6 의 코드로 재생성). 실행:

```bash
GENESIS_DEBUG_GRAD=1 python /tmp/probe_j4_grad_trace.py 2>&1 | grep -E "GRAD|v.grad"
GENESIS_DEBUG_GRAD=2 python /tmp/probe_j4_grad_trace.py 2>&1 | grep "after compute_qacc"   # slot-level 보기
```

기대 출력 (현재 상태):
```
v.grad = [+3.752e-02, +5.667e-02, -3.405e-02, -3.773e-05, +2.598e-03, +7.180e-03, 0.000e+00]
FD     = [+3.752e-02, +5.667e-02, -3.405e-02, -2.56e-05,  +2.619e-03, +7.055e-03, +5.937e-05]
                                              ↑ 1.5× 어긋남                       ↑ analytical 0
```

`after compute_qacc.grad` 에서 `dofs_state.acc_smooth_bw.grad` 가 14 entries (2 슬롯 × 7 dof) 인데, slot [0] 은 전부 0 / slot [1] 만 non-zero → Step 2 backward 가 발화 안 함 (= cross-iter chain 단절).

### 4.4 FD 안정성 검증
v.grad[6] 의 FD 값이 실제 물리적 signal 임을 확인 (noise 가 아님):
```bash
python /tmp/probe_fd_eps_v6.py   # eps sweep 1e-3..1e-7
```
출력에서 모든 eps 에서 FD = +5.937e-5 로 일치하면 OK. O(eps²) scaling 도 깔끔.

### 4.5 Quadrants AD 한계 격리 probe
```bash
python /tmp/probe_quadrants_self_ref.py        # simple self-ref → 작동 ✓
python /tmp/probe_quadrants_step1_pattern.py   # Step 1 cross-iter 패턴 → 실패 ✗
python /tmp/probe_quadrants_step1_fix.py       # local accumulator 시도 → 같은 실패
python /tmp/probe_quadrants_step1_fix2.py      # 별도 buffer 분리 → 같은 실패
```

이 4 개의 probe 가 "원인은 cross-iter dependency 자체이고 multi-write 나 same-tensor 가 아니다" 를 입증함. 코드는 §6 에 보관됨.

---

## 5. 다음 단계 — 권장 순서

### Phase A — 격리 회귀 테스트 (먼저 할 일)
1. `/tmp/probe_quadrants_step1_pattern.py` 를 `tests/test_quadrants_self_ref_ad.py` 로 옮겨서 **현재 실패하는 Quadrants AD 한계를 영구 회귀 테스트로 박제**. `pytest.xfail` 마커 + 명확한 docstring 으로 "이게 통과하면 LDLT solve fix 가 가능해진다" 를 표시.
2. J4 의 `dofs_state.force.grad` vs FD 비교하는 isolated unit test 를 `tests/test_diff_solve_mass.py` 에 추가. 현재 FAIL. 비슷하게 xfail 처리.

### Phase B — LDLT solve backward fix
가설: `func_solve_mass_entity` 의 Step 1/Step 3 outer loop 를 **per-dof kernel split** 으로 리팩토링 (per-link split 했던 것과 동일 전략).

구체:
- `func_solve_mass_entity_one_dof(i_e, i_b, i_d_offset, ...)` 추가 — 한 dof 만 처리.
- `kernel_solve_mass_step1_one_dof(i_d_offset, ...)`, `kernel_solve_mass_step3_one_dof(i_d_offset, ...)` 추가.
- `func_compute_qacc` 호출 측에서 entity 별 max n_dofs 만큼 split sequential 호출. Step 1 은 i_d_offset = n-1 → 0 (descending), Step 3 는 0 → n-1 (ascending). backward 도 역순.

리스크: kernel launch overhead 가 n_dofs 배. typical 로봇은 dof < 30 이라 감당 가능하지만 perf 측정 필요.

### Phase C — 일반화 + 정리
1. `_MAX_LINKS = 2` 하드코딩을 entity max n_links 로 자동화 → J5 chain3 (3 links) 통과 확인.
2. `func_forward_velocity_entity_one_link` 가 효과 없음을 확인 후 (Phase B 가 진짜 fix 면 더 안 필요할 수도) **제거 or PR 로 분리**.
3. `_debug_grad_dump` 호출 site 정리. helper 자체는 보존 (env var gating 되어있음).
4. 최종 `tests/test_diff_forward_kinematics.py` 의 J4/J5 case xfail 해제, tolerance 조정.

### Phase D — RL 데모
- cartpole / pusher / ant / walker MJCF 로 `loss.backward()` 가 안정적으로 도는지 확인.
- 이게 PR #2742 후속 작업의 본진. (현재 task #7 "Identify what user needs to take over PR #2742" 가 그 맥락)

---

## 6. 보조 코드 archive

이 섹션의 probe 스크립트들은 `/tmp` 가 휘발성이라서 사라졌을 경우 그대로 복붙해서 재생성할 수 있도록 박제.

### `/tmp/probe_j4_grad_trace.py`
J4 의 forward + backward 를 돌리고 `v.grad` 출력. `GENESIS_DEBUG_GRAD=1` 또는 `=2` 로 단계별 grad 트레이스.

```python
"""J4 (freejoint root + revolute child) grad trace for set_dofs_velocity → links_pos."""
import os
import tempfile

import numpy as np
import torch

import genesis as gs


MJCF_FREE_REV = """
<mujoco model="free_with_child">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      <body name="arm" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def main():
    gs.init(backend=gs.cpu, precision="64", logging_level="info")
    fd_, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd_, "w") as f:
        f.write(MJCF_FREE_REV)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0), requires_grad=True),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False, enable_self_collision=False,
            enable_joint_limit=False, disable_constraint=True,
            use_hibernation=False, use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=path))
    scene.build()

    n_dofs = robot.n_dofs
    print(f"n_dofs={n_dofs}, n_links={robot.n_links}", flush=True)

    rng = np.random.default_rng(72)
    base_v = rng.standard_normal(n_dofs).astype(np.float64)
    rng_t = np.random.default_rng(61)
    target = torch.from_numpy(rng_t.standard_normal((1, robot.n_links, 3)).astype(np.float64)).to(dtype=gs.tc_float, device=gs.device)

    v = gs.tensor(base_v, dtype=gs.tc_float, requires_grad=True)
    scene.reset()
    robot.set_dofs_velocity(v)
    scene.step()
    pos = scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)].links_pos
    print(f"forward links_pos =\n{pos.detach().cpu().numpy()}", flush=True)
    loss = ((pos.reshape(-1) - target.reshape(-1)) ** 2).sum()
    print(f"loss = {loss.item():.6f}", flush=True)
    qpos_post = scene._sim.rigid_solver._rigid_global_info.qpos
    from genesis.utils.misc import qd_to_torch as _qt
    print(f"post-integrate qpos = {_qt(qpos_post, copy=True).cpu().numpy().squeeze()}", flush=True)

    loss.backward()
    print(f"v.grad = {v.grad.cpu().numpy()}", flush=True)


if __name__ == "__main__":
    main()
```

### `/tmp/probe_fd_eps_v6.py`
v.grad[6] (child revolute qvel) 의 FD 값이 eps 스윕에서 안정인지 확인. base_v / target 시드는 위 trace 스크립트와 동일하게 맞춰져있어 직접 비교 가능.

```python
"""Sweep FD eps for v.grad[6] (child revolute joint velocity) to check stability."""
import os
import tempfile

import numpy as np
import torch

import genesis as gs


MJCF_FREE_REV = """
<mujoco model="free_with_child">
  <worldbody>
    <body name="chassis" pos="0 0 0">
      <freejoint/>
      <inertial mass="1.0" pos="0 0 0" diaginertia="0.1 0.1 0.1"/>
      <geom type="box" size="0.1 0.1 0.1" contype="0" conaffinity="0"/>
      <body name="arm" pos="0.2 0 0">
        <joint type="hinge" axis="0 1 0"/>
        <inertial mass="0.5" pos="0.1 0 0" diaginertia="0.01 0.01 0.01"/>
        <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def build():
    fd_, path = tempfile.mkstemp(suffix=".xml")
    with os.fdopen(fd_, "w") as f:
        f.write(MJCF_FREE_REV)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0), requires_grad=False),
        rigid_options=gs.options.RigidOptions(
            enable_collision=False, enable_self_collision=False,
            enable_joint_limit=False, disable_constraint=True,
            use_hibernation=False, use_contact_island=False,
        ),
        show_viewer=False,
    )
    robot = scene.add_entity(gs.morphs.MJCF(file=path))
    scene.build()
    return scene, robot


def main():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")
    scene, robot = build()
    rng = np.random.default_rng(72)
    base_v = rng.standard_normal(robot.n_dofs).astype(np.float64)
    rng_t = np.random.default_rng(61)
    target = torch.from_numpy(rng_t.standard_normal((1, robot.n_links, 3)).astype(np.float64)).to(dtype=gs.tc_float, device=gs.device)

    def loss_at(v_vec):
        scene.reset()
        robot.set_dofs_velocity(gs.tensor(v_vec, dtype=gs.tc_float))
        scene.step()
        pos = scene.get_state().solvers_state[scene.solvers.index(scene.rigid_solver)].links_pos
        return float(((pos.reshape(-1) - target.reshape(-1)) ** 2).sum().detach().cpu())

    print(f"base loss = {loss_at(base_v):.10e}", flush=True)
    for eps in [1e-3, 1e-4, 1e-5, 1e-6, 1e-7]:
        v_p = base_v.copy(); v_p[6] += eps
        v_m = base_v.copy(); v_m[6] -= eps
        Lp = loss_at(v_p)
        Lm = loss_at(v_m)
        slope = (Lp - Lm) / (2 * eps)
        print(f"eps={eps:.0e}: L+={Lp:.12e}  L-={Lm:.12e}  diff={Lp-Lm:+.3e}  FD={slope:+.6e}", flush=True)


if __name__ == "__main__":
    main()
```

### `/tmp/probe_quadrants_step1_pattern.py`
Quadrants AD 의 cross-iter same-buffer dep 한계를 격리하는 최소 reproduction. 결과: FD `[1.0, 0.9, 0.53, -0.168]` vs Quadrants `[1, 1, 1, 1]` (chain rule 누락). `tests/test_quadrants_self_ref_ad.py` 로 이관 권장.

```python
"""Replicate Step 1 pattern: overwrite from vec + cross-iter self-update with same buffer."""
import numpy as np

import genesis as gs
import quadrants as qd


def main():
    gs.init(backend=gs.cpu, precision="64", logging_level="warning")

    n = 4
    vec = qd.field(dtype=gs.qd_float, shape=(n,), needs_grad=True)
    out_bw = qd.field(dtype=gs.qd_float, shape=(2, n), needs_grad=True)
    L = qd.field(dtype=gs.qd_float, shape=(n, n), needs_grad=False)

    @qd.kernel
    def step1():
        for i_d_ in range(n):
            i_d = n - i_d_ - 1
            out_bw[0, i_d] = vec[i_d]
            for j_d in range(i_d + 1, n):
                out_bw[0, i_d] = out_bw[0, i_d] - L[j_d, i_d] * out_bw[0, j_d]

    for i in range(n):
        vec[i] = float(i + 1)
        out_bw[0, i] = 0.0; out_bw[1, i] = 0.0
        for j in range(n):
            L[i, j] = 0.0
    L[1, 0] = 0.1
    L[2, 0] = 0.2; L[2, 1] = 0.3
    L[3, 0] = 0.4; L[3, 1] = 0.5; L[3, 2] = 0.6

    step1()
    w = [float(out_bw[0, i]) for i in range(n)]
    print(f"forward w (= L^-T vec): {w}")

    # FD
    eps = 1e-6
    fd = []
    for k in range(n):
        for sign in (+1, -1):
            for i in range(n):
                vec[i] = float(i + 1) + (sign * eps if i == k else 0)
                out_bw[0, i] = 0.0
            step1()
            if sign == +1:
                wp = sum(float(out_bw[0, i]) for i in range(n))
            else:
                wm = sum(float(out_bw[0, i]) for i in range(n))
        fd.append((wp - wm) / (2 * eps))
    print(f"FD d(sum w)/d(vec[k]) per k: {fd}")

    # Analytical
    for i in range(n):
        vec[i] = float(i + 1)
        out_bw[0, i] = 0.0
        out_bw.grad[0, i] = 1.0
        out_bw.grad[1, i] = 0.0
        vec.grad[i] = 0.0
    step1()
    step1.grad()
    ana = [float(vec.grad[i]) for i in range(n)]
    print(f"Analytical vec.grad      : {ana}")
    print(f"Match FD? {np.allclose(fd, ana, rtol=1e-5)}")
    print(f"After .grad: out_bw.grad[0,:] = {[float(out_bw.grad[0,i]) for i in range(n)]}")
    print(f"After .grad: out_bw.grad[1,:] = {[float(out_bw.grad[1,i]) for i in range(n)]}")


if __name__ == "__main__":
    main()
```

---

## 7. 참고 — 관련 문서

- `notes/forward_pass_callchain.md` — forward pass 호출 트리 (J2 디버깅 때 작성)
- `tests/test_diff_forward_kinematics.py` — 메인 회귀 테스트 (J1~J5)
- 관련 외부 PR: `#2742` (cleanup autodiff static for-loops, by duburcqa). `#2537` (freejoint+child hang fix). 둘 다 이미 cherry-pick / verify 완료.

### 7.1 backward-pass grad trace 방법론

사용자가 Genesis backward pass 의 silent-zero grad 디버깅에서 1차 도구로 쓰는 방법. (이 머신 외부 에이전트가 못 보는 로컬 메모에서 인용)

**적용 순서:**

1. 깨진 grad 를 보이는 **가장 작은 reproducer** 를 잡는다 (entity 1 개, step 1 개, joint topology 최소). `tests/test_diff_forward_kinematics.py` 와 `/tmp/probe_j*_grad_trace.py` 가 좋은 출발 템플릿.
2. `RigidSolver.substep_pre_coupling_grad` (`genesis/engine/solvers/rigid/rigid_solver.py`) 안의 **모든 `kernel.*.grad(...)` 호출 앞뒤에 grad-dump 프린트를 삽입**. dump 는 주요 adjoint 필드의 `abs().max()` 와 `norm()`:
   - `rigid_global_info.qpos`, `dofs_state.{vel,pos,acc,force}`
   - `links_state.{pos,quat,pos_bw,quat_bw}`
   - `joints_state.{xanchor,xaxis}`  
   각 adjoint buffer 는 `qd_to_torch(field.grad, copy=True)` 로 읽기. 평소엔 잠들도록 env var (`GENESIS_DEBUG_GRAD=1`) 로 gating. **이 helper 는 이미 `_debug_grad_dump` 로 구현돼있음 (§3.3.a).**
3. env var 켜고 reproducer 실행. upstream adjoint 가 진입하지만 (예: `kernel_update_cartesian_space.grad` 호출 직전 `links_state.quat.grad` non-zero) downstream adjoint 가 안 들어오는 (예: 호출 후 `rigid_global_info.qpos.grad` 0 그대로) 지점을 찾는다. **그 kernel 이 reverse-mode chain rule 이 빠지는 지점.**
4. 한 kernel 로 좁혀지면, kernel body 안의 **non-linear / branched 조각을 하나씩 trivial linear placeholder 로 바꿔가며** 재 trace. 예: `qd_rotvec_to_quat(axis*angle)` → 상수 `[1, axis*angle/2, ...]`. downstream adjoint 가 복구되는 지점이 범인.
5. Quadrants AD 자주 걸리는 함정 (우선순위 순):  
   (a) kernel 내 dynamic `if` 분기  
   (b) `qd.sqrt` / 0 근처 division (branch guard 가 있어도 발생)  
   (c) explicit field write-then-read 패턴  
   (d) **outer loop cross-iter same-buffer read** ← 현재 LDLT solve 막힌 원인  
   표준 fix: branch-free regularization (`sqrt(x*x + eps*eps)`), 특이점 근처 Taylor 전개, per-iteration kernel split, 최후 수단으로 `@qd.ad.grad_replaced` + 커스텀 backward kernel.

**왜 이 방법인가:**
- 사용자가 이미 여러 Genesis AD 이슈를 정확히 이 trace-driven 접근으로 잡았고 first-line tool 로 여김.
- Genesis 의 수동 `W/R/WR` cache 패턴과 `is_backward` template parameter 가 Quadrants adstack 과 비자명하게 상호작용 → static IR inspection 보다 trace 가 1 차 localization 에 훨씬 빠름.
- Trace 는 **회귀 체크 역할도 겸함** — fix 가 들어가면 adjoint 가 예상 downstream 필드까지 전달돼야 하고, 미래 regression 도 trace 에 즉시 나타남.
