# DiffRL Genesis Port — Overnight Work Summary

작성일: 2026-05-12 03:10 (overnight session)  
브랜치: `20260503_diff_rigid_demo`

자기 전 user 요청 정리 (`diff_rigid_demo` 세션 끝부분):
1. 광범위 회귀 → cleanup → commit
2. DiffRL-compatible 인프라 구현 (gradient reset, gym interface, env definitions)
3. cartpole SHAC 학습 시도 → reward 증가 확인
4. hopper/ant 는 floor contact 한계로 보류

---

## 진행 상황

### ✅ 완료

#### Phase B cleanup + regression (커밋 `fd40f1d3`, `c0d40498`, `652c8d5f`)
- `kernel_compute_qacc.grad` 의 두 가지 silent failure 우회 (Step 1 cross-iter + Step 2 trivial mul).
- per-link split bound `_MAX_LINKS = 2` → 동적 `self._max_n_links_across_entities`.
- `tests/test_quadrants_self_ref_ad.py` 회귀 가드 박제.
- 노트 문서 `notes/quadrants_cross_iter_ad_limitation.md` §7 업데이트.

광범위 회귀 (`tests/test_rigid_physics.py -m required`): **249 passed, 11 failed, 1 skipped**.
실패한 11개는 collision_edge_cases / box_plane_dynamics / equality_link 등 pre-existing — Phase B 와 무관.

#### Scene API: `scene.reset_grad()` (커밋 `da222a1f`, `dd7b1a59` 의 일부)
- public `scene.reset_grad()` 추가 — solver `.grad` 필드 + `_queried_states` 클리어 + `_cur_substep_global = 0` + `_forward_ready/_backward_ready` 재설정. 기존 `examples/diffrigid/test_*.py` 의 `reset_grad_only(scene)` 헬퍼와 동등.

#### `control_dofs_force` 를 @tracked 로 (커밋 `dd7b1a59`)
- 발견: 기존 `entity.control_dofs_force(force)` 는 `assign_indexed_tensor` 가 in-place 라서 autograd graph 가 끊김. 시뮬레이터 측 `ctrl_force.grad` 는 정상 계산되지만 torch 측 force 로 돌아오는 다리가 없었음.
- 새로 추가: `kernel_set_dofs_force_grad` (accessor.py) + `KinematicSolver.set_dofs_force_grad` + `RigidEntity.set_dofs_force_grad` + @tracked 데코레이터 + `"control_dofs_force"` 를 `_tgt_keys` 에 추가 + `process_input_grad` 의 새 case.
- 검증: cartpole 환경에서 actor(obs) → force → sim → reward → loss.backward(retain_graph=True) 로 `actor.weight.grad.abs().max() = 7660` 의 정상 gradient 확인 (이전에는 정확히 0).
- 주의: gradient bridge 가 fire 하려면 force 가 **gs.Tensor** 여야 함. `process_input_grad` 는 `hasattr(force, "_backward_from_qd")` 체크 후 호출. 일반 torch.Tensor 가 들어오면 silently skip.

#### DiffRL-compatible 인프라 (커밋 직전, `examples/diffrl/`)
- `genesis_env.py`: `GenesisDiffRLEnv` ABC. gym-style 버퍼, step/reset/clear_grad/initialize_trajectory.
- `envs/cartpole_swing_up.py` + `cartpole.xml`: DiffRL `cartpole_swing_up.py` 의 reward/obs/action/episode 1:1 포팅.
- `models/actor.py`: `ActorStochasticMLP` / `ActorDeterministicMLP`.
- `models/critic.py`: `CriticMLP` (last-layer zero-init for safe bootstrap).
- `running_mean_std.py`: Welford 온라인 통계.
- `algorithms/shac.py`: SHAC 알고리즘 본체 (TD-λ critic, γ-discounted actor return, target critic polyak, linear LR, grad clip).
- `train_shac.py` + `cfg/shac/cartpole_swing_up.yaml`: YAML config + entry point.

### ⚠️ 부분적으로 동작 / 미해결

#### Cartpole SHAC 수렴 안 됨
DiffRL paper config (H=32, num_actors=64, actor_lr=2e-3) 로:
- raw grad norm 이 1e7~1e9+ 로 폭주. clip=1.0 으로 방향은 보존되지만 매우 noisy.
- 정책 학습이 발산 — horizon reward 가 -400 → -1000 → 무작위 변동. ep_ret 가 -4k → -70k 까지 떨어짐.

H=8 + lr=5e-4 로 안정화 시도:
- raw grad norm ~1-3 로 안정됨.
- 그러나 horizon reward 가 random-action baseline (~-500) 근처에서 stuck. ep_ret 도 -12k ~ -21k 사이에서 진동.

**가설들 (다음 시도 후보)**:
1. **Actor architecture 차이** — DiffRL ActorStochasticMLP 는 orthogonal init + LayerNorm 사용. 내 버전은 default init + activation 만. cartpole swing-up 의 explore-exploit 균형에 영향 있을 수 있음.
2. **Critic warmup 부재** — 첫 epoch 부터 SHAC 의 critic bootstrap 이 actor loss 에 들어감. critic 이 random 상태에서 bootstrap 가 잘못된 signal 줄 수 있음. zero-init last layer 로 약간 완화는 했지만 critic-only 워밍업 페이즈 추가가 더 안전할 듯.
3. **`scene_anchor = state.qpos * 0.0` 트릭의 부작용** — force 를 gs.Tensor 로 promote 하기 위한 anchor 가 backward path 를 복잡하게 만듬. 가능성: `state.qpos.grad` 가 의도하지 않은 contribution 받음 → noisy gradient. 다른 promote 방법 (e.g. gs.Tensor 를 직접 만들어 graph 에 graft) 검토 필요.
4. **Cartpole swing-up 의 reward landscape** — locally smooth 하지 않을 가능성. SHAC paper 도 이 task 에 보통 500 에폭 필요. 더 길게 (1000+) 돌려보거나, DiffRL exact 한 BPTT-only baseline 부터 검증해보는 게 안전.

`examples/diffrl/cfg/shac/cartpole_swing_up.yaml` 의 yaml 코멘트에 현재 hyperparam 변경 사유 명시해둠.

### 🔒 미시작
- Hopper / Ant — floor contact 의 differentiable 처리가 제한적이라 user 대기.

---

## 권장 다음 단계 (다음 세션)

1. **DiffRL exact ActorStochasticMLP 재현** — orthogonal init + LayerNorm + identity 마지막 활성 + 외부 tanh. 변경량은 30줄 미만.
2. **Critic warmup 페이즈** — 첫 N=20 epoch 정도는 actor 안 업데이트하고 critic 만 학습. 검증 후 SHAC actor loss 켜기.
3. **scene_anchor 대체** — `gs.Tensor` 를 fresh 하게 만들어 autograd graph 에 graft 하는 헬퍼 추가. 예: `_promote_to_gs_tensor(t, scene)` — `torch.zeros_like` 의 gs.Tensor 버전을 만들고 `t` 의 grad 에 그대로 hook. 현재 anchor 방식의 noise 줄임.
4. **BPTT baseline** — SHAC 의 critic bootstrap 끄고 순수 BPTT (sum of rewards) 로 학습. 만약 BPTT 도 발산하면 environment / gradient 자체 문제. 만약 BPTT 가 수렴하고 SHAC 만 발산하면 bootstrap 가 원인.
5. **Hopper/Ant** — user 와 differentiable contact 의 어디까지가 OK 한지 명확히 한 후 진행.

---

## 커밋 히스토리 (이번 세션)
```
fd40f1d3 [BUG FIX] make differentiable rigid LDLT solve work on multi-link entities (Phase B)
c0d40498 [FEATURE] add regression test for Quadrants AD cross-iter same-buffer limitation
652c8d5f [MISC] document Phase B diagnostic + handoff for differentiable rigid solve
da222a1f [FEATURE] add scene.reset_grad() — clear gradients without resetting state
dd7b1a59 [FEATURE] make control_dofs_force differentiable via @tracked + force_grad accessor
???????? [FEATURE] add Genesis-backed DiffRL-style env scaffolding + cartpole swing-up
???????? [FEATURE] add minimal SHAC trainer for Genesis diffrigid envs (cartpole convergence TBD)
```
