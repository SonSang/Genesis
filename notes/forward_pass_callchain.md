# Forward pass 호출 체인 (rigid solver, `requires_grad=True`)

J2 (fixed-base revolute) autograd 실패를 디버깅하면서 정리한 노트.
아래 체인은 **단일 `scene.step()` 호출 중** 실제로 도는 함수들의 순서를 보여줌.
diff 관련 fix를 추가할 위치를 잡거나, J2 grad가 어디서 흘러야 하는지 확인할 때 참고.

## 전체 호출 트리

```
scene.step()
└── simulator.step()
    └── for each substep f:
        └── solver.substep(f)   [rigid_solver.py:940]
            │
            ├── kernel_save_adjoint_cache(f=0)   # f==0에서만, dofs_state.vel/qpos 캐싱
            │
            ├── kernel_step_1                    [rigid_solver.py:2763]
            │   ├── func_update_cartesian_space  # FK at qpos_t (이미 갱신된 경우 skip)
            │   │   └── func_update_cartesian_space_entity (per entity)
            │   │       ├── func_forward_kinematics_entity   ← REVOLUTE 분기 (forward_kinematics.py:622)
            │   │       ├── func_COM_links_entity
            │   │       └── func_update_geoms_entity
            │   ├── func_forward_velocity        # cdof_ang, cdof_vel 계산
            │   └── func_forward_dynamics        # mass matrix, bias forces, ...
            │
            ├── _func_constraint_force()         # 제약 풀이 (disable_constraint=True면 no-op)
            │
            ├── kernel_step_2                    [rigid_solver.py:2827]
            │   ├── func_update_acc              # 가속도 갱신
            │   ├── func_integrate               ← qpos_{t+1} = qpos_t + v · dt
            │   │
            │   ├── (forward only: is_backward=False)
            │   │   ├── func_copy_next_to_curr
            │   │   ├── func_update_cartesian_space   ← 2번째 FK at qpos_{t+1}  ★ state.quat 의 출처
            │   │   │   └── (위와 같은 호출 체인; REVOLUTE 분기 다시 진입)
            │   │   └── func_forward_velocity
            │
            └── kernel_save_adjoint_cache(f+1)
```

## `func_forward_kinematics_entity` 내부 (per link, per joint)

```
for i_l in entity.link_range:
    pos  = links_info.pos[i_l]
    quat = links_info.quat[i_l]                                  # 초기화
    if parent_idx != -1:
        부모의 pos/quat 와 합성                                   # J2 root는 skip (parent_idx == -1)

    for i_j in link.joint_range:
        xanchor, xaxis 설정 (joint_type 분기)
        if joint_type == FREE:        quat = qpos[q+3:q+7] / ||·||
        elif joint_type == REVOLUTE:                             ← J2 케이스
            angle = qpos[q] - qpos0[q]
            dofs_state.pos[d] = angle                            # write
            qloc  = rotvec_to_quat(axis * dofs_state.pos[d])     # 방금 쓴 값을 즉시 read
            quat  = compose(qloc, prev_quat)
        elif joint_type == PRISMATIC:
            마찬가지로 dofs_state.pos[d] 에 write-then-read

    # 최종 commit
    if not (parent_idx == -1 AND is_fixed):
        links_state.pos[i_l]  = R(pos_bw,  I_jf, pos, BW)
        links_state.quat[i_l] = R(quat_bw, I_jf, quat, BW)
```

## `state.quat` 의 의존성 (forward 방향)

```
v  →  func_integrate  →  qpos_{t+1}
                ↓
        func_update_cartesian_space (2번째, 적분 후)
                ↓
        func_forward_kinematics_entity → REVOLUTE 분기
                ↓
        links_state.quat[i_l, i_b]
                ↓
        scene.get_state() / kernel_get_state
                ↓
        state.quat
```

## Backward sequence (참고)

`requires_grad=True` 모드의 backward 는 매 substep마다
`rigid_solver.substep_pre_coupling_grad` (line 1224+) 에서 다음 순서로 트리거:

```
loss.backward() → scene._backward() → 매 substep마다:
  1. kernel_forward_velocity.grad(is_backward=True)
       ← kernel_step_2 안의 적분 후 forward_velocity 의 backward
  2. kernel_update_cartesian_space.grad(is_backward=True)
       ← kernel_step_2 안의 적분 후 FK 의 backward
       ★ J2 의 quat → qpos → v 체인이 끊기는 지점
  3. kernel_begin_backward_substep
  4. kernel_step_2.grad(is_backward=True)        ← 적분기 backward (qpos → vel)
  5. … 이후 kernel_step_1.grad 류
```

즉 J2 의 사라진 grad (`set_dofs_velocity → state.quat`, analytical = 0 vs FD = 0.013) 는
위 2단계, **`kernel_update_cartesian_space.grad` 안에서** 손실됨 —
fixed-base root 에 대해 `func_forward_kinematics_entity` 의 REVOLUTE 분기를 통한 adjoint propagation 이 누락된다.

## 다음 디버깅 단계

- REVOLUTE 분기에서 한 부분씩 떼어내고 `test_diff_fk_revolute` 재실행:
  - `quat_ = transform_quat_by_quat(qloc, R(…))` 의 `R(links_state.quat_bw, …)` 인다이렉션 제거,
    `quat` 직접 합성
  - `links_state.quat[i_l] = R(quat_bw, I_jf, quat, BW)` 를 `quat` 의 직접 대입으로 교체
  - `dofs_state.pos[dof_start] = angle; … axis * dofs_state.pos[d]` 의 write-then-read 패턴을
    완전 우회하고 로컬 `angle` 만 사용
- `QD_DUMP_IR=1` 로 `kernel_update_cartesian_space` 의 컴파일된 backward IR 을 확인,
  REVOLUTE 분기에 chain rule 이 실제로 들어가는지 검증
- FREE 분기 (J1, 정상 동작) 와 비교 — forward 모양은 동일하고 joint-type 분기만 다름.
  REVOLUTE 에서 adstack 파이프라인이 어느 라인을 elide 하는지 좁히는 데 유용.
