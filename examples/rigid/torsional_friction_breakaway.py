"""Torsional friction breakaway demo (force-controlled grip).

Two free-floating plates squeeze a free-floating sphere from opposite sides (along ±y).
Each plate is pushed inward by a constant external force ``--grip_force``; gravity is
compensated on all bodies so the only horizontal force at the contact is that grip force.
In equilibrium the contact normal force is then exactly ``F_n = grip_force`` per plate
(modulo plate inertia which is small for the duration of the run).

A linearly ramped external torque is then applied to the sphere about y. The sphere stays
stuck while the applied torque is below ``μ_t · F_n``, and breaks away once the torque
crosses that threshold. Sweeping μ_t with grip force fixed traces out the Coulomb stiction
line. To prevent unbounded post-breakaway acceleration (and the LCP NaN that follows), the
applied torque is frozen at the breakaway value once the sphere starts to spin.

This demo isolates the kinetic-friction regime that the binary spin demo could not show:
in ``torsional_friction_grasp.py`` the plates are kinematic (``fixed=True``), so the LCP
can supply unbounded F_n on demand and stiction looks effectively infinite.

Reliable μ_t range
------------------
Genesis (and MuJoCo) implement the friction cone via the pyramidal approximation. Each
pyramid edge augments the normal direction by ``μ·tangent``; as ``μ`` grows past ~0.3 this
augmented direction becomes nearly tangential, the diag scaling ``μ²·invweight_rot``
dominates, and the LCP loses the ability to cleanly separate normal and friction force.
Empirically, this demo gives clean Coulomb behaviour in **``μ_t ∈ [0.05, 0.3]``** with
``grip_force`` ~ 5-30 N — the observed breakaway is then ~2× the per-contact prediction
``μ_t · F_n``, matching the two-contact pyramid sum. Outside this range:

* ``μ_t < 0.01`` is clamped to 0.01 by the contact-merge floor (mirrors slide friction's
  stability floor); below that the constraint is too soft and oscillates.
* ``μ_t > 0.3`` produces a *spurious early breakaway*: the breakaway torque actually
  *decreases* as ``μ_t`` grows, because the ill-conditioned LCP releases the sphere
  prematurely. This is a known limitation of the pyramidal cone formulation (MuJoCo
  recommends switching to the elliptic cone, ``solver=Newton`` with ``cone=elliptic``,
  for high-friction regimes; Genesis does not yet implement the elliptic cone).
* Combinations with very large ``μ_t · F_n`` (e.g. ``μ_t=0.5, grip_force≥10``) can also
  drive the LCP to NaN; the sweep mode caps the per-row torque ramp accordingly.

Run examples (use ``-v`` to open the viewer):

    python examples/rigid/torsional_friction_breakaway.py --mu_t 0.05  --grip_force 20
    python examples/rigid/torsional_friction_breakaway.py --mu_t 0.20  --grip_force 20
    python examples/rigid/torsional_friction_breakaway.py --sweep
"""

import argparse

import numpy as np

import genesis as gs


def _build_scene(mu_t: float, enable_torsional: bool, grip_force: float, vis: bool, use_cpu: bool):
    """Build a sphere clamped between two free-floating, gravity-compensated plates."""
    sphere_radius = 0.10
    sphere_center_z = 0.40
    plate_size = (0.30, 0.025, 0.30)
    plate_offset_y = sphere_radius + 0.5 * plate_size[1] + 0.005  # plates start just outside sphere

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001),  # small dt: free plates need it for stability
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=enable_torsional,
            noslip_iterations=10,
        ),
        show_viewer=vis,
    )
    scene.add_entity(gs.morphs.Plane())

    # Sphere, gravity compensated so the only forces on it are the contact + applied torque.
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=sphere_radius, pos=(0.0, 0.0, sphere_center_z)),
        material=gs.materials.Rigid(rho=300.0, friction=1.5, friction_torsional=mu_t, gravity_compensation=1.0),
        surface=gs.surfaces.Plastic(color=(1.0, 0.3, 0.3)),
    )
    # Visible spin marker: a yellow bar that follows the sphere's pose every step,
    # so its rotation about y is unambiguous in the viewer.
    marker_local_offset = np.array([0.0, 0.0, sphere_radius + 0.005])
    marker = scene.add_entity(
        gs.morphs.Box(
            size=(0.18, 0.020, 0.020),
            pos=(0.0, 0.0, sphere_center_z + marker_local_offset[2]),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )
    plates = []
    for sign, color in ((-1.0, (0.3, 0.4, 0.9)), (1.0, (0.3, 0.4, 0.9))):
        plate = scene.add_entity(
            gs.morphs.Box(
                size=plate_size,
                pos=(0.0, sign * plate_offset_y, sphere_center_z),
            ),
            material=gs.materials.Rigid(rho=2000.0, friction=1.5, friction_torsional=mu_t, gravity_compensation=1.0),
            surface=gs.surfaces.Plastic(color=color),
        )
        plates.append((plate, sign))

    return scene, sphere, plates, marker, marker_local_offset


def _apply_grip_force(rigid, plates, grip_force, lin_damp=30.0, ang_damp=10.0):
    """Push each plate toward the sphere with a constant horizontal force, plus a small
    linear/angular velocity damping force to suppress free-body oscillation while still
    letting the plate settle into a force balance with the contact."""
    for plate, sign in plates:
        plate_vel = plate.get_dofs_velocity().cpu().numpy().squeeze()  # 6-vec (lin xyz, ang xyz)
        f_y = -sign * grip_force - lin_damp * plate_vel[1]
        f_x = -lin_damp * plate_vel[0]
        f_z = -lin_damp * plate_vel[2]
        rigid.apply_links_external_force(np.array([f_x, f_y, f_z]), links_idx=[plate.base_link_idx])
        rigid.apply_links_external_torque(-ang_damp * plate_vel[3:6], links_idx=[plate.base_link_idx])


def _sync_marker_to_sphere(sphere, marker, marker_local_offset):
    """Copy sphere pose to the kinematic marker so it visually tracks rotation."""
    sphere_pos = sphere.get_pos().cpu().numpy().squeeze()
    sphere_quat = sphere.get_quat().cpu().numpy().squeeze()
    w, x, y, z = sphere_quat
    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    marker.set_pos(sphere_pos + R @ marker_local_offset)
    marker.set_quat(sphere_quat)


def _settle(scene, rigid, plates, grip_force, n_settle, sphere, marker, marker_local_offset):
    for _ in range(n_settle):
        _apply_grip_force(rigid, plates, grip_force)
        scene.step()
        _sync_marker_to_sphere(sphere, marker, marker_local_offset)


def _run_single(
    mu_t: float,
    enable_torsional: bool,
    grip_force: float,
    torque_max: float,
    n_steps: int,
    vis: bool,
    use_cpu: bool,
    n_settle: int = 800,
):
    scene, sphere, plates, marker, marker_local_offset = _build_scene(
        mu_t=mu_t, enable_torsional=enable_torsional, grip_force=grip_force, vis=vis, use_cpu=use_cpu
    )
    scene.build()
    rigid = scene.sim.rigid_solver

    _settle(scene, rigid, plates, grip_force, n_settle, sphere, marker, marker_local_offset)

    sphere_link = sphere.base_link_idx
    omegas = []
    applied = []
    breakaway_threshold = 0.5
    held_torque = None  # freeze τ at breakaway to prevent unbounded acceleration / NaN
    for i in range(n_steps):
        tau = torque_max * (i + 1) / n_steps if held_torque is None else held_torque
        _apply_grip_force(rigid, plates, grip_force)
        rigid.apply_links_external_torque(np.array([0.0, tau, 0.0]), links_idx=[sphere_link])
        scene.step()
        _sync_marker_to_sphere(sphere, marker, marker_local_offset)
        omega = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[4])
        omegas.append(omega)
        applied.append(tau)
        if held_torque is None and abs(omega) > breakaway_threshold:
            held_torque = tau
    return omegas, applied


def _breakaway_step(omegas, threshold=0.5):
    return next((i for i, w in enumerate(omegas) if abs(w) > threshold), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--mu_t", type=float, default=0.1)
    parser.add_argument("--grip_force", type=float, default=20.0, help="Per-plate inward force [N].")
    parser.add_argument("--torque_max", type=float, default=8.0, help="Peak external torque on sphere [N*m].")
    parser.add_argument("--n_steps", type=int, default=600)
    parser.add_argument("--sweep", action="store_true", default=False)
    args = parser.parse_args()

    if args.sweep:
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
        mu_t_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
        print(f"Force-controlled grip: F_n = {args.grip_force} N per plate (compensated gravity).")
        print(
            "For each μ_t the torque ramp peak is sized to ~2.5× the per-contact Coulomb "
            "prediction, leaving headroom past the 2-contact breakaway without runaway."
        )
        print()
        header = (
            f"{'mu_t':>8}  {'predicted τ_max':>16}  {'breakaway τ':>14}  {'τ_break / (μ_t·F_n)':>22}  {'peak |ω|':>10}"
        )
        print(header)
        print("-" * len(header))
        for mu_t in mu_t_values:
            enable = mu_t > 0.0
            # Per-row torque_max sized to comfortably cover the 2-contact threshold (~2*μ_t*F_n)
            # without ramping so far past it that the LCP destabilises.
            row_torque_max = max(0.5, 2.5 * mu_t * args.grip_force)
            omegas, applied = _run_single(
                mu_t=mu_t,
                enable_torsional=enable,
                grip_force=args.grip_force,
                torque_max=row_torque_max,
                n_steps=args.n_steps,
                vis=False,
                use_cpu=args.cpu,
            )
            bs = _breakaway_step(omegas)
            tau_b = applied[bs] if bs is not None else None
            predicted = mu_t * args.grip_force  # naive Coulomb prediction (single contact)
            ratio = (tau_b / predicted) if (tau_b is not None and predicted > 0) else None
            tau_b_str = f"{tau_b:.3f}" if tau_b is not None else f"> {row_torque_max:.2f}"
            ratio_str = f"{ratio:.2f}" if ratio is not None else "  -- "
            print(
                f"{mu_t:>8.3f}  {predicted:>16.3f}  {tau_b_str:>14}  "
                f"{ratio_str:>22}  {max(abs(w) for w in omegas):>10.3f}"
            )
        print()
        print(
            "predicted τ_max = μ_t · grip_force (single contact; the actual cone with two "
            "contacts opposing has roughly 2× headroom, and the LCP regularization adds a bit "
            "more, so observed breakaway is typically a small multiple of the prediction)."
        )
        return

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    enable = args.mu_t > 0
    print(f"=== mu_t={args.mu_t}, grip_force={args.grip_force} N, torque ramp 0→{args.torque_max} N*m ===")
    omegas, applied = _run_single(
        mu_t=args.mu_t,
        enable_torsional=enable,
        grip_force=args.grip_force,
        torque_max=args.torque_max,
        n_steps=args.n_steps,
        vis=args.vis,
        use_cpu=args.cpu,
    )
    bs = _breakaway_step(omegas)
    print()
    if bs is None:
        print(f"  Sphere never broke stiction (peak |ω|={max(abs(w) for w in omegas):.3f} rad/s).")
        print(f"  Applied torque maxed out at {args.torque_max:.3f} N*m.")
    else:
        print(f"  Stiction broke at step {bs} with applied torque = {applied[bs]:.3f} N*m.")
        print(f"  Naive Coulomb prediction (μ_t · grip_force) = {args.mu_t * args.grip_force:.3f} N*m.")
        print(f"  Final |ω| = {abs(omegas[-1]):.3f} rad/s (kinetic friction regime).")


if __name__ == "__main__":
    main()
