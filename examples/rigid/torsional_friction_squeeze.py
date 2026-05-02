"""Torsional friction grip-force-vs-stop-time demo.

Sphere is given an initial spin about y at t=0. Two plates start a small distance away
and are pushed inward by a constant external force ``--grip_force``; gravity is compensated
on all bodies. As the plates close in and establish contact, friction at the contacts
decelerates the spinning sphere — and the stronger the grip, the faster the spin dies.

Why grip force, not μ_t?
------------------------
In MuJoCo's pyramidal-cone LCP with soft regularization, kinetic friction is *not* simply
``μ_t · F_n`` once a body is already slipping. The augmented constraint Jacobian (∝ μ_t)
and its diagonal regularization (∝ μ_t²) cause the μ_t factor to cancel in the constraint
force the LCP returns. The result is a viscous-style damping torque whose magnitude scales
linearly with the contact normal force ``F_n`` (= grip force here, when plates are in
quasi-static balance) but is independent of ``μ_t`` once ``μ_t > 0``. The on/off switch
still works (``μ_t = 0`` means no torsional resistance and the sphere keeps spinning), and
the *static* stiction threshold *is* ``μ_t``-dependent (see ``torsional_friction_breakaway.py``);
only the kinetic-decay rate is μ-cancelled. This demo therefore varies grip force rather
than μ_t, since that is what physically controls the kinetic decay rate in this LCP
formulation.

Run examples (use ``-v`` for the viewer, watch the yellow bar slow as plates close in):

    python examples/rigid/torsional_friction_squeeze.py --grip_force 5    # gentle squeeze, slow stop
    python examples/rigid/torsional_friction_squeeze.py --grip_force 50   # firm squeeze,  fast stop
    python examples/rigid/torsional_friction_squeeze.py --sweep            # compare grip forces

Reliable μ_t range
------------------
Same caveat as the other demos: keep ``--mu_t`` inside ``[0.01, 0.3]`` for a well-conditioned
LCP. The default of 0.1 is fine. See ``materials.Rigid.friction_torsional`` for details.
"""

import argparse

import numpy as np

import genesis as gs


def _build_scene(mu_t: float, enable_torsional: bool, initial_gap: float, vis: bool):
    sphere_radius = 0.10
    sphere_center_z = 0.40
    plate_size = (0.30, 0.025, 0.30)
    plate_offset_y = sphere_radius + 0.5 * plate_size[1] + initial_gap

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.001),
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=enable_torsional,
            noslip_iterations=10,
        ),
        show_viewer=vis,
    )
    scene.add_entity(gs.morphs.Plane())

    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=sphere_radius, pos=(0.0, 0.0, sphere_center_z)),
        material=gs.materials.Rigid(rho=300.0, friction=1.5, friction_torsional=mu_t, gravity_compensation=1.0),
        surface=gs.surfaces.Plastic(color=(1.0, 0.3, 0.3)),
    )
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
    for sign in (-1.0, 1.0):
        plate = scene.add_entity(
            gs.morphs.Box(
                size=plate_size,
                pos=(0.0, sign * plate_offset_y, sphere_center_z),
            ),
            material=gs.materials.Rigid(rho=2000.0, friction=1.5, friction_torsional=mu_t, gravity_compensation=1.0),
            surface=gs.surfaces.Plastic(color=(0.3, 0.4, 0.9)),
        )
        plates.append((plate, sign))

    return scene, sphere, marker, marker_local_offset, plates


def _sync_marker_to_sphere(sphere, marker, marker_local_offset):
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


def _apply_grip_force(rigid, plates, grip_force, lin_damp=30.0, ang_damp=10.0):
    """Push each plate inward + linear/angular damping to keep approach stable."""
    for plate, sign in plates:
        plate_vel = plate.get_dofs_velocity().cpu().numpy().squeeze()
        f_y = -sign * grip_force - lin_damp * plate_vel[1]
        f_x = -lin_damp * plate_vel[0]
        f_z = -lin_damp * plate_vel[2]
        rigid.apply_links_external_force(np.array([f_x, f_y, f_z]), links_idx=[plate.base_link_idx])
        rigid.apply_links_external_torque(-ang_damp * plate_vel[3:6], links_idx=[plate.base_link_idx])


def _run_single(
    mu_t: float,
    enable_torsional: bool,
    grip_force: float,
    initial_spin: float,
    initial_gap: float,
    n_steps: int,
    vis: bool,
):
    scene, sphere, marker, marker_offset, plates = _build_scene(
        mu_t=mu_t, enable_torsional=enable_torsional, initial_gap=initial_gap, vis=vis
    )
    scene.build()
    rigid = scene.sim.rigid_solver

    # Set initial spin, then start applying grip force from t=0.
    sphere_dofs_vel = np.zeros(6)
    sphere_dofs_vel[4] = initial_spin
    sphere.set_dofs_velocity(velocity=sphere_dofs_vel)

    dt = scene.sim_options.dt
    omegas = []
    plate_y_history = []
    for i in range(n_steps):
        _apply_grip_force(rigid, plates, grip_force)
        scene.step()
        _sync_marker_to_sphere(sphere, marker, marker_offset)
        omega = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[4])
        omegas.append(omega)
        # Track right plate's y position (sign=+1) to see when contact establishes.
        plate_y_history.append(float(plates[1][0].get_pos().cpu().numpy().squeeze()[1]))
    return omegas, plate_y_history, dt


def _stop_step(omegas, threshold_ratio: float, initial_spin: float):
    """First step at which |omega| drops below threshold_ratio * |initial_spin|."""
    target = abs(initial_spin) * threshold_ratio
    return next((i for i, w in enumerate(omegas) if abs(w) < target), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument("--mu_t", type=float, default=0.1)
    parser.add_argument("--grip_force", type=float, default=20.0)
    parser.add_argument("--initial_spin", type=float, default=10.0, help="rad/s about y at t=0.")
    parser.add_argument(
        "--initial_gap",
        type=float,
        default=0.015,
        help="Initial half-gap between plate inner face and sphere surface [m]. Plates start "
        "this far away and squeeze inward.",
    )
    parser.add_argument("--n_steps", type=int, default=2500)
    parser.add_argument("--sweep", action="store_true", default=False)
    args = parser.parse_args()

    if args.sweep:
        gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")
        grip_force_values = [2.0, 5.0, 10.0, 20.0, 40.0, 80.0]
        print(
            f"Initial spin = {args.initial_spin} rad/s. Plates start {args.initial_gap * 1000:.0f} mm "
            f"away, squeeze inward; μ_t fixed at {args.mu_t}."
        )
        print("Stronger grip → faster contact establishment + larger F_n → faster spin decay.")
        print()
        header = (
            f"{'grip_force':>11}  {'t (50% spin)':>13}  {'t (10% spin)':>13}  {'t (1% spin)':>13}  {'final |ω|':>10}"
        )
        print(header)
        print("-" * len(header))
        for gf in grip_force_values:
            omegas, _, dt = _run_single(
                mu_t=args.mu_t,
                enable_torsional=args.mu_t > 0,
                grip_force=gf,
                initial_spin=args.initial_spin,
                initial_gap=args.initial_gap,
                n_steps=args.n_steps,
                vis=False,
            )
            t_half = _stop_step(omegas, 0.5, args.initial_spin)
            t_tenth = _stop_step(omegas, 0.1, args.initial_spin)
            t_pct = _stop_step(omegas, 0.01, args.initial_spin)
            t_half_s = f"{t_half * dt:.3f}s" if t_half is not None else "  -- "
            t_tenth_s = f"{t_tenth * dt:.3f}s" if t_tenth is not None else "  -- "
            t_pct_s = f"{t_pct * dt:.3f}s" if t_pct is not None else "  -- "
            print(f"{gf:>11.1f}  {t_half_s:>13}  {t_tenth_s:>13}  {t_pct_s:>13}  {abs(omegas[-1]):>10.3f}")
        print()
        print(
            "Within Genesis' pyramidal-cone soft-constraint LCP the kinetic-decay rate scales "
            "with F_n (here ≈ grip_force) but is independent of μ_t once μ_t > 0. The static "
            "stiction threshold remains μ_t-dependent (see torsional_friction_breakaway.py)."
        )
        return

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)
    print(
        f"=== mu_t={args.mu_t}, grip_force={args.grip_force} N, "
        f"initial_spin={args.initial_spin} rad/s, initial_gap={args.initial_gap * 1000:.0f} mm ==="
    )
    enable = args.mu_t > 0
    omegas, plate_y, dt = _run_single(
        mu_t=args.mu_t,
        enable_torsional=enable,
        grip_force=args.grip_force,
        initial_spin=args.initial_spin,
        initial_gap=args.initial_gap,
        n_steps=args.n_steps,
        vis=args.vis,
    )

    contact_step = next((i for i, py in enumerate(plate_y) if abs(py) < 0.16), None)
    t_half = _stop_step(omegas, 0.5, args.initial_spin)
    t_tenth = _stop_step(omegas, 0.1, args.initial_spin)
    t_pct = _stop_step(omegas, 0.01, args.initial_spin)

    print()
    print(
        f"  contact established (right plate inside ~16cm) at: {contact_step * dt:.3f}s"
        if contact_step is not None
        else "  no contact established within run"
    )
    print(f"  time for spin to drop to 50% : {t_half * dt:.3f}s" if t_half is not None else "  -- (never)")
    print(f"  time for spin to drop to 10% : {t_tenth * dt:.3f}s" if t_tenth is not None else "  -- (never)")
    print(f"  time for spin to drop to  1% : {t_pct * dt:.3f}s" if t_pct is not None else "  -- (never)")
    print(f"  final |ω|                    : {abs(omegas[-1]):.3f} rad/s")


if __name__ == "__main__":
    main()
