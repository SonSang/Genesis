"""Torsional friction demo.

Two fixed plates squeeze a sphere from opposite sides (along ±y). The sphere is given an
initial angular velocity about y — i.e. about the contact-normal at both plate contacts.

At a point contact, sliding friction has zero moment arm about the contact normal, so it
cannot oppose this kind of rotation. With ``enable_torsional_friction=False`` (legacy
behavior) the sphere therefore keeps spinning between the plates indefinitely. With the
flag enabled and ``friction_torsional > 0`` on both surfaces, contact develops a torque
``|τ_n| ≤ μ_t · F_n`` that bleeds the angular momentum out and the spin decays.

Suggested runs (use ``-v`` to open the viewer):

    python examples/rigid/torsional_friction_grasp.py                          # OFF: spin persists indefinitely
    python examples/rigid/torsional_friction_grasp.py --torsional --mu_t 1.0   # large μ_t: stops within one step
    python examples/rigid/torsional_friction_grasp.py --torsional --mu_t 0.01  # moderate: stops in ~10 steps
    python examples/rigid/torsional_friction_grasp.py --torsional --mu_t 0.005 # small: visibly graduated decay
    python examples/rigid/torsional_friction_grasp.py --sweep                  # automated sweep over μ_t values

The yellow bar embedded in the red sphere visualises the rotation. ``noslip_iterations=10``
is on by default to remove the soft-constraint vertical drift; pass ``--noslip_iterations 0``
to see the legacy slide drift (a separate phenomenon from torsional friction).
"""

import argparse

import numpy as np

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    parser.add_argument("-c", "--cpu", action="store_true", default=False)
    parser.add_argument(
        "--torsional",
        action="store_true",
        default=False,
        help="Enable torsional friction (default: off; sphere spins forever).",
    )
    parser.add_argument(
        "--mu_t",
        type=float,
        default=1.0,
        help="Torsional friction coefficient applied to both plates and sphere when --torsional is set.",
    )
    parser.add_argument(
        "--initial_spin",
        type=float,
        default=10.0,
        help="Initial angular velocity given to the sphere about the pinch axis [rad/s].",
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=600,
        help="Total simulation steps after the spin is applied.",
    )
    parser.add_argument(
        "--noslip_iterations",
        type=int,
        default=10,
        help="Number of noslip projection iterations (default 10; pass 0 to disable). Soft "
        "contacts let the sphere drift downward slightly even when sticking should hold; "
        "noslip projects friction forces back onto the hard friction cone after the main "
        "solver and removes that drift. This is independent of torsional friction.",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        default=False,
        help="Run a quick sweep over several μ_t values (no viewer) and print a comparison "
        "table of decay timescales. Overrides --torsional and --mu_t.",
    )
    parser.add_argument(
        "--mode",
        choices=("spin", "torque"),
        default="spin",
        help="'spin': impulsive initial spin → watch decay. 'torque': zero initial spin, apply "
        "a slowly ramped external torque → watch the breakaway threshold (μ_t · F_n) where "
        "stiction gives way to slip.",
    )
    parser.add_argument(
        "--torque_ramp_max",
        type=float,
        default=4.0,
        help="In --mode torque, the maximum external torque [N*m] reached at the end of the run "
        "(ramped linearly from zero). Larger μ_t requires larger applied torque to break stiction.",
    )
    args = parser.parse_args()

    if args.sweep:
        if args.mode == "torque":
            _run_torque_sweep(use_cpu=args.cpu, n_steps=args.n_steps, torque_ramp_max=args.torque_ramp_max)
        else:
            _run_sweep(use_cpu=args.cpu, n_steps=args.n_steps, initial_spin=args.initial_spin)
        return

    gs.init(backend=gs.cpu if args.cpu else gs.gpu)

    label = "ON" if args.torsional else "OFF"
    mu_t = args.mu_t if args.torsional else 0.0
    print(f"=== Torsional friction {label} (mu_t={mu_t}) ===")

    sphere_radius = 0.15  # large flywheel-style sphere — high inertia for visible graduated decay
    sphere_center_z = 0.30
    plate_thickness = 0.02
    overlap = 0.002  # plates penetrate the sphere this much, generating a steady grip force
    sphere_density = 100.0  # kg/m^3, low to keep gravitational F_n modest while I_y stays high

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=args.torsional,
            noslip_iterations=args.noslip_iterations,
        ),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    sphere = scene.add_entity(
        gs.morphs.Sphere(
            radius=sphere_radius,
            pos=(0.0, 0.0, sphere_center_z),
        ),
        material=gs.materials.Rigid(rho=sphere_density, friction=1.5, friction_torsional=mu_t),
        surface=gs.surfaces.Plastic(color=(1.0, 0.3, 0.3)),
    )

    # Visible spin marker: a thin bright bar that follows the sphere's pose every step,
    # so its rotation about y (the pinch axis) is unambiguous in the viewer. Kept out of
    # collision; not welded to avoid coupling its inertia/gravity into the sphere dynamics.
    marker_local_offset = np.array([0.0, 0.0, sphere_radius + 0.005])
    marker = scene.add_entity(
        gs.morphs.Box(
            size=(0.25, 0.025, 0.025),
            pos=(0.0, 0.0, sphere_center_z + marker_local_offset[2]),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(1.0, 1.0, 0.0)),
    )

    # Two kinematic plates pinching the sphere along ±y. ``fixed=True`` makes them static
    # world geometry so the contact normal force at the sphere is whatever the LCP needs
    # to keep the sphere from penetrating, giving a steady, repeatable grip force.
    plate_y = sphere_radius + 0.5 * plate_thickness - overlap  # interpenetration = ``overlap``
    for sign, name in ((-1.0, "plate_neg_y"), (1.0, "plate_pos_y")):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.3, plate_thickness, 0.3),
                pos=(0.0, sign * plate_y, sphere_center_z),
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=1.5, friction_torsional=mu_t),
            surface=gs.surfaces.Plastic(color=(0.3, 0.4, 0.9)),
        )

    scene.build()

    rigid = scene.sim.rigid_solver

    def _sync_marker_to_sphere():
        """Copy sphere pose to the kinematic marker so it visually tracks rotation."""
        sphere_pos = sphere.get_pos().cpu().numpy().squeeze()
        sphere_quat = sphere.get_quat().cpu().numpy().squeeze()
        # Rotate the local marker offset by the sphere's quaternion to find world position.
        # quat is [w, x, y, z] in Genesis convention; compose: p_world = sphere_pos + R(quat) @ offset_local
        w, x, y, z = sphere_quat
        R = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        marker_pos = sphere_pos + R @ marker_local_offset
        marker.set_pos(marker_pos)
        marker.set_quat(sphere_quat)

    # 1) Settle the sphere between the plates so the grip force converges.
    print("Settling sphere between the plates...")
    for _ in range(80):
        scene.step()
        _sync_marker_to_sphere()
    z_after_settle = float(sphere.get_pos().cpu().numpy().squeeze()[2])

    dt = scene.sim_options.dt
    omegas = []
    z_history = []
    applied_torques = []
    cumulative_angle = 0.0

    if args.mode == "spin":
        # 2a) Impulsive initial spin → watch decay.
        sphere_dofs_vel = np.zeros(6)
        sphere_dofs_vel[4] = args.initial_spin
        sphere.set_dofs_velocity(velocity=sphere_dofs_vel)
        print(f"Initial spin set: omega_y = {args.initial_spin} rad/s; running {args.n_steps} steps...")

        for _ in range(args.n_steps):
            scene.step()
            _sync_marker_to_sphere()
            omega_y = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[4])
            omegas.append(omega_y)
            z_history.append(float(sphere.get_pos().cpu().numpy().squeeze()[2]))
            cumulative_angle += omega_y * dt
    else:
        # 2b) Apply linearly increasing external torque → watch breakaway from stiction.
        sphere_link_idx = sphere.base_link_idx
        rigid_solver = rigid
        print(
            f"Ramping external torque 0 → {args.torque_ramp_max} N*m over {args.n_steps} steps; "
            "watching for stiction breakaway..."
        )
        for i in range(args.n_steps):
            tau = args.torque_ramp_max * (i + 1) / args.n_steps
            rigid_solver.apply_links_external_torque(np.array([0.0, tau, 0.0]), links_idx=[sphere_link_idx])
            scene.step()
            _sync_marker_to_sphere()
            omega_y = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[4])
            omegas.append(omega_y)
            z_history.append(float(sphere.get_pos().cpu().numpy().squeeze()[2]))
            applied_torques.append(tau)
            cumulative_angle += omega_y * dt

    drop_total = z_after_settle - z_history[-1]

    print()
    if args.mode == "spin":
        print("[spin]")
        print(f"  initial |omega_y|:                 {abs(args.initial_spin):.3f} rad/s")
        print(f"  |omega_y| @ step 100:              {abs(omegas[99]):.3f} rad/s")
        print(f"  |omega_y| @ step 300:              {abs(omegas[299]):.3f} rad/s")
        print(f"  |omega_y| @ step {args.n_steps}:              {abs(omegas[-1]):.3f} rad/s")
        print(f"  cumulative rotation (radians):     {cumulative_angle:+.2f}")
        print(f"  cumulative rotation (degrees):     {cumulative_angle * 180 / np.pi:+.1f}")
        decay_ratio = abs(omegas[-1]) / max(abs(args.initial_spin), 1e-9)
        print(f"  decay ratio (final/initial):       {decay_ratio:.3f}")
    else:
        # Detect breakaway: first step where |omega| crosses a small threshold.
        breakaway_threshold = 0.5  # rad/s
        breakaway_step = next((i for i, w in enumerate(omegas) if abs(w) > breakaway_threshold), None)
        breakaway_torque = applied_torques[breakaway_step] if breakaway_step is not None else None
        peak_omega = max(abs(w) for w in omegas)
        print("[external torque → breakaway]")
        print(f"  applied torque ramp:               0 → {args.torque_ramp_max:.3f} N*m")
        print(f"  |omega_y| @ step 100:              {abs(omegas[99]):.3f} rad/s")
        print(f"  |omega_y| @ step 300:              {abs(omegas[299]):.3f} rad/s")
        print(f"  |omega_y| @ step {args.n_steps}:              {abs(omegas[-1]):.3f} rad/s")
        print(f"  peak |omega_y| during run:         {peak_omega:.3f} rad/s")
        print(f"  cumulative rotation (degrees):     {cumulative_angle * 180 / np.pi:+.1f}")
        if breakaway_torque is None:
            print(f"  breakaway torque:                  > {args.torque_ramp_max:.3f} N*m (never broke)")
        else:
            print(f"  breakaway torque (|ω|>{breakaway_threshold} rad/s): {breakaway_torque:.3f} N*m")
    print()
    print("[slide / vertical drift]")
    print(f"  z after settle:                    {z_after_settle:.5f} m")
    print(f"  z @ end:                           {z_history[-1]:.5f} m")
    print(f"  total drop over run:               {drop_total * 1000:+.2f} mm")
    print(f"  noslip_iterations:                 {args.noslip_iterations}")
    print()


def _build_scene_and_run_torque(mu_t: float, enable_torsional: bool, n_steps: int, torque_ramp_max: float):
    """Headless single-run helper for torque mode. Returns (omegas, applied_torques)."""
    sphere_radius = 0.15
    sphere_center_z = 0.30
    plate_thickness = 0.02
    overlap = 0.002
    sphere_density = 100.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=enable_torsional,
            noslip_iterations=10,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=sphere_radius, pos=(0.0, 0.0, sphere_center_z)),
        material=gs.materials.Rigid(rho=sphere_density, friction=1.5, friction_torsional=mu_t),
    )
    plate_y = sphere_radius + 0.5 * plate_thickness - overlap
    for sign in (-1.0, 1.0):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.4, plate_thickness, 0.4),
                pos=(0.0, sign * plate_y, sphere_center_z),
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=1.5, friction_torsional=mu_t),
        )
    scene.build()
    rigid = scene.sim.rigid_solver
    sphere_link_idx = sphere.base_link_idx

    for _ in range(80):
        scene.step()

    omegas = []
    applied = []
    for i in range(n_steps):
        tau = torque_ramp_max * (i + 1) / n_steps
        rigid.apply_links_external_torque(np.array([0.0, tau, 0.0]), links_idx=[sphere_link_idx])
        scene.step()
        omegas.append(float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[4]))
        applied.append(tau)
    return omegas, applied


def _run_torque_sweep(use_cpu: bool, n_steps: int, torque_ramp_max: float):
    gs.init(backend=gs.cpu if use_cpu else gs.gpu, logging_level="warning")

    mu_t_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
    print(f"Running torque-ramp sweep: 0 → {torque_ramp_max} N*m over {n_steps} steps for each μ_t...")
    print()
    rows = []
    for mu_t in mu_t_values:
        enable = mu_t > 0.0
        omegas, applied = _build_scene_and_run_torque(
            mu_t=mu_t, enable_torsional=enable, n_steps=n_steps, torque_ramp_max=torque_ramp_max
        )
        breakaway_step = next((i for i, w in enumerate(omegas) if abs(w) > 0.5), None)
        breakaway_torque = applied[breakaway_step] if breakaway_step is not None else None
        peak = max(abs(w) for w in omegas)
        rows.append({"mu_t": mu_t, "breakaway": breakaway_torque, "peak_omega": peak, "final_omega": omegas[-1]})

    header = f"{'mu_t':>8}  {'breakaway τ (N*m)':>20}  {'peak |ω|':>10}  {'final ω':>10}"
    print(header)
    print("-" * len(header))
    for r in rows:
        bt = f"{r['breakaway']:.3f}" if r["breakaway"] is not None else f"> {torque_ramp_max:.2f}"
        print(f"{r['mu_t']:>8.3f}  {bt:>20}  {r['peak_omega']:>10.3f}  {r['final_omega']:>10.3f}")
    print()
    print(
        "Breakaway torque = applied torque at which |omega| first exceeds 0.5 rad/s. "
        "Larger μ_t should require more torque to break stiction."
    )
    print("  μ_t = 0  → no friction at all, sphere accelerates immediately at any torque.")


def _build_scene_and_run(mu_t: float, enable_torsional: bool, n_steps: int, initial_spin: float):
    """Headless single-run helper. Returns (omega_history, z_drop_total)."""
    sphere_radius = 0.15
    sphere_center_z = 0.30
    plate_thickness = 0.02
    overlap = 0.002
    sphere_density = 100.0

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.005),
        rigid_options=gs.options.RigidOptions(
            enable_torsional_friction=enable_torsional,
            noslip_iterations=10,
        ),
        show_viewer=False,
    )
    scene.add_entity(gs.morphs.Plane())
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=sphere_radius, pos=(0.0, 0.0, sphere_center_z)),
        material=gs.materials.Rigid(rho=sphere_density, friction=1.5, friction_torsional=mu_t),
        surface=gs.surfaces.Plastic(color=(1.0, 0.3, 0.3)),
    )
    plate_y = sphere_radius + 0.5 * plate_thickness - overlap
    for sign in (-1.0, 1.0):
        scene.add_entity(
            gs.morphs.Box(
                size=(0.3, plate_thickness, 0.3),
                pos=(0.0, sign * plate_y, sphere_center_z),
                fixed=True,
            ),
            material=gs.materials.Rigid(friction=1.5, friction_torsional=mu_t),
        )
    scene.build()

    for _ in range(80):
        scene.step()
    z_after_settle = float(sphere.get_pos().cpu().numpy().squeeze()[2])

    sphere_dofs_vel = np.zeros(6)
    sphere_dofs_vel[4] = initial_spin
    sphere.set_dofs_velocity(velocity=sphere_dofs_vel)

    omegas = []
    for _ in range(n_steps):
        scene.step()
        omegas.append(float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[4]))

    z_end = float(sphere.get_pos().cpu().numpy().squeeze()[2])
    return omegas, (z_after_settle - z_end)


def _time_to_threshold(omegas, dt: float, threshold: float):
    """Return wall-clock time (seconds) at which |omega| first drops below threshold,
    or None if it never reaches the threshold."""
    for i, w in enumerate(omegas):
        if abs(w) < threshold:
            return (i + 1) * dt
    return None


def _run_sweep(use_cpu: bool, n_steps: int, initial_spin: float):
    gs.init(backend=gs.cpu if use_cpu else gs.gpu, logging_level="warning")

    # Match Genesis' slide-friction stability floor: clamp mu_t to >= 1e-2 at the contact
    # merge step (see collider/contact.py). Values below that are reserved for "off".
    mu_t_values = [0.0, 0.01, 0.02, 0.05, 0.1, 0.3, 1.0]
    dt = 0.005
    rows = []
    print("Running sweep over torsional friction coefficients...")
    print()
    for mu_t in mu_t_values:
        enable = mu_t > 0.0
        omegas, drop = _build_scene_and_run(
            mu_t=mu_t, enable_torsional=enable, n_steps=n_steps, initial_spin=initial_spin
        )
        rows.append(
            {
                "mu_t": mu_t,
                "omega_initial": initial_spin,
                "omega_step50": omegas[49] if len(omegas) > 49 else float("nan"),
                "omega_step200": omegas[199] if len(omegas) > 199 else float("nan"),
                "omega_final": omegas[-1],
                "decay_ratio": abs(omegas[-1]) / max(abs(initial_spin), 1e-9),
                "t_to_half": _time_to_threshold(omegas, dt, abs(initial_spin) * 0.5),
                "t_to_tenth": _time_to_threshold(omegas, dt, abs(initial_spin) * 0.1),
                "t_to_001": _time_to_threshold(omegas, dt, 0.01),
            }
        )

    print()
    header = (
        f"{'mu_t':>8}  {'omega@50':>10}  {'omega@200':>10}  {'omega_end':>10}  "
        f"{'decay':>7}  {'t_50%':>7}  {'t_10%':>7}  {'t<0.01':>7}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        t50 = f"{r['t_to_half']:.3f}s" if r["t_to_half"] is not None else "  -- "
        t10 = f"{r['t_to_tenth']:.3f}s" if r["t_to_tenth"] is not None else "  -- "
        tz = f"{r['t_to_001']:.3f}s" if r["t_to_001"] is not None else "  -- "
        print(
            f"{r['mu_t']:>8.3f}  {r['omega_step50']:>10.3f}  {r['omega_step200']:>10.3f}  "
            f"{r['omega_final']:>10.3f}  {r['decay_ratio']:>7.3f}  {t50:>7}  {t10:>7}  {tz:>7}"
        )
    print()
    print("Legend: t_50% / t_10% / t<0.01 = simulated time at which |omega_y| first drops")
    print("        below 50% / 10% / 0.01 rad/s of its initial value. '--' means it never did.")
    print(f"        (n_steps = {n_steps}, total simulated time = {n_steps * dt:.2f}s, dt = {dt}s)")


if __name__ == "__main__":
    main()
