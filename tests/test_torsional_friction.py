"""Unit tests for torsional (spinning) friction in the rigid solver.

Torsional friction resists relative angular velocity about the contact normal.
With ``mu_torsional > 0``, a body that pivots about a contact point should
decelerate; with ``mu_torsional == 0`` (default), spinning is unimpeded.
"""

import numpy as np
import pytest

import genesis as gs
import genesis.utils.geom as gu


pytestmark = pytest.mark.parametrize("backend", [gs.cpu])


def _build_spinning_sphere_scene(mu_t_plane: float, mu_t_sphere: float, dt: float = 0.005):
    """Sphere on plane (single point contact). Slide friction does not couple to spin
    at a point contact, so spin decay is purely from torsional friction."""
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(dt=dt, gravity=(0.0, 0.0, -9.81), enable_torsional_friction=True),
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=1.0, friction_torsional=mu_t_plane),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=0.1, pos=(0.0, 0.0, 0.1)),
        material=gs.materials.Rigid(friction=1.0, friction_torsional=mu_t_sphere),
    )
    scene.build()
    return scene, sphere


@pytest.mark.required
def test_torsional_friction_zero_preserves_spin(backend):
    """μ_torsional = 0 (default behavior): spinning sphere on plane keeps spinning forever."""
    scene, sphere = _build_spinning_sphere_scene(mu_t_plane=0.0, mu_t_sphere=0.0)
    sphere.set_dofs_velocity(velocity=np.array([0, 0, 0, 0, 0, 10.0]))

    for _ in range(200):
        scene.step()

    omega_z = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[5])
    assert abs(omega_z - 10.0) < 0.05, f"Expected spin to be preserved, got omega_z={omega_z}"


@pytest.mark.required
def test_torsional_friction_decays_spin(backend):
    """μ_torsional > 0: spin should monotonically decrease and eventually stop."""
    scene, sphere = _build_spinning_sphere_scene(mu_t_plane=0.05, mu_t_sphere=0.05)
    sphere.set_dofs_velocity(velocity=np.array([0, 0, 0, 0, 0, 10.0]))

    omega_history = []
    for _ in range(200):
        scene.step()
        omega_history.append(float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[5]))

    omega_initial = 10.0
    omega_mid = omega_history[50]
    omega_final = omega_history[-1]

    assert omega_mid < omega_initial, "Spin should have decreased after 50 steps"
    assert abs(omega_final) < 0.5, f"Spin should be near zero at end, got {omega_final}"
    assert omega_final < omega_mid + 1e-3, "Spin should not increase over time (modulo noise)"


def test_torsional_friction_strength_ordering(backend):
    """Higher μ_torsional should stop the sphere faster (or at least no slower)."""
    durations = []
    for mu_t in (0.005, 0.05, 0.5):
        scene, sphere = _build_spinning_sphere_scene(mu_t_plane=mu_t, mu_t_sphere=mu_t)
        sphere.set_dofs_velocity(velocity=np.array([0, 0, 0, 0, 0, 10.0]))

        # Number of steps until |omega_z| drops below 0.1
        n_steps_to_stop = None
        for step in range(500):
            scene.step()
            omega_z = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[5])
            if abs(omega_z) < 0.1:
                n_steps_to_stop = step
                break
        durations.append(n_steps_to_stop if n_steps_to_stop is not None else 500)

    # Larger mu_t => fewer steps to stop (allow ties since very large mu_t can saturate within 1 step)
    assert durations[1] <= durations[0], f"μ_t=0.05 should stop no slower than μ_t=0.005, got {durations}"
    assert durations[2] <= durations[1], f"μ_t=0.5 should stop no slower than μ_t=0.05, got {durations}"


@pytest.mark.required
def test_torsional_friction_default_zero_via_material(backend):
    """Material default (no friction_torsional specified) should be 0 for backward compatibility."""
    mat = gs.materials.Rigid(friction=1.0)
    assert mat.friction_torsional is None  # resolves to 0 at geom build via default_friction_torsional()


def test_torsional_friction_per_component_max_merge(backend):
    """Per-contact friction should take per-component max of the two geoms' friction_torsional."""
    scene = gs.Scene(
        show_viewer=False,
        rigid_options=gs.options.RigidOptions(dt=0.005, gravity=(0.0, 0.0, -9.81), enable_torsional_friction=True),
    )
    scene.add_entity(
        gs.morphs.Plane(),
        material=gs.materials.Rigid(friction=1.0, friction_torsional=0.001),
    )
    sphere = scene.add_entity(
        gs.morphs.Sphere(radius=0.1, pos=(0.0, 0.0, 0.1)),
        material=gs.materials.Rigid(friction=1.0, friction_torsional=0.05),
    )
    scene.build()

    # Verify geom-level storage matches what was set.
    plane_idx = scene.entities[0].geoms[0].idx
    sphere_idx = sphere.geoms[0].idx
    plane_mu_t = scene.sim.rigid_solver.get_geoms_friction_torsional(plane_idx).item()
    sphere_mu_t = scene.sim.rigid_solver.get_geoms_friction_torsional(sphere_idx).item()
    assert abs(plane_mu_t - 0.001) < 1e-6
    assert abs(sphere_mu_t - 0.05) < 1e-6

    # The merged contact should use max(0.001, 0.05) = 0.05, dominating decay.
    sphere.set_dofs_velocity(velocity=np.array([0, 0, 0, 0, 0, 10.0]))
    for _ in range(150):
        scene.step()
    omega_final = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[5])
    # If max-merge worked, mu_t=0.05 dominates and spin stops; if min-merge, mu_t=0.001 leaves substantial spin.
    assert abs(omega_final) < 1.0, f"Per-component max merge failed: spin still {omega_final} after 150 steps"


def test_torsional_friction_setter_runtime(backend):
    """Verify per-geom torsional friction can be changed at runtime."""
    scene, sphere = _build_spinning_sphere_scene(mu_t_plane=0.0, mu_t_sphere=0.0)
    sphere.set_dofs_velocity(velocity=np.array([0, 0, 0, 0, 0, 10.0]))
    # spin freely for 30 steps
    for _ in range(30):
        scene.step()
    omega_before = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[5])
    assert abs(omega_before - 10.0) < 0.1

    # Crank up torsional friction at runtime
    sphere.geoms[0].set_friction_torsional(0.5)
    scene.entities[0].geoms[0].set_friction_torsional(0.5)
    for _ in range(50):
        scene.step()

    omega_after = float(sphere.get_dofs_velocity().cpu().numpy().squeeze()[5])
    assert abs(omega_after) < 1.0, f"Expected spin to stop after runtime setter, got {omega_after}"
