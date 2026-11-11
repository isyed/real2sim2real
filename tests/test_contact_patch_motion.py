import numpy as np
import pytest

from real2sim2real.controller_actor_training.contact_patch_motion import (
    ContactPatchMotion,
    relative_motion,
    summarize,
)


def settings():
    return dict(
        minimum_force_n=0.1,
        sliding_speed_m_s=0.001,
        rolling_speed_rad_s=0.02,
        maximum_sample_gap_s=0.05,
    )


def test_material_slip_is_invariant_to_common_motion_and_excludes_normal_motion():
    v, rolling = relative_motion(
        [0.02, 0, 0.1], [0.01, 0, 0], [0, 0.1, 0], [0, 0, 0], [0, 0, 1]
    )
    np.testing.assert_allclose(v, [0.01, 0, 0])
    assert rolling == 0.1
    v, rolling = relative_motion(
        [0.02, 0.03, 0.04], [0.02, 0.03, 0.04], [0, 0.1, 0], [0, 0.1, 0], [0, 0, 1]
    )
    np.testing.assert_array_equal(v, [0, 0, 0])
    assert rolling == 0


def test_centroid_migration_is_not_slip_and_gaps_are_not_integrated():
    def row(time, x):
        return dict(
            phase="hold",
            time_s=time,
            contact_patch_motion=dict(
                pads={
                    "pad": dict(
                        loaded=True,
                        force_n=2.0,
                        tangent_speed_m_s=0.0,
                        sliding=False,
                        rolling_compatible=True,
                        object_local_centroid_m=[x, 0, 0],
                        object_collision_geoms=[1],
                    )
                }
            ),
        )

    report = summarize([row(0, 0), row(0.02, 0.002), row(0.2, 0.1)], settings())
    p = report["pads"]["pad"]
    assert p["sampled_sliding_path_m"] == 0
    assert p["object_centroid_migration_m"] == 0.002
    assert p["rolling_compatible_fraction_of_loaded"] == 1
    assert report["centroid_migration_is_material_slip"] is False


def test_solved_contact_velocity_logging_preserves_physical_state():
    mj = pytest.importorskip("mujoco")
    m = mj.MjModel.from_xml_string("""<mujoco><worldbody>
    <body name="hand"><joint name="slide" type="slide" axis="1 0 0"/><geom name="pad" type="box" size=".1 .1 .01" mass="1"/></body>
    <body name="object" pos="0 0 .03"><freejoint/><geom size=".01" mass=".2"/></body>
    </worldbody></mujoco>""")
    d = mj.MjData(m)
    for _ in range(100):
        mj.mj_step(m, d)
    gid = m.geom("pad").id
    audit = ContactPatchMotion(
        m,
        m.body("object").id,
        {"primary": {gid}},
        [dict(gid=gid, role="primary", group="thumb")],
        settings(),
    )
    obj = int(m.body_jntadr[m.body("object").id])
    dof = int(m.jnt_dofadr[obj])
    d.qvel[:] = 0
    d.qvel[dof] = 0.02
    mj.mj_forward(m, d)
    before = d.qpos.copy(), d.qvel.copy(), m.jnt_range.copy(), d.qfrc_applied.copy()
    r = audit.sample(m, d)["pads"]["primary:thumb"]
    assert r["loaded"] and r["sliding"]
    assert r["tangent_speed_m_s"] == pytest.approx(0.02, abs=1e-5)
    for got, want in zip([d.qpos, d.qvel, m.jnt_range, d.qfrc_applied], before):
        np.testing.assert_array_equal(got, want)
    d.qvel[0] = 0.02
    mj.mj_forward(m, d)
    assert audit.sample(m, d)["pads"]["primary:thumb"]["tangent_speed_m_s"] < 1e-8


def test_contact_roles_cannot_overlap_or_relabel_a_pad():
    from types import SimpleNamespace

    model = SimpleNamespace(geom=lambda g: SimpleNamespace(name=f"geom-{g}"))
    with pytest.raises(ValueError, match="Disjoint"):
        ContactPatchMotion(
            model,
            3,
            {"primary": {1}, "secondary": {1}},
            [dict(gid=1, role="primary", group="thumb")],
            settings(),
        )
    with pytest.raises(ValueError, match="Disjoint"):
        ContactPatchMotion(
            model,
            3,
            {"primary": {1, 2}},
            [
                dict(gid=1, role="primary", group="thumb"),
                dict(gid=2, role="primary", group="thumb"),
            ],
            settings(),
        )
