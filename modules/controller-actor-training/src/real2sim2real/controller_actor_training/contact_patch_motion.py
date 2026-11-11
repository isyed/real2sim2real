"""Read-only solved-contact motion at coherent simulator sample states.

Tangential material velocity diagnoses sliding. Low slip plus relative angular
motion is rolling-compatible, not proof of persistent material correspondence.
Force-weighted patch-centroid migration is kept separate from material slip.
"""

import numpy as np


def validate_settings(settings):
    caps = dict(
        minimum_force_n=1,
        sliding_speed_m_s=0.02,
        rolling_speed_rad_s=1,
        maximum_sample_gap_s=0.1,
    )
    if set(settings) != set(caps) or any(
        not np.isfinite(settings[k]) or not 0 < settings[k] <= cap
        for k, cap in caps.items()
    ):
        raise ValueError(
            "Explicit finite contact-motion diagnostic thresholds required"
        )


def relative_motion(hand_velocity, object_velocity, hand_omega, object_omega, normal):
    v, w, n = (
        np.asarray(hand_velocity, float) - np.asarray(object_velocity, float),
        np.asarray(hand_omega, float) - np.asarray(object_omega, float),
        np.asarray(normal, float),
    )
    if any(
        a.shape != (3,) or not np.isfinite(a).all() for a in (v, w, n)
    ) or not np.isclose(np.linalg.norm(n), 1):
        raise ValueError(
            "Finite world point velocities and a unit contact normal required"
        )
    tangent = v - (v @ n) * n
    return tangent, float(np.linalg.norm(w - (w @ n) * n))


class ContactPatchMotion:
    def __init__(self, model, object_id, role_geoms, pads, settings):
        validate_settings(settings)
        self.settings = settings
        self.object_id = int(object_id)
        self.roles = {int(g): role for role, geoms in role_geoms.items() for g in geoms}
        self.labels = {int(p["gid"]): p["role"] + ":" + p["group"] for p in pads}
        if len(self.labels) != len(pads) or not set(self.labels) <= set(self.roles):
            raise ValueError("Distinct owned collision pads required for contact audit")
        if (
            sum(len(g) for g in role_geoms.values()) != len(self.roles)
            or len(set(self.labels.values())) != len(pads)
            or any(self.roles[int(p["gid"])] != p["role"] for p in pads)
        ):
            raise ValueError("Disjoint hand ownership and distinct pad labels required")
        self.names = {g: model.geom(g).name or f"geom-{g}" for g in self.roles}

    def sample(self, model, data):
        """Caller must provide coherent position/velocity/contact/force fields."""
        import mujoco

        grouped = {label: [] for label in self.labels.values()}
        other = {}
        obj = self.object_id
        ro = data.xmat[obj].reshape(3, 3)
        for i, contact in enumerate(data.contact):
            for pad, other_geom, sign in (
                (int(contact.geom1), int(contact.geom2), -1),
                (int(contact.geom2), int(contact.geom1), 1),
            ):
                if pad not in self.roles or model.geom_bodyid[other_geom] != obj:
                    continue
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, force)
                if not np.isfinite(force).all():
                    raise ValueError("Nonfinite solved contact force")
                fn = max(0.0, float(force[0]))
                if pad not in self.labels:
                    name = self.roles[pad] + ":" + self.names[pad]
                    other[name] = other.get(name, 0.0) + fn
                    continue
                if fn <= 0:
                    continue
                body = int(model.geom_bodyid[pad])
                rh = data.xmat[body].reshape(3, 3)
                jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
                mujoco.mj_jac(model, data, jacp, jacr, contact.pos, body)
                vh, wh = jacp @ data.qvel, jacr @ data.qvel
                mujoco.mj_jac(model, data, jacp, jacr, contact.pos, obj)
                vo, wo = jacp @ data.qvel, jacr @ data.qvel
                normal = sign * np.asarray(contact.frame).reshape(3, 3)[0]
                tangent, rolling = relative_motion(vh, vo, wh, wo, normal)
                mu = float(min(contact.friction[:2]))
                grouped[self.labels[pad]].append(
                    dict(
                        force=fn,
                        speed=float(np.linalg.norm(tangent)),
                        tangent=tangent,
                        rolling=rolling,
                        object_point=ro.T @ (contact.pos - data.xpos[obj]),
                        pad_point=rh.T @ (contact.pos - data.xpos[body]),
                        linear_ratio=float(np.linalg.norm(force[1:3]) / (mu * fn))
                        if mu > 0
                        else None,
                        object_geom=other_geom,
                    )
                )
        result = {}
        for label, contacts in grouped.items():
            total = sum(c["force"] for c in contacts)
            row = dict(force_n=total, loaded=total >= self.settings["minimum_force_n"])
            if row["loaded"]:
                average = lambda key: sum(c["force"] * c[key] for c in contacts) / total
                speed, rolling = float(average("speed")), float(average("rolling"))
                ratios = [
                    c["linear_ratio"] for c in contacts if c["linear_ratio"] is not None
                ]
                row.update(
                    tangent_speed_m_s=speed,
                    tangent_velocity_world_m_s=average("tangent").tolist(),
                    relative_tangent_angular_speed_rad_s=rolling,
                    object_local_centroid_m=average("object_point").tolist(),
                    pad_local_centroid_m=average("pad_point").tolist(),
                    maximum_linear_friction_ratio=max(ratios, default=None),
                    object_collision_geoms=sorted({c["object_geom"] for c in contacts}),
                    sliding=speed > self.settings["sliding_speed_m_s"],
                    rolling_compatible=(
                        speed <= self.settings["sliding_speed_m_s"]
                        and rolling >= self.settings["rolling_speed_rad_s"]
                    ),
                )
            result[label] = row
        return dict(pads=result, other_object_contacts_n=other)


def summarize(records, settings):
    validate_settings(settings)
    hold = [r for r in records if r["phase"] == "hold"]
    result = {}
    labels = (
        set().union(*(r["contact_patch_motion"]["pads"] for r in hold))
        if hold
        else set()
    )
    for label in sorted(labels):
        rows = [(r["time_s"], r["contact_patch_motion"]["pads"][label]) for r in hold]
        loaded = [r for _, r in rows if r["loaded"]]
        slip, migration, switches = 0.0, 0.0, 0
        for (ta, a), (tb, b) in zip(rows, rows[1:]):
            if (
                a["loaded"]
                and b["loaded"]
                and 0 < tb - ta <= settings["maximum_sample_gap_s"]
            ):
                slip += (
                    0.5 * (tb - ta) * (a["tangent_speed_m_s"] + b["tangent_speed_m_s"])
                )
                migration += float(
                    np.linalg.norm(
                        np.asarray(b["object_local_centroid_m"])
                        - a["object_local_centroid_m"]
                    )
                )
                switches += int(
                    a["object_collision_geoms"] != b["object_collision_geoms"]
                )
        result[label] = dict(
            loaded_fraction=len(loaded) / len(rows),
            mean_force_n=float(np.mean([r["force_n"] for _, r in rows])),
            sliding_fraction_of_loaded=float(np.mean([r["sliding"] for r in loaded]))
            if loaded
            else None,
            rolling_compatible_fraction_of_loaded=float(
                np.mean([r["rolling_compatible"] for r in loaded])
            )
            if loaded
            else None,
            tangent_speed_p95_m_s=float(
                np.percentile([r["tangent_speed_m_s"] for r in loaded], 95)
            )
            if loaded
            else None,
            sampled_sliding_path_m=slip,
            object_centroid_migration_m=migration,
            collision_piece_changes=switches,
        )
    return dict(
        pads=result,
        sampled_evidence_only=True,
        centroid_migration_is_material_slip=False,
        independent_contact_validation=False,
    )
