# Copyright (c) Meta Platforms, Inc. and affiliates.
# Adapted from SAM 3D Body under SAM_LICENSE.txt.
# NumPy-only conversion; no network/model loading or inference side effects.
import numpy as np

_HAND_DOFS_IN_ORDER = np.asarray(
    [3, 1, 1, 3, 1, 1, 3, 1, 1, 3, 1, 1, 2, 3, 1, 1], dtype=np.int64
)
_BODY_3DOF_ROT_IDXS = np.asarray(
    [
        (0, 2, 4),
        (6, 8, 10),
        (12, 13, 14),
        (15, 16, 17),
        (18, 19, 20),
        (21, 22, 23),
        (24, 25, 26),
        (27, 28, 29),
        (34, 35, 36),
        (37, 38, 39),
        (44, 45, 46),
        (53, 54, 55),
        (64, 65, 66),
        (85, 69, 73),
        (86, 70, 79),
        (87, 71, 82),
        (88, 72, 76),
        (91, 92, 93),
        (112, 96, 100),
        (113, 97, 106),
        (114, 98, 109),
        (115, 99, 103),
        (130, 131, 132),
    ],
    dtype=np.int64,
)
_BODY_1DOF_ROT_IDXS = np.asarray(
    [
        1,
        3,
        5,
        7,
        9,
        11,
        30,
        31,
        32,
        33,
        40,
        41,
        42,
        43,
        47,
        48,
        49,
        50,
        51,
        52,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        67,
        68,
        74,
        75,
        77,
        78,
        80,
        81,
        83,
        84,
        89,
        90,
        94,
        95,
        101,
        102,
        104,
        105,
        107,
        108,
        110,
        111,
        116,
        117,
        118,
        119,
        120,
        121,
        122,
        123,
    ],
    dtype=np.int64,
)
_BODY_1DOF_TRANS_IDXS = np.asarray([124, 125, 126, 127, 128, 129], dtype=np.int64)


def xyz_from_columns6d(value):
    """SAM body/hand layout: first column followed by second column."""
    value = np.asarray(value, dtype=np.float64)
    x = value[..., :3]
    x_norm = np.linalg.norm(x, axis=-1, keepdims=True)
    z = np.cross(x / np.maximum(x_norm, 1e-12), value[..., 3:])
    z_norm = np.linalg.norm(z, axis=-1, keepdims=True)
    if np.any(x_norm < 1e-10) or np.any(z_norm < 1e-10):
        raise ValueError("Degenerate body/hand 6D rotation")
    x = x / x_norm
    z = z / z_norm
    y = np.cross(z, x)
    matrix = np.stack((x, y, z), axis=-1)
    sy = np.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2)
    singular = sy < 1e-6
    rx = np.where(
        singular,
        np.arctan2(-matrix[..., 1, 2], matrix[..., 1, 1]),
        np.arctan2(matrix[..., 2, 1], matrix[..., 2, 2]),
    )
    ry = np.arctan2(-matrix[..., 2, 0], sy)
    rz = np.where(singular, 0.0, np.arctan2(matrix[..., 1, 0], matrix[..., 0, 0]))
    return np.stack((rx, ry, rz), axis=-1)


def compact_cont_to_model_params_body(value):
    value = np.asarray(value)
    if value.shape[-1] != 260:
        raise ValueError("Expected 260 continuous body parameters")
    result = np.zeros((*value.shape[:-1], 133), dtype=np.float64)
    result[..., _BODY_3DOF_ROT_IDXS.reshape(-1)] = xyz_from_columns6d(
        value[..., :138].reshape(*value.shape[:-1], 23, 6)
    ).reshape(*value.shape[:-1], 69)
    pairs = value[..., 138:254].reshape(*value.shape[:-1], 58, 2)
    if np.any(np.linalg.norm(pairs, axis=-1) < 1e-10):
        raise ValueError("Degenerate body sine/cosine rotation")
    result[..., _BODY_1DOF_ROT_IDXS] = np.arctan2(pairs[..., 0], pairs[..., 1])
    result[..., _BODY_1DOF_TRANS_IDXS] = value[..., 254:]
    return result


def compact_cont_to_model_params_hand(value):
    value = np.asarray(value)
    if value.shape[-1] != 54:
        raise ValueError("Expected 54 continuous hand parameters")
    cont3 = np.concatenate([np.full(2 * k, k == 3) for k in _HAND_DOFS_IN_ORDER])
    model3 = np.concatenate([np.full(k, k == 3) for k in _HAND_DOFS_IN_ORDER])
    result = np.zeros((*value.shape[:-1], 27), dtype=np.float64)
    result[..., model3] = xyz_from_columns6d(
        value[..., cont3].reshape(*value.shape[:-1], -1, 6)
    ).reshape(*value.shape[:-1], -1)
    pairs = value[..., ~cont3].reshape(*value.shape[:-1], -1, 2)
    if np.any(np.linalg.norm(pairs, axis=-1) < 1e-10):
        raise ValueError("Degenerate hand sine/cosine rotation")
    result[..., ~model3] = np.arctan2(pairs[..., 0], pairs[..., 1])
    return result
