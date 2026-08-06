from __future__ import annotations

import json
import math


CAMERA_PROTOCOL_VERSION = "libero_36_agentview_v1"
AGENTVIEW_CAMERA_NAME = "agentview"
RAW_RESET_AGENTVIEW_FOVY_DEG = 45.0
AGENTVIEW_FOVY_DEG = 75.0
AGENTVIEW_POSITION_M = (
    0.6586131746834771,
    0.0,
    1.6103500240372423,
)
AGENTVIEW_QUATERNION_WXYZ = (
    0.6380177736282349,
    0.3048497438430786,
    0.30484986305236816,
    0.6380177736282349,
)
CAMERA_HEIGHT = 128
CAMERA_WIDTH = 128

CAMERA_VECTOR_ATOL = 1e-8
CAMERA_FOVY_ATOL = 1e-9

OBSERVATION_CAMERA_SPEC_FIELD = "observation_camera_spec_json"


class FrozenCameraMismatch(RuntimeError):
    pass


def frozen_camera_spec() -> dict:
    return {
        "camera_protocol_version": CAMERA_PROTOCOL_VERSION,
        "agentview_camera_name": AGENTVIEW_CAMERA_NAME,
        "raw_reset_agentview_fovy_deg": RAW_RESET_AGENTVIEW_FOVY_DEG,
        "agentview_fovy_deg": AGENTVIEW_FOVY_DEG,
        "agentview_position_m": list(AGENTVIEW_POSITION_M),
        "agentview_quaternion_wxyz": list(
            AGENTVIEW_QUATERNION_WXYZ
        ),
        "camera_height": CAMERA_HEIGHT,
        "camera_width": CAMERA_WIDTH,
        "apply_after_every_reset": True,
        "observation_refresh_after_apply": True,
    }


def frozen_camera_spec_json() -> str:
    return json.dumps(
        frozen_camera_spec(),
        sort_keys=True,
        separators=(",", ":"),
    )


def _vector_close(
    actual,
    expected: tuple[float, ...],
    *,
    allow_quaternion_sign: bool = False,
) -> bool:
    values = tuple(float(value) for value in actual)
    targets = tuple(float(value) for value in expected)
    if len(values) != len(targets):
        return False

    if allow_quaternion_sign:
        values_norm = math.sqrt(
            sum(value * value for value in values)
        )
        targets_norm = math.sqrt(
            sum(value * value for value in targets)
        )
        if values_norm <= 1e-12 or targets_norm <= 1e-12:
            return False

        values = tuple(
            value / values_norm for value in values
        )
        targets = tuple(
            value / targets_norm for value in targets
        )

    def close(candidate) -> bool:
        return all(
            math.isclose(
                value,
                target,
                rel_tol=0.0,
                abs_tol=CAMERA_VECTOR_ATOL,
            )
            for value, target in zip(candidate, targets)
        )

    return close(values) or (
        allow_quaternion_sign
        and close(tuple(-value for value in values))
    )


def validate_frozen_camera_spec_row(row: dict) -> None:
    if OBSERVATION_CAMERA_SPEC_FIELD not in row:
        raise ValueError(
            f"Camera spec field is missing: {OBSERVATION_CAMERA_SPEC_FIELD}"
        )
    spec = json.loads(row[OBSERVATION_CAMERA_SPEC_FIELD])
    expected_keys = set(frozen_camera_spec())
    if set(spec) != expected_keys:
        raise ValueError(
            "Camera spec keys changed: "
            f"{sorted(set(spec) ^ expected_keys)}"
        )
    if spec["camera_protocol_version"] != CAMERA_PROTOCOL_VERSION:
        raise ValueError("Camera protocol version changed")
    if spec["agentview_camera_name"] != AGENTVIEW_CAMERA_NAME:
        raise ValueError("Agentview camera name changed")
    if not math.isclose(
        float(spec["raw_reset_agentview_fovy_deg"]),
        RAW_RESET_AGENTVIEW_FOVY_DEG,
        rel_tol=0.0,
        abs_tol=CAMERA_FOVY_ATOL,
    ):
        raise ValueError("Raw-reset agentview field of view changed")
    if not math.isclose(
        float(spec["agentview_fovy_deg"]),
        AGENTVIEW_FOVY_DEG,
        rel_tol=0.0,
        abs_tol=CAMERA_FOVY_ATOL,
    ):
        raise ValueError("Agentview vertical field of view changed")
    if not _vector_close(
        spec["agentview_position_m"],
        AGENTVIEW_POSITION_M,
    ):
        raise ValueError("Agentview camera position changed")
    if not _vector_close(
        spec["agentview_quaternion_wxyz"],
        AGENTVIEW_QUATERNION_WXYZ,
        allow_quaternion_sign=True,
    ):
        raise ValueError("Agentview camera quaternion changed")
    if spec["camera_height"] != CAMERA_HEIGHT:
        raise ValueError("Camera height changed")
    if spec["camera_width"] != CAMERA_WIDTH:
        raise ValueError("Camera width changed")
    if spec["apply_after_every_reset"] is not True:
        raise ValueError("Camera reset application contract changed")
    if spec["observation_refresh_after_apply"] is not True:
        raise ValueError("Camera observation refresh contract changed")


def _base_environment(env):
    return getattr(env, "env", env)


def _simulation(env):
    simulation = getattr(env, "sim", None)
    if simulation is None:
        simulation = getattr(_base_environment(env), "sim", None)
    if simulation is None:
        raise FrozenCameraMismatch("Environment has no MuJoCo simulation")
    return simulation


def frozen_camera_record(env, pre_apply_fovy_deg: float) -> dict:
    simulation = _simulation(env)
    camera_id = simulation.model.camera_name2id(
        AGENTVIEW_CAMERA_NAME
    )
    return {
        "camera_protocol_version": CAMERA_PROTOCOL_VERSION,
        "camera_id": int(camera_id),
        "camera_name": AGENTVIEW_CAMERA_NAME,
        "pre_apply_fovy_deg": float(pre_apply_fovy_deg),
        "position_m": [
            float(value)
            for value in simulation.model.cam_pos[camera_id]
        ],
        "quaternion_wxyz": [
            float(value)
            for value in simulation.model.cam_quat[camera_id]
        ],
        "fovy_deg": float(simulation.model.cam_fovy[camera_id]),
        "height": CAMERA_HEIGHT,
        "width": CAMERA_WIDTH,
    }


def validate_runtime_camera_record(record: dict) -> None:
    if record["camera_protocol_version"] != CAMERA_PROTOCOL_VERSION:
        raise FrozenCameraMismatch("Camera protocol version mismatch")
    if record["camera_name"] != AGENTVIEW_CAMERA_NAME:
        raise FrozenCameraMismatch("Agentview camera name mismatch")
    if not math.isclose(
        float(record["pre_apply_fovy_deg"]),
        RAW_RESET_AGENTVIEW_FOVY_DEG,
        rel_tol=0.0,
        abs_tol=CAMERA_FOVY_ATOL,
    ):
        raise FrozenCameraMismatch(
            "Raw-reset agentview FOV differs from the frozen default: "
            f"{record['pre_apply_fovy_deg']}"
        )
    if not _vector_close(
        record["position_m"],
        AGENTVIEW_POSITION_M,
    ):
        raise FrozenCameraMismatch(
            "Agentview position differs from the frozen protocol: "
            f"{record['position_m']}"
        )
    if not _vector_close(
        record["quaternion_wxyz"],
        AGENTVIEW_QUATERNION_WXYZ,
        allow_quaternion_sign=True,
    ):
        raise FrozenCameraMismatch(
            "Agentview quaternion differs from the frozen protocol: "
            f"{record['quaternion_wxyz']}"
        )
    if not math.isclose(
        float(record["fovy_deg"]),
        AGENTVIEW_FOVY_DEG,
        rel_tol=0.0,
        abs_tol=CAMERA_FOVY_ATOL,
    ):
        raise FrozenCameraMismatch(
            "Agentview FOV differs from the frozen protocol: "
            f"{record['fovy_deg']}"
        )
    if record["height"] != CAMERA_HEIGHT:
        raise FrozenCameraMismatch("Camera height mismatch")
    if record["width"] != CAMERA_WIDTH:
        raise FrozenCameraMismatch("Camera width mismatch")


def apply_frozen_agentview_camera(env):
    """Apply and verify the v4 camera after every environment reset."""

    simulation = _simulation(env)
    camera_id = simulation.model.camera_name2id(
        AGENTVIEW_CAMERA_NAME
    )

    # A hard reset reconstructs the MuJoCo model. Fail on a changed
    # camera pose rather than silently hiding an environment mismatch.
    raw_position = simulation.model.cam_pos[camera_id]
    raw_quaternion = simulation.model.cam_quat[camera_id]
    raw_fovy = float(simulation.model.cam_fovy[camera_id])
    if not _vector_close(raw_position, AGENTVIEW_POSITION_M):
        raise FrozenCameraMismatch(
            "Raw-reset agentview position changed: "
            f"{[float(value) for value in raw_position]}"
        )
    if not _vector_close(
        raw_quaternion,
        AGENTVIEW_QUATERNION_WXYZ,
        allow_quaternion_sign=True,
    ):
        raise FrozenCameraMismatch(
            "Raw-reset agentview quaternion changed: "
            f"{[float(value) for value in raw_quaternion]}"
        )
    if not math.isclose(
        raw_fovy,
        RAW_RESET_AGENTVIEW_FOVY_DEG,
        rel_tol=0.0,
        abs_tol=CAMERA_FOVY_ATOL,
    ):
        raise FrozenCameraMismatch(
            "Raw-reset agentview FOV changed: "
            f"{raw_fovy}"
        )

    simulation.model.cam_fovy[camera_id] = AGENTVIEW_FOVY_DEG
    simulation.forward()
    observation = _base_environment(env)._get_observations(
        force_update=True
    )
    for key in ["agentview_image", "robot0_eye_in_hand_image"]:
        image = observation.get(key)
        if getattr(image, "shape", None) != (
            CAMERA_HEIGHT,
            CAMERA_WIDTH,
            3,
        ):
            raise FrozenCameraMismatch(
                f"Unexpected refreshed {key} shape: "
                f"{getattr(image, 'shape', None)}"
            )
    record = frozen_camera_record(env, raw_fovy)
    validate_runtime_camera_record(record)
    return observation, record
