from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
from robosuite.utils import transform_utils as transform

from scripts import validate_libero_36_envs as reset_validator


TASK_ID = 0
LAYOUT_ID = 1
LANGUAGE = "pick up the akita black bowl and place it on the plate in the left region"
# Deterministic reset seeds screened with the unchanged formal replay.
# All five are distinct initial states and complete the same 658-step action trace.
SEEDS = (361000, 361002, 361005, 361006, 361009)
SETTLE_STEPS = 20
TERMINAL_HOLD_STEPS = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dataset(group, name, shape, dtype, image=False):
    if image:
        return group.create_dataset(name, shape=shape, dtype=dtype,
                                    chunks=(1, 128, 128, 3),
                                    compression="lzf", shuffle=True)
    return group.create_dataset(name, shape=shape, dtype=dtype,
                                compression="lzf", shuffle=True)


def row_for_task() -> dict:
    rows = reset_validator.read_layout_spec(Path("data/libero_36/layout_spec.csv"))
    rows = [r for r in rows if int(r["task_id"]) == TASK_ID and int(r["layout_id"]) == LAYOUT_ID]
    if len(rows) != 1:
        raise RuntimeError(f"expected one task-0/layout-1 row, got {len(rows)}")
    return rows[0]


def collect_episode(plan: dict, seed: int, demo_group) -> dict:
    actions = np.asarray(plan["actions"], dtype=np.float32)
    phases = np.asarray(plan["phase"]).astype(str)
    if actions.shape != (len(phases), 7):
        raise ValueError(f"invalid formal plan shapes: {actions.shape}, {phases.shape}")
    env = reset_validator.make_environment(Path(plan["bddl_path"]))
    try:
        obs, reset_attempts, camera_record = reset_validator.safe_reset(env, seed, 100)
        # The official controller's replay begins after the same deterministic passive settle.
        for _ in range(SETTLE_STEPS):
            obs, _, _, _ = env.step(np.zeros(7, dtype=np.float32))
        initial_state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
        n = actions.shape[0]
        obs_group = demo_group.create_group("obs")
        ds = {
            "actions": dataset(demo_group, "actions", (n, 7), np.float32),
            "states": dataset(demo_group, "states", (n, initial_state.size), np.float64),
            "robot_states": dataset(demo_group, "robot_states", (n, 9), np.float64),
            "rewards": dataset(demo_group, "rewards", (n,), np.float32),
            "dones": dataset(demo_group, "dones", (n,), np.uint8),
            "agentview": dataset(obs_group, "agentview_rgb", (n, 128, 128, 3), np.uint8, True),
            "wrist": dataset(obs_group, "eye_in_hand_rgb", (n, 128, 128, 3), np.uint8, True),
            "joint": dataset(obs_group, "joint_states", (n, 7), np.float64),
            "ee_pos": dataset(obs_group, "ee_pos", (n, 3), np.float64),
            "ee_ori": dataset(obs_group, "ee_ori", (n, 3), np.float64),
            "gripper": dataset(obs_group, "gripper_states", (n, 2), np.float64),
        }
        success_trace = np.zeros(n, dtype=bool)
        rewards = np.zeros(n, dtype=np.float32)
        max_action_diff = 0.0
        finite = True
        for i, action in enumerate(actions):
            state = np.asarray(env.get_sim_state(), dtype=np.float64).copy()
            agentview = np.asarray(obs["agentview_image"])
            wrist = np.asarray(obs["robot0_eye_in_hand_image"])
            joint = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)
            ee_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64)
            ee_ori = np.asarray(transform.quat2axisangle(obs["robot0_eef_quat"]), dtype=np.float64)
            gripper = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64)
            if agentview.shape != (128, 128, 3) or wrist.shape != (128, 128, 3):
                raise ValueError(f"seed {seed} frame {i}: invalid camera shape")
            ds["actions"][i] = action
            ds["states"][i] = state
            ds["robot_states"][i] = np.concatenate([joint, gripper])
            ds["joint"][i] = joint
            ds["ee_pos"][i] = ee_pos
            ds["ee_ori"][i] = ee_ori
            ds["gripper"][i] = gripper
            ds["agentview"][i] = agentview.astype(np.uint8, copy=False)
            ds["wrist"][i] = wrist.astype(np.uint8, copy=False)
            next_obs, reward, done, _ = env.step(action)
            rewards[i] = float(reward)
            ds["dones"][i] = 1 if i == n - 1 else 0
            success_trace[i] = bool(env.check_success())
            finite = finite and all(np.isfinite(x).all() for x in (state, joint, ee_pos, ee_ori, gripper, action))
            obs = next_obs
        ds["rewards"][:] = rewards
        plan_actions = np.asarray(plan["actions"], dtype=np.float32)
        max_action_diff = float(np.max(np.abs(np.asarray(ds["actions"]) - plan_actions)))
        success = bool(success_trace[-1] or np.any(rewards > 0))
        terminal_hold = bool(success_trace[-TERMINAL_HOLD_STEPS:].all())
        passed = bool(success and terminal_hold and finite and max_action_diff == 0.0)
        demo_group.attrs.update({
            "num_samples": n, "seed": seed, "task_id": TASK_ID, "layout_id": LAYOUT_ID,
            "init_state": initial_state, "reset_attempts": reset_attempts,
            "camera_record": json.dumps(camera_record, sort_keys=True),
            "source_plan": str(plan["path"]), "source_plan_sha256": sha256(Path(plan["path"])),
            "formal_action_replay_exact": max_action_diff == 0.0,
            "finite": finite, "success": success, "terminal_success_hold": terminal_hold,
            "passed": passed, "formal_gate5_evidence_modified": False,
        })
        return {"seed": seed, "frames": n, "reset_attempts": reset_attempts,
                "success": success, "terminal_success_hold": terminal_hold,
                "finite": finite, "action_max_abs_diff": max_action_diff,
                "reward_positive_steps": int(np.count_nonzero(rewards > 0)), "passed": passed}
    finally:
        env.close()


def validate(path: Path) -> dict:
    failures = []
    states = []
    frames = 0
    with h5py.File(path, "r") as handle:
        data = handle["data"]
        names = sorted(data.keys(), key=lambda x: int(x.split("_")[-1]))
        if len(names) != 5:
            failures.append(f"episodes={len(names)}")
        for name in names:
            d = data[name]; n = int(d.attrs["num_samples"]); frames += n
            states.append(np.asarray(d.attrs["init_state"]))
            required = {"actions": (n, 7), "obs/agentview_rgb": (n,128,128,3),
                        "obs/eye_in_hand_rgb": (n,128,128,3), "obs/joint_states": (n,7),
                        "obs/ee_pos": (n,3), "obs/ee_ori": (n,3), "obs/gripper_states": (n,2)}
            for key, shape in required.items():
                if key not in d or d[key].shape != shape: failures.append(f"{name}:{key}")
            if not bool(d.attrs.get("passed", False)): failures.append(f"{name}:pilot_failed")
            if np.asarray(d["obs/agentview_rgb"]).dtype != np.uint8: failures.append(f"{name}:agentview_dtype")
            if np.asarray(d["obs/eye_in_hand_rgb"]).dtype != np.uint8: failures.append(f"{name}:wrist_dtype")
        for i in range(len(states)):
            for j in range(i + 1, len(states)):
                if np.array_equal(states[i], states[j]): failures.append(f"duplicate_initial_states:{i}:{j}")
        if int(data.attrs.get("num_demos", -1)) != len(names): failures.append("num_demos")
        if int(data.attrs.get("total", -1)) != frames: failures.append("total")
    return {"passed": not failures, "failures": failures, "episodes": len(states), "frames": frames,
            "bytes": path.stat().st_size, "sha256": sha256(path)}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-root", type=Path, default=Path("data/demos/libero36_task0_5_success_pilot_v1")); ap.add_argument("--overwrite", action="store_true"); args = ap.parse_args()
    out = args.output_root.resolve()
    if out.exists():
        if not args.overwrite: raise FileExistsError(out)
        shutil.rmtree(out)
    out.mkdir(parents=True)
    plan_path = Path("results/libero36_gate5_official_v6/task_000_put_on_top/replay.npz").resolve()
    original_path = Path("results/libero36_gate5_official_v6/task_000_put_on_top/original.npz").resolve()
    with np.load(plan_path, allow_pickle=False) as z:
        plan = {k: z[k] for k in z.files}
    row = row_for_task(); plan["path"] = str(plan_path); plan["bddl_path"] = row["bddl_path"]
    h5 = out / "task_000_layout_1_5_success.hdf5"; results = []
    with h5py.File(h5, "w") as handle:
        data = handle.create_group("data")
        data.attrs["bddl_file_name"] = str(row["bddl_path"]); data.attrs["env_name"] = "Libero_Tabletop_Manipulation"
        data.attrs["problem_info"] = json.dumps({"problem_name":"libero_tabletop_manipulation","domain_name":"robosuite","language_instruction":LANGUAGE,"task_id":TASK_ID,"layout_id":LAYOUT_ID}, sort_keys=True)
        data.attrs["env_args"] = json.dumps({"camera_names":["robot0_eye_in_hand","agentview"],"camera_heights":128,"camera_widths":128,"control_freq":20,"action_dimension":7}, sort_keys=True)
        data.attrs["formal_original_sha256"] = sha256(original_path); data.attrs["formal_replay_sha256"] = sha256(plan_path)
        for i, seed in enumerate(SEEDS):
            result = collect_episode(plan, seed, data.create_group(f"demo_{i}")); results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
        data.attrs["num_demos"] = len(results); data.attrs["total"] = sum(x["frames"] for x in results)
    h5_qc = validate(h5)
    qc_csv = out / "episode_qc.csv"
    with qc_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0])); w.writeheader(); w.writerows(results)
    manifest = {"schema_version":1,"artifact_kind":"libero36_task0_success_demonstration_pilot","task_id":TASK_ID,"layout_id":LAYOUT_ID,"language_instruction":LANGUAGE,"seeds":list(SEEDS),"formal_gate5_evidence_modified":False,"formal_replay_path":str(plan_path),"formal_original_path":str(original_path),"episodes_qc":results,"hdf5_qc":h5_qc,"passed":bool(h5_qc["passed"] and all(x["passed"] for x in results))}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    if not manifest["passed"]: raise SystemExit(2)


if __name__ == "__main__": main()
