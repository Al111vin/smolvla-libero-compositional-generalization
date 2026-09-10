"""Find clean Task-25 contact-entry wrist orientations before any push."""

from __future__ import annotations

import csv
from pathlib import Path

from scripts import calibrate_libero_36_push as c


def main() -> None:
    row = next(
        item
        for item in c.reset_validator.read_layout_spec(
            Path("data/libero_36/layout_spec.csv")
        )
        if int(item["task_id"]) == 25 and int(item["layout_id"]) == 1
    )
    layout = Path("data/libero_36/layout_spec.csv")
    reset_manifest = Path("results/libero_36_reset_audit.manifest.json")
    reset = c.goal_validator.validate_reset_gate(reset_manifest, layout)
    c.validate_gate4(
        Path("results/libero_36_goal_audit.manifest.json"),
        layout,
        reset_manifest,
        reset,
    )
    c.bind_validated_gate3_seeds([row], reset)
    results = []
    for yaw in (-90.0, -45.0, 0.0, 45.0, 90.0):
        candidate = {
            "candidate_index": int(yaw + 120),
            "pusher_finger": "right",
            "pad_height_offset_m": 0.010,
            "desired_yaw_degrees": yaw,
            "candidate_order_key": f"entry_yaw_{yaw:+.0f}",
            # A single commanded step is sufficient to record the entry
            # contact and avoids classifying an unsafe motion as a push.
            "push_iterations": 1,
            "push_inner_steps": 1,
            "contact_follow": True,
            "contact_follow_penetration_m": 0.001,
        }
        try:
            original = c.run_original_attempt(row, candidate, int(row["calibration_seed"]))
            arrays = original["arrays"]
            results.append({
                "yaw": yaw,
                "contact": original["contact_step"] is not None,
                "selected_steps": original["selected_contact_steps"],
                "nonselected_steps": original["nonselected_robot_target_contact_steps"],
                "distractor_steps": original["robot_distractor_contact_steps"],
                "unexpected_steps": original["unexpected_object_contact_steps"],
                "support": original["support_all"],
                "steps": len(arrays["actions"]),
                "error": "",
            })
        except Exception as error:
            results.append({"yaw": yaw, "error": str(error)})
    path = Path("/tmp/task25_entry_yaw_scan.csv")
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=sorted({key for row in results for key in row}))
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
