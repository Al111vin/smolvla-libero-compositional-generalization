"""Strict lightweight summary of the registered six environment captures."""
def summarize(records, exit_code):
    assert exit_code == 0 and len(records) == 6
    summary = {}
    for mode in ("frozen_order", "preseed"):
        rows = [r for r in records if r["mode"] == mode]
        assert len(rows) == 3
        assert all(r["policy_calls"] == r["predicted_actions"] == 0 and
                   r["wait_steps"] == 10 and r["init"] == 3 for r in rows)
        assert all(r["seed"] == (12351 if mode == "preseed" else None) for r in rows)
        fields = set(rows[0]["model_arrays"])
        assert all(set(r["model_arrays"]) == fields for r in rows)
        cameras = set(rows[0]["cameras"])
        assert cameras == {"agentview_image", "robot0_eye_in_hand_image"}
        summary[mode] = {
            "unique_states": len({r["state"] for r in rows}),
            "model_unique_counts": {k: len({r["model_arrays"][k] for r in rows}) for k in sorted(fields)},
            "camera_unique_counts": {k: len({r["cameras"][k] for r in rows}) for k in sorted(cameras)}}
    return summary
