"""Bounded40 midpoint rollout supervisor; no retries or success-conditioned budget."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from midpoint_eval_schedule_v1 import schedule, validate_protocol, environment_signature, validate_rollout
from diagnostic_process_guard_v1 import conflicting_processes
from capture_reset_render_provenance_v1 import EVALUATOR_SHA

REQUIRED_CODE = ('run_midpoint_eval_v1.py', 'midpoint_eval_schedule_v1.py',
    'controlled_eval_schedule_v1.py', 'eval_controlled_environment_v1.py',
    'controlled_eval_hooks_v1.py', 'capture_reset_render_provenance_v1.py',
    'first_decision_provenance_v1.py', 'diagnostic_process_guard_v1.py')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--manifest",required=True)
    args=parser.parse_args()
    manifest=json.loads(Path(args.manifest).read_text())
    root=Path(manifest["root"])
    assert not root.exists()
    base=Path(__file__).parent
    assert set(manifest['code_hashes']) == set(REQUIRED_CODE)
    assert hashlib.sha256(Path(manifest['evaluator']).read_bytes()).hexdigest() == EVALUATOR_SHA
    for name,sha in manifest["code_hashes"].items():
        assert hashlib.sha256((base/name).read_bytes()).hexdigest()==sha
    assert set(manifest["models"])=={"single20k","joint80k"}
    for model in manifest["models"].values():
        assert hashlib.sha256((Path(model["path"])/"model.safetensors").read_bytes()).hexdigest()==model["sha"]
    with open("/root/smolvla-training-prep/teacher_control_gpu.lock","r+") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not conflicting_processes()
        assert not subprocess.check_output(["nvidia-smi","--query-compute-apps=pid","--format=csv,noheader"],text=True).strip()
        assert os.statvfs(root.parent).f_bavail*os.statvfs(root.parent).f_frsize>2_000_000_000
        root.mkdir()
        code=1
        deadline=time.monotonic()+5400
        signatures={}
        model_signatures={}
        records=[]
        try:
            for row in schedule():
                remaining=deadline-time.monotonic()
                assert remaining>0, "Registered90-minute wall budget exhausted"
                target=root/row["key"]
                model=manifest["models"][row["model"]]
                command=[sys.executable,str(base/"eval_controlled_environment_v1.py"),
                    "--evaluator",manifest["evaluator"],"--checkpoint",model["path"],
                    "--model-sha",model["sha"],"--init-index",str(row["init"]),
                    "--cli-seed",str(row["cli_seed"]),"--root",str(target)]
                with (root/(row["key"]+".log")).open("x") as log:
                    result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,
                        timeout=min(600,remaining),cwd="/root/smolvla-eval-prep")
                assert result.returncode==0,(row,result.returncode)
                protocol=json.loads((target/"protocol.json").read_text())
                validate_protocol(row,protocol)
                assert protocol["checkpoint"]==model["path"] and protocol["model_sha"]==model["sha"]
                capture=json.loads((target/"first_input.json").read_text())
                signature=environment_signature(capture)
                if row["init"] in signatures:
                    assert signature==signatures[row["init"]],("ENVIRONMENT_MISMATCH",row)
                signatures[row["init"]]=signature
                key=(row["model"],row["init"])
                policy_signature={k:capture[k] for k in ("input","processor","loaded_model","rng")}
                if key in model_signatures:
                    assert policy_signature==model_signatures[key],("POLICY_INPUT_MISMATCH",row)
                model_signatures[key]=policy_signature
                summaries=list(target.glob("*_summary.csv"))
                assert len(summaries)==1
                with summaries[0].open() as f: summary=list(csv.DictReader(f))
                assert len(summary)==1 and summary[0]["success"] in ("True","False")
                action_files=list(target.glob("*_actions.csv"))
                assert len(action_files)==1
                with action_files[0].open() as f: actions=list(csv.DictReader(f))
                validate_rollout(row,summary[0],actions)
                records.append(dict(row,success=summary[0]["success"]=="True",steps=int(summary[0]["steps"])))
                with (root/(row["key"]+".verified.json")).open("x") as f: json.dump(records[-1],f)
                print("VERIFIED",row["key"],flush=True)
            with (root/"completion.json").open("x") as f: json.dump(records,f,indent=2)
            code=0
        finally:
            with (root/"exit_code").open("x") as f: f.write(str(code)+"\n")


if __name__=="__main__": main()
