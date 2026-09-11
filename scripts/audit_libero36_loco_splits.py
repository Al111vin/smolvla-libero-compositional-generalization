from __future__ import annotations
import csv, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DATASET=ROOT/"datasets/lerobot/libero36_feasible_32_frozen_v1"
OUT=ROOT/"data/training/libero36_feasible_v1/splits"
INCLUDED=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,16,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,34]
def factors(s):
 s=s.lower()
 obj=("akita_black_bowl" if "akita black bowl" in s else "yellow_white_mug" if ("yellow and white mug" in s or "white and yellow mug" in s) else "alphabet_soup" if "alphabet soup" in s else "cream_cheese_box" if "cream cheese box" in s else None)
 skill="push" if s.startswith("push") else "pick_place_inside" if "inside the basket" in s else "pick_place_on_top" if "on the plate" in s else None
 region="left" if "left region" in s or "left target region" in s else "middle" if ("middle region" in s or "middle target region" in s or "middle of the table" in s) else "right" if "right region" in s or "right target region" in s else None
 if None in (obj,skill,region): raise ValueError(s)
 return obj,skill,region
def main():
 import pyarrow.parquet as pq
 vals=pq.read_table(DATASET/"meta/tasks.parquet").to_pydict()
 rows=[]
 for idx,instr in zip(vals["task_index"],vals["task"]):
  tid=INCLUDED[int(idx)]; obj,skill,region=factors(instr)
  rows.append({"task_id":tid,"task_index":int(idx),"instruction":instr,"object":obj,"skill":skill,"region":region})
 assert {r["task_id"] for r in rows}==set(INCLUDED)
 candidates=[]
 for held in rows:
  train=[r for r in rows if r["task_id"]!=held["task_id"]]
  cov={k:held[k] in {r[k] for r in train} for k in ("object","skill","region")}
  candidates.append({"task_id":held["task_id"],"task_index":held["task_index"],"held_out_task_id":held["task_id"],"instruction":held["instruction"],"object":held["object"],"skill":held["skill"],"region":held["region"],"train_task_ids":[r["task_id"] for r in train],"factor_coverage":cov,"eligible":all(cov.values())})
 assert all(x["eligible"] for x in candidates)
 primary=next(x for x in candidates if x["held_out_task_id"]==0)
 OUT.mkdir(parents=True,exist_ok=True)
 fields=["task_id","task_index","instruction","object","skill","region"]
 with (OUT/"loco_train_v1.csv").open("w",newline="") as f:
  w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(r for r in rows if r["task_id"] in primary["train_task_ids"])
 with (OUT/"loco_heldout_candidates_v1.csv").open("w",newline="") as f:
  fs=fields+["factor_coverage","eligible"]; w=csv.DictWriter(f,fieldnames=fs); w.writeheader()
  for r in candidates: w.writerow({**{k:r[k] for k in fields},"factor_coverage":json.dumps(r["factor_coverage"],sort_keys=True,separators=(",",":")),"eligible":str(r["eligible"]).lower()})
 audit={"schema_version":"libero36-loco-factor-audit-v1","dataset_root":str(DATASET.relative_to(ROOT)),"feasible_task_ids":INCLUDED,"factor_policy":"Each one-task held-out candidate must retain its object, skill, and region values in the 31-task training set.","candidate_count":len(candidates),"all_candidates_eligible":all(x["eligible"] for x in candidates),"primary_split":{"name":"holdout_task_000","held_out_task_ids":[0],"train_task_ids":primary["train_task_ids"],"train_csv":str((OUT/"loco_train_v1.csv").relative_to(ROOT)),"heldout_csv":str((OUT/"loco_heldout_candidates_v1.csv").relative_to(ROOT))},"candidates":candidates}
 (OUT/"loco_audit_v1.json").write_text(json.dumps(audit,indent=2)+"\n")
 print("wrote",OUT/"loco_audit_v1.json","candidates",len(candidates))
if __name__=="__main__": main()
