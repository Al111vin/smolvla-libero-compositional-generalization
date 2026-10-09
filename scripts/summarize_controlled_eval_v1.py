"""Complete preregistered outcomes only; no partial gate claims."""
from math import comb
try:
    from controlled_eval_schedule_v1 import schedule
except ImportError:
    from scripts.controlled_eval_schedule_v1 import schedule


def summarize(records, exit_code):
    assert exit_code == 0 and len(records) == 48
    expected={r["key"]:r for r in schedule()}
    assert len({r["key"] for r in records}) == 48
    for r in records:
        assert r["key"] in expected
        assert all(r[k]==v for k,v in expected[r["key"]].items())
        assert type(r["success"]) is bool and 1 <= r["steps"] <= 300
    by_key={r["key"]:r for r in records}
    models={}
    for model in ("40k","80k"):
        paired=[by_key[f"paired_{i}_r0_{model}"]["success"] for i in range(20)]
        fixed=[paired[3]]+[by_key[f"extra_init3_3_r{r}_{model}"]["success"] for r in range(1,5)]
        models[model]={"paired_successes":sum(paired),"paired_total":20,
            "fixed_init3":fixed,"gate_passed":sum(paired)>=11 and all(fixed)}
    gains=[]; losses=[]
    for i in range(20):
        a=by_key[f"paired_{i}_r0_40k"]["success"]
        b=by_key[f"paired_{i}_r0_80k"]["success"]
        if b and not a: gains.append(i)
        if a and not b: losses.append(i)
    n=len(gains)+len(losses)
    p=min(1.,2*sum(comb(n,k) for k in range(min(len(gains),len(losses))+1))/2**n) if n else 1.
    return {"models":models,"gains":gains,"losses":losses,"mcnemar_exact_p":p,
        "scope":"controlled_environment_seed_v1, previously used init set; not historical V3 parity or unused-init generalization"}
