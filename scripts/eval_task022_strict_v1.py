import argparse,csv,hashlib,json,os,random,subprocess
from pathlib import Path
os.environ.setdefault('MUJOCO_GL','egl'); os.environ.setdefault('PYOPENGL_PLATFORM','egl'); os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np, torch
torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False; torch.use_deterministic_algorithms(True,warn_only=False)
from libero.libero.envs import OffScreenRenderEnv
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from eval_v3_task0 import action_to_numpy, observation_to_frame
from libero_36_camera import apply_frozen_agentview_camera

def sha(p):
 h=hashlib.sha256(); h.update(Path(p).read_bytes()); return h.hexdigest()
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--checkpoint-dir',required=True); ap.add_argument('--states-dir',required=True); ap.add_argument('--bddl',required=True); ap.add_argument('--out',required=True); ap.add_argument('--instruction',required=True); ap.add_argument('--steps',type=int,default=280); ap.add_argument('--wait',type=int,default=10); ap.add_argument('--n-action-steps',type=int,default=25); ap.add_argument('--max-states',type=int,default=50); args=ap.parse_args()
 out=Path(args.out); out.mkdir(parents=True,exist_ok=True); sd=Path(args.states_dir); manifest=json.loads((sd/'manifest.json').read_text()); assert manifest['count']==50
 ckpts=sorted(Path(args.checkpoint_dir).glob('*/pretrained_model'))
 all_summary=[]
 for ck in ckpts:
  label=ck.parent.name; d=out/f'checkpoint_{label}'; d.mkdir(exist_ok=True)
  policy=SmolVLAPolicy.from_pretrained(str(ck)); policy.to('cuda'); policy.eval(); policy.config.n_action_steps=args.n_action_steps
  pre,post=make_pre_post_processors(policy.config,pretrained_path=str(ck),preprocessor_overrides={'device_processor':{'device':'cuda'}})
  env=OffScreenRenderEnv(bddl_file_name=args.bddl,camera_heights=128,camera_widths=128,horizon=1000)
  for rec in manifest['states'][:args.max_states]:
   idx=rec['index']; state=np.load(sd/rec['path'])['state']; seed=2000+idx
   np.random.seed(1000+idx); random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
   env.reset(); env.set_init_state(state); obs,_=apply_frozen_agentview_camera(env); policy.reset()
   for _ in range(args.wait): obs,_,done,_=env.step(np.array([0,0,0,0,0,0,-1],dtype=np.float32));
   success=False; total=0.; steps=0
   for t in range(args.steps):
    with torch.inference_mode(): a=action_to_numpy(post(policy.select_action(pre(observation_to_frame(obs,args.instruction)))))
    a=np.asarray(a,dtype=np.float32).squeeze(); assert a.shape==(7,) and np.all(np.isfinite(a)); obs,r,done,info=env.step(np.clip(a,-1,1)); total+=float(r); steps=t+1
    success=bool(getattr(env,'check_success',lambda:False)()) or total>0 or bool(info.get('success',False));
    if success or done: break
   row={'checkpoint':label,'checkpoint_sha256':sha(ck/'model.safetensors'),'task_id':22,'layout_id':1,'state_index':idx,'state_seed':rec['seed'],'policy_seed':seed,'success':success,'total_reward':total,'steps':steps,'max_steps':args.steps,'wait_steps':args.wait,'n_action_steps':args.n_action_steps,'bddl_sha256':manifest['bddl_sha256'],'pure_vision':True}
   (d/f'state_{idx:03d}.json').write_text(json.dumps(row,indent=2)+'\n'); all_summary.append(row)
  env.close()
 summary={'schema_version':'libero36_task022_strict_eval_v1','task_id':22,'layout_id':1,'instruction':args.instruction,'states_manifest_sha256':sha(sd/'manifest.json'),'bddl_sha256':manifest['bddl_sha256'],'records':len(all_summary),'checkpoints':{}}
 for r in all_summary:
  s=summary['checkpoints'].setdefault(r['checkpoint'],{'num_rollouts':0,'num_success':0,'success_indices':[],'mean_reward':0.0}); s['num_rollouts']+=1; s['num_success']+=int(r['success']); s['mean_reward']+=r['total_reward']; s['success_indices'] += [r['state_index']] if r['success'] else []
 for s in summary['checkpoints'].values(): s['mean_reward']/=s['num_rollouts']; s['success_rate']=s['num_success']/s['num_rollouts']
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
