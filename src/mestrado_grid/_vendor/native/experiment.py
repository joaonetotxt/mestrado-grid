"""Bounded multi-seed isolated comparison; no production control imports."""
from bootstrap import *
import copy,time,math,traceback,gzip,random,shutil,datetime
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
from meta_planning.schema import Study
from meta_planning.evaluator import Evaluator,rank,_compare_reopened
from meta_planning.prepared_search import SearchState,compact,digest
from meta_planning.persistent_search import run_persistent
from meta_planning.engines import METHODS
from meta_planning.search_deadline import SearchDeadlineReached
from stage_cache import StageCachedEvaluator
from control import DiskContexts
from paired_policy import install
from accounting_checks import finance_checks

METHOD_LIST=list(METHODS)+['L-SHADE']
SEEDS=[96101,96109,96121,96137,96149]
SEARCH_CALLS=4096
TOTAL_CALLS=SEARCH_CALLS+8
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def brief(r):
 if r is None:return None
 m=r.get('metrics') or {}
 return {k:r.get(k) for k in ('status','candidate_id','plan_id','normalized_violation','normalized_violations','approved_cases','expected_cases','failed_cases')}|dict(FO=m.get('selection_objective'),VP30=m.get('objective_VP'),CVaR_VP=m.get('CVaR_total_VP'))
def label(job):
 regime,sim,beta,seed,method=job
 return f'{regime}_sim{sim}_b{beta:02d}_s{seed}/{method}'
def options(method):
 p=read(D/'campaign_protocol_reference.json');o=p['persistent_options']
 # The FULL scheduler is retained. This label explicitly declines to claim a
 # complete 96-case x every-mode/site sweep at a finite 4096-call search budget.
 o['coverage_contract']='partial_pilot';o['engine_parameters']=p['parameters_by_engine'].get(method,{})
 return o

class LabState(SearchState):
 def __init__(self,study,seed,dst,pop,total,method):
  self.total=total;self.method=method;self.cpu_start=time.process_time();self.stop_reason=None
  super().__init__(study,seed,8,dst,'inherited_explicit',pop)
  self.ev=StageCachedEvaluator(study,max_cases=768,max_bytes=134217728)
 def contexts_factory(self,scheduler):
  self.context_store=DiskContexts(self.dst/'contexts',dict(study=self.study,codec=self.codec,base=scheduler.base),capacity=1)
  return self.context_store
 def decorate_record(self,r,c):r['search_selection']='feasibility_violation_exact_cost_tie_v1';return r
 def history_record(self,r):return dict(**brief(r),search_elapsed_seconds=time.monotonic()-self.start,process_cpu_seconds=time.process_time()-self.cpu_start)
 def check_budget(self):
  if self.evaluations>=self.total:self.stop_reason='call_budget';raise SearchDeadlineReached()
  if (D/'STOP_REQUEST.json').exists():self.stop_reason='user_request';raise SearchDeadlineReached()
 def evaluate(self,x,event):
  self.check_budget();r=super().evaluate(x,event)
  while len(self.cache)>16:del self.cache[next(iter(self.cache))]
  return r
 def status(self,state='running',phase=None):
  save(self.dst/'status.json',dict(state=state,phase=phase,method=self.method,pid=os.getpid(),updated_utc=now(),
   evaluations=self.evaluations,budget=self.total,best=brief(self.best),initial=brief(self.initial_best),
   elapsed_s=time.monotonic()-self.start,cpu_s=time.process_time()-self.cpu_start,
   new_AC_cases=self.ev.case_misses,counters=dict(self.counters),visits=len(self.sessions)))
 def save_best(self):
  if self.best_x is None:return
  if getattr(self,'saved_id',None)!=self.best['candidate_id']:
   super().save_best();np.save(self.dst/'best_genome.npy',self.best_x,allow_pickle=False)
   save(self.dst/'best_record.json',brief(self.best));self.saved_id=self.best['candidate_id']
  self.status(phase='search')

def audit(state,job):
 search_wall=time.monotonic()-state.start;search_cpu=time.process_time()-state.cpu_start;at=time.monotonic()
 candidate=state.codec.decode(state.best_x)[0]
 cached=state.ev.evaluate(candidate);fresh=Evaluator(state.study).evaluate(candidate)
 _compare_reopened(plain({k:v for k,v in cached.items() if k!='wall_seconds'}),plain({k:v for k,v in fresh.items() if k!='wall_seconds'}))
 assert rank(state.decorate_record(compact(fresh),candidate))==rank(state.best)
 if fresh['status']=='feasible':
  assert fresh['approved_cases']==fresh['expected_cases']==96 and fresh['failed_cases']==0
  assert fresh['normalized_violation']==0 and all(v==0 for v in fresh['normalized_violations'].values())
  assert len(fresh['cases'])==96 and all(c['passed'] for c in fresh['cases'])
 financial=finance_checks(fresh)
 assert financial['beta']==job[2]/10 and financial['alpha']==.9
 with gzip.open(state.dst/'validation.json.gz','wt',encoding='utf-8') as f:json.dump(plain(fresh),f,ensure_ascii=False,allow_nan=False)
 guidance=state.persistent_diagnostics['guidance'];coverage=guidance['protected_resources']
 if state.total==TOTAL_CALLS:
  if not coverage['initial_opportunity_sweep_complete'] or len(coverage['families'])!=15 or not all(v.get('native_proposals',0)>0 for v in coverage['families'].values()):
   raise AssertionError('Fixed full-agenda trial failed to offer native proposals in every resource family; cannot certify comparison')
 summary=dict(job=list(job),context=label(job),seed=job[3],method=job[4],regime=job[0],simulation=job[1],beta=job[2]/10,
  start=brief(state.initial_best),final=brief(fresh),completed=True,calls=state.evaluations,budget=state.total,
  initial_calls=8,search_calls=state.evaluations-8,origin_counts=dict(state.counters),
  initial_population_sha256=digest(state.initial),source_population_sha256=sha(D/'inputs'/job[0]/f'sim{job[1]}.npy'),
  study_sha256=sha(D/'inputs'/job[0]/f'beta_{job[2]:02d}.json'),study_id=state.study.input_id,
  search_wall_s=search_wall,search_cpu_s=search_cpu,final_audit_s=time.monotonic()-at,final_audit_fresh_AC=96,
  cache=state.ev.cache_report(),fresh_audit_matches=True,finance=financial,
  resource_coverage=coverage,allocation_policy=guidance['allocation_policy'],
  effective_coverage=state.persistent_diagnostics['effective_coverage'],
  memories=state.persistent_diagnostics['memory'],constraint_tolerance_unchanged=True)
 assert summary['calls']==state.total
 save(state.dst/'summary.json',summary);state.status('complete','fresh_final_audit')
 # Disk-paged contexts are scratch, not inputs/evidence. Keep their final memory
 # summaries and hashes, all calls, incumbent genome and independently audited
 # candidate. No future run is allowed to silently resume or reuse this trial.
 if hasattr(state,'context_store'):
  save(state.dst/'scratch_context_hashes.json',dict(hashes=state.context_store.hashes,live=list(state.context_store.live),retention='Discarded scratch after final audit; no restart artifact advertised'))
  target=state.context_store.path.resolve();target.relative_to((D/'runs').resolve()) if 'runs' in target.parts else target.relative_to((D/'smoke').resolve())
  shutil.rmtree(target)
 return summary

def trial(job,total=TOTAL_CALLS,section='runs'):
 import gc
 gc.collect()  # Release prior trial's disk-proxy cycles in a reusable worker.
 out=D/section/label(job)
 if out.exists():raise FileExistsError('Never overwrite or silently repeat a trial: '+str(out))
 regime,sim,beta,seed,method=job
 try:
  st=Study(**read(D/'inputs'/regime/f'beta_{beta:02d}.json'))
  pop=np.load(D/'inputs'/regime/f'sim{sim}.npy',allow_pickle=False)
  install(method,beta/10,regime)
  state=LabState(st,seed,out,pop,total,method)
  save(out/'start_provenance.json',dict(source_population=str(D/'inputs'/regime/f'sim{sim}.npy'),
   source_sha256=sha(D/'inputs'/regime/f'sim{sim}.npy'),population_members=8,
   unique_genomes=len({x.tobytes() for x in pop}),copies_declared=len({x.tobytes() for x in pop})<8,
   kind='original MAT population' if sim==1 else 'original auxiliary heuristic population',
   optimized_pilot_or_production_finalist_used=False,same_fixed_population_across_seeds=True))
  save(out/'identity.json',dict(pid=os.getpid(),created_utc=now(),job=list(job),argv=sys.argv,release_sha256=sha(D/'execution_hashes.json')))
  state.initialize();run_persistent(state,method,total-8,.25,options(method))
  if state.evaluations!=total:raise RuntimeError(f'Trial ended at {state.evaluations}/{total}: {state.stop_reason}')
  result=audit(state,job)
  return dict(job=list(job),final=result['final'],calls=state.evaluations)
 except Exception:
  save(out/'error.json',dict(error=traceback.format_exc(),updated_utc=now(),job=list(job)))
  save(out/'status.json',dict(state='error',pid=os.getpid(),updated_utc=now()))
  raise

def verify():
 release=read(D/'execution_hashes.json')
 for rel,h in release.items():
  if sha(D/rel)!=h:raise RuntimeError('Isolated frozen source/input changed: '+rel)
 return len(release)

def jobs_for(mode):
 if mode=='smoke':return [(r,2,5,SEEDS[0],m) for r in ('normal','extremo') for m in METHOD_LIST]
 jobs=[];rng=random.Random(6172026)
 cells=[(r,s,b,seed) for seed in SEEDS for r in ('normal','extremo') for s in (1,2) for b in (0,5,10)]
 rng.shuffle(cells)
 for cell in cells:
  methods=METHOD_LIST.copy();rng.shuffle(methods);jobs.extend([(*cell,m) for m in methods])
 return jobs

def main(mode,workers):
 verify();jobs=jobs_for(mode);section='smoke' if mode=='smoke' else 'runs';budget=72 if mode=='smoke' else TOTAL_CALLS
 status_path=D/(mode+'_status.json');started=now();results=[];errors=[]
 if status_path.exists():raise FileExistsError('An experiment cannot be relaunched implicitly')
 save(D/(mode+'_order.json'),[list(j) for j in jobs])
 def status(state):save(status_path,dict(state=state,started_utc=started,updated_utc=now(),pid=os.getpid(),total=len(jobs),completed=len(results),failed=len(errors),errors=errors,workers=workers,budget_per_arm=budget))
 status('running')
 with ProcessPoolExecutor(max_workers=workers) as pool:
  futures={pool.submit(trial,j,budget,section):j for j in jobs}
  for f in as_completed(futures):
   j=futures[f]
   try:r=f.result();results.append(r);print(json.dumps(r),flush=True)
   except Exception:errors.append(dict(job=list(j),error=traceback.format_exc()));print(json.dumps(errors[-1]),flush=True)
   status('running')
   if mode=='benchmark':
    try:
     from report import make_report
     make_report()
    except Exception:save(D/'report_error.json',dict(error=traceback.format_exc(),updated_utc=now(),completed=len(results)))
 status('complete' if not errors else 'errors');verify()
 if mode=='benchmark':
  from report import make_report
  make_report()
 if errors:raise RuntimeError('Some trials failed; no automatic retries')

if __name__=='__main__':main(sys.argv[1],int(sys.argv[2]) if len(sys.argv)>2 else 8)
