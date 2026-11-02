"""One isolated process per arm. Same complete-evaluator accounting for all ten."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
import sys,json,time,gzip,hashlib,copy,traceback,datetime,collections
from pathlib import Path
VENDOR=Path(__file__).resolve().parent/"_vendor"
D=None
SETTINGS={}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.pending');temp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');temp.replace(p)
def append(p,x):
    with Path(p).open('a',encoding='utf-8') as f:f.write(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n')
def verify():
    release=read(VENDOR/'manifest.json')['files']
    for rel,h in release.items():
        if sha(VENDOR/rel)!=h:raise RuntimeError('Installed scientific source changed: '+rel)
    return len(release)
def brief(r):
    f=(r.get('metrics') or {}).get('finance') or {}
    return {k:r.get(k) for k in ('status','candidate_id','plan_id','approved_cases','expected_cases','failed_cases','normalized_violation','normalized_violations')}|dict(FO=(r.get('metrics') or {}).get('selection_objective'),PC1=f.get('PC1'),expected_operation=f.get('expected_operation'),CVaR_operation=f.get('CVaR_operation'))
class BudgetEnd(Exception):pass

class Meter:
    """Observe ALL evaluator instances, including hidden refiner/source checks."""
    def __init__(self,Evaluator,budget):
        self.budget=budget;self.phase='search';self.search_calls=0;self.audit_calls=0
        self.search_AC=0;self.audit_AC=0;self.in_evaluation=0;self.lp_calls=0;self.lp_wall=0.
        self.source_revalidations=0;self.Evaluator=Evaluator
        original_evaluate=Evaluator.evaluate;original_case=Evaluator._case
        meter=self
        def evaluate(ev,c):
            if meter.phase=='search':
                if meter.search_calls>=meter.budget:raise BudgetEnd()
                meter.search_calls+=1
            else:meter.audit_calls+=1
            meter.in_evaluation+=1
            try:return original_evaluate(ev,c)
            finally:meter.in_evaluation-=1
        def case(ev,*a,**kw):
            if not meter.in_evaluation:raise RuntimeError('Unbudgeted isolated AC call outside complete evaluation')
            if meter.phase=='search':meter.search_AC+=1
            else:meter.audit_AC+=1
            return original_case(ev,*a,**kw)
        Evaluator.evaluate=evaluate;Evaluator._case=case
    def instrument_lp(self,module):
        old=module.linprog;meter=self
        def run(*a,**kw):
            t=time.monotonic();meter.lp_calls+=1
            try:return old(*a,**kw)
            finally:meter.lp_wall+=time.monotonic()-t
        module.linprog=run
    def report(self):
        return dict(search_calls=self.search_calls,search_AC_cases=self.search_AC,
            independent_audit_calls=self.audit_calls,independent_audit_AC_cases=self.audit_AC,
            source_revalidations_charged=self.source_revalidations,LP_solves=self.lp_calls,LP_wall_s=self.lp_wall,
            candidate_cache=False,case_cache=False)

def run(job,budget,section):
    sim,beta,seed,method=job
    native=method not in ('HYBRID','HYBRID_AI')
    branch=VENDOR/('native' if native else 'hybrid');sys.path.insert(0,str(branch))
    if native:
        import experiment as ex
    else:
        import reviewed_search as rev
    from meta_planning.schema import Study,plain
    from meta_planning.search_arrays import ArrayCodec
    from meta_planning.evaluator import Evaluator,rank
    from accounting_checks import finance_checks
    import numpy as np
    st=Study(**read(D/'inputs'/f'beta_{beta:02d}.json'))
    assert st.c['alpha']==.9 and st.c['beta']==beta/10
    pop=np.load(D/'inputs'/f'sim{sim}.npy',allow_pickle=False)
    codec=ArrayCodec(st);initial_candidates=[codec.decode(x)[0] for x in pop]
    out=D/'result'
    if out.exists():raise FileExistsError('No trial overwrite or implicit retry: '+str(out))
    meter=Meter(Evaluator,budget)
    start=time.monotonic();cpu=time.process_time();initial=None;state=None;controller=None
    def trajectory(best,event):
        append(out/'trajectory.jsonl',dict(call=meter.search_calls,wall_s=time.monotonic()-start,
            cpu_s=time.process_time()-cpu,best=brief(best),event=event))
    def stopped():return (D/'STOP_TESTS.json').exists()
    class FullEvaluator(Evaluator):
        def __init__(self,study):super().__init__(study);self.calls=0;self.ac_cases=0;self.case_misses=0
        def _case(self,*a,**kw):
            self.ac_cases+=1;self.case_misses+=1;return super()._case(*a,**kw)
        def evaluate(self,c):
            r=super().evaluate(c);self.calls+=1
            if r.get('metrics') and r.get('failed_cases')==0:r['search_selection']='feasibility_violation_exact_cost_tie_v1'
            return r
    try:
        if native:
            from meta_planning.search_deadline import SearchDeadlineReached
            class NativeState(ex.LabState):
                def __init__(self):
                    super().__init__(st,seed,out,pop,budget,method);self.ev=FullEvaluator(st)
                def check_budget(self):
                    if stopped():raise RuntimeError('Isolated test stopped by user')
                    super().check_budget()
                def evaluate(self,x,event):
                    self.cache.clear()
                    r=super().evaluate(x,event)
                    assert self.evaluations==meter.search_calls
                    trajectory(self.best,event);return r
            ex.install(method,beta/10,SETTINGS['regime']);state=NativeState()
        else:
            from complete_modes import CompleteKernel
            from revision_policy import ReviewGate
            from coverage_fix import SharedDispatchMode
            from review_state import BranchVault
            import joint_dispatch_base
            # Daily and per-case refiners share this one LP entry point.
            assert joint_dispatch_base.ca is rev.ca
            meter.instrument_lp(rev.ca)
            # Only the experiment adapter changes lifecycle and accounting. The
            # deployed review, action selection and execution methods are reused.
            class IsolatedHybrid(rev.ReviewedSearch):
                def __init__(self):
                    self.st=self.strict=st;self.seed=seed;self.method=method;self.phase='strict'
                    self.out=out;self.clock=start;self.previous_wall=0.;self.smoke_calls=budget
                    self.review_gate=ReviewGate();self.shared_modes=SharedDispatchMode();self.review_sequence=0
                    self.queue=[];self.active_step=None;self.analytic_visit=0;self.visit=0;self.has_ai=False
                    self.calls=0;self.best=None;self.current=None;self.init_index=0
                    self.kernel=CompleteKernel(st,out,budget);self.ev=FullEvaluator(st)
                    self.kernel.ev=self.ev;self.kernel.evaluate=self.evaluate;self.kernel.progress=self.progress
                    self.sup=rev.Analyst(out/'ai_review') if method=='HYBRID_AI' else None
                    self.vault=BranchVault(out/'suspended',None)
                def check(self):
                    if stopped():raise rev.PairStopped()
                    if meter.search_calls>=budget:raise rev.SmokeComplete()
                    errors=out/'inference_errors.jsonl'
                    if errors.exists() and len(errors.read_text(encoding='utf-8').splitlines())>=3:
                        raise rev.PairStopped('Three model/plan failures; no synthetic fallback')
                def stopped(self):return stopped()
                def evaluate(self,c,event):
                    self.check();r=self.ev.evaluate(c);self.calls=meter.search_calls
                    k=self.kernel;accepted=k.best is None or rank(r)<rank(k.best)
                    if accepted:
                        k.best=r;k.current=copy.deepcopy(c)
                        save(out/'best_candidate.json',plain(c.to_dict()))
                    k.bank.offer(c,r,event|dict(call=self.calls));self.best=k.best;self.current=k.current
                    append(out/'history.jsonl',dict(call=self.calls,event=event,accepted=accepted,result=brief(r)))
                    trajectory(self.best,event);self.progress();return r
                def fresh(self,c,study,label):
                    # Unlike production monitoring, this optional decision-time
                    # audit consumes the common test budget and grants no free work.
                    meter.source_revalidations+=1
                    return self.evaluate(c,dict(role='charged_source_revalidation',label=label))
                def progress(self,state='running',**extra):
                    save(out/'status.json',dict(state=state,updated_utc=now(),pid=os.getpid(),job=job,
                        calls=meter.search_calls,budget=budget,best=brief(self.best) if self.best else None,
                        elapsed_s=time.monotonic()-start,CPU_s=time.process_time()-cpu,
                        reviews=self.review_sequence,shared_dispatch_visits=self.shared_modes.visits,
                        pending_experiments=len(self.queue),meter=meter.report(),**extra))
                def checkpoint(self):
                    # Compact evidence, not a production checkpoint or automatic resume.
                    save(out/'review_state.json',dict(call=self.calls,seed=self.seed,gate=vars(self.review_gate),
                        shared_dispatch_visits=self.shared_modes.visits,review_sequence=self.review_sequence,
                        pending=self.queue,vault=self.vault.export(),automatic_resume=False))
            controller=IsolatedHybrid()
            original_observation=rev.observation
            def observation(*a,**kw):
                obj=original_observation(*a,**kw);obj['paired_seed']=seed
                obj['budget_definition']='All evaluator calls, including source revalidation, charged; one independent final audit additional. No access to other trials.'
                return obj
            rev.observation=observation
            np.save(out/'initial_population.npy',pop,allow_pickle=False)
        save(out/'identity.json',dict(pid=os.getpid(),parent_pid=os.getppid(),created_utc=now(),argv=sys.argv,
            executable=sys.executable,job=job,budget=budget,release_sha256=sha(VENDOR/'manifest.json')))
        save(out/'start_provenance.json',dict(job=job,study_id=st.input_id,
            population_sha256=sha(D/'inputs'/f'sim{sim}.npy'),study_sha256=sha(D/'inputs'/f'beta_{beta:02d}.json'),
            candidate_ids=[c.candidate_id for c in initial_candidates],members=8,
            unique_members=len({c.candidate_id for c in initial_candidates}),starting_point=SETTINGS['starting_point'],
            strict_original_limits=True,input_population_copied=True))
        if native:
            state.initialize();initial=brief(state.best)
            ex.run_persistent(state,method,budget-8,.25,ex.options(method))
            best=state.best;c=state.codec.decode(state.best_x)[0]
            counters=dict(state.counters);details=state.persistent_diagnostics
            assert state.evaluations==budget
            coverage=details.get('guidance',{}).get('protected_resources',{})
            details['full_family_sweep_verified']=bool(coverage.get('initial_opportunity_sweep_complete') and
                len(coverage.get('families',{}))==15 and all(v.get('native_proposals',0)>0 for v in coverage.get('families',{}).values()))
        else:
            for i,c in enumerate(initial_candidates):controller.evaluate(c,dict(role='initialization',member=i))
            initial=brief(controller.best)
            try:
                while meter.search_calls<budget:
                    controller.step();controller.checkpoint()
            except (rev.SmokeComplete,BudgetEnd):pass
            best=controller.best;c=controller.current;counters=dict(controller.kernel.counts)
            details=dict(shared_dispatch_visits=controller.shared_modes.visits,review_count=controller.review_sequence,
                review_gate=vars(controller.review_gate),coverage=[dict(stage=k,states=list(cy),visits=n) for (k,cy),n in controller.kernel.coverage.visits.items()])
        assert meter.search_calls==budget,(meter.search_calls,budget)
        search_wall=time.monotonic()-start;search_cpu=time.process_time()-cpu
        meter.phase='audit';t=time.monotonic();fresh=FullEvaluator(st).evaluate(c)
        assert fresh['candidate_id']==best['candidate_id'] and fresh['status']==best['status']
        assert fresh['normalized_violations']==best['normalized_violations']
        assert abs(fresh['metrics']['selection_objective']-best['metrics']['selection_objective'])<=1e-5
        if fresh['status']=='feasible':
            assert fresh['approved_cases']==fresh['expected_cases']==96 and fresh['failed_cases']==0
            assert fresh['normalized_violation']==0 and all(v==0 for v in fresh['normalized_violations'].values())
        fin=finance_checks(fresh);assert fin['all_pass'] and fin['beta']==beta/10 and fin['alpha']==.9
        assert meter.audit_calls==1
        save(out/'best_candidate.json',plain(c.to_dict()))
        (out/'validation.json.gz').write_bytes(gzip.compress(json.dumps(plain(fresh),allow_nan=False).encode('utf-8'),compresslevel=2))
        ai=None
        if controller and controller.sup:
            accepted=list((out/'ai_review').glob('*.accepted.json'))
            assert accepted and controller.review_sequence==len(accepted)
            ai=dict(provider='codex_cli_authenticated_chatgpt',requests=controller.sup.calls,accepted=len(accepted),
                wait_s=controller.sup.wait_s,usage=controller.sup.usage,tool_events=sum(len(read(p)['tool_events']) for p in accepted),
                deterministic_given_seed=False)
            assert ai['tool_events']==0
        result=dict(job=job,method=method,simulation=sim,beta=beta/10,seed=seed,regime=SETTINGS['regime'],case_kind=SETTINGS['case_kind'],
            completed=True,initial=initial,final=brief(fresh),budget=budget,meter=meter.report(),
            search_wall_s=search_wall,search_cpu_s=search_cpu,final_audit_s=time.monotonic()-t,
            population_sha256=sha(D/'inputs'/f'sim{sim}.npy'),study_id=st.input_id,
            source_version='native_full_agenda' if native else 'rolling_review_v2',
            counters=counters,details=plain(details),finance=fin,AI=ai,fresh_audit_matches=True,
            strict_original_constraints=True,full_catalogue_eligible=True,full_catalogue_visited_not_guaranteed=True,
            accounting='all complete evaluations charged; one independent final audit additional; wall/CPU/LP/AI reported separately')
        save(out/'summary.json',result);save(out/'status.json',dict(state='complete',updated_utc=now(),pid=os.getpid(),job=job,
            calls=meter.search_calls,budget=budget,best=brief(fresh),meter=meter.report()))
        # Preserve native context evidence without retaining disposable large scratch.
        if native and hasattr(state,'context_store'):
            path=state.context_store.path.resolve();path.relative_to(out.resolve())
            save(out/'scratch_context_hashes.json',dict(hashes=state.context_store.hashes,live=list(state.context_store.live),no_resume=True))
            import shutil
            # Contexts retained in this local output for inspection.
        return result
    except BaseException:
        out.mkdir(parents=True,exist_ok=True)
        save(out/'error.json',dict(job=job,utc=now(),traceback=traceback.format_exc(),meter=meter.report()))
        save(out/'status.json',dict(state='failed',job=job,updated_utc=now(),pid=os.getpid(),calls=meter.search_calls))
        raise
def main():
    global D, SETTINGS
    D=Path(sys.argv[1]).resolve()
    SETTINGS=read(D/'job.json')
    verify()
    result=run(SETTINGS['job'],SETTINGS['budget'],'runs')
    verify()
    print(json.dumps(dict(completed=result['completed'],method=result['method'],final=result['final'],meter=result['meter'])),flush=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
