"""Fresh heuristic preparation followed by native cooperative engine search.

Prepared artifacts must come from this module, seed, study and fresh population.
Preparation is an explicit optimizer charged to each recipient's budget. The
auxiliary never runs in the subsequent native phase. No historical plan input.
"""
import argparse, collections, copy, hashlib, json, os, time
from pathlib import Path
import numpy as np
from .schema import Study, plain
from .search_arrays import ArrayCodec
from .engines import Engine, SearchConfig, METHODS
from .runner import common_population
from .evaluator import Evaluator, rank, _compare_reopened
from .case_cache import CaseCachedEvaluator
from .hierarchical_state_variation import HierarchicalStateVariation
from .native_blocks import Block, Field
from .balanced_blocks import BalancedScheduler, physical_key, diverse_projection
from .financial_policy import resolve

METRICS=('objective_VP','expected_VP','deterministic_VP','expected_operation_VP',
         'CVaR_operation_VP','CVaR_total_VP','cut_equivalent_MWh','selection_objective')

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)

def digest(x):return hashlib.sha256(np.asarray(x,dtype='<f8').tobytes()).hexdigest()

def source_hashes():
    return {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob('*.py')) if not p.name.startswith('test_')}

def compact(result):
    out={k:result.get(k) for k in ('status','candidate_id','plan_id','normalized_violation','approved_cases','expected_cases','failed_cases')}
    out['metrics']=None if result.get('metrics') is None else {k:result['metrics'][k] for k in METRICS}
    if out['metrics'] is not None:
        for key in ('financial_policy','finance','financial_views','EAC'):
            out['metrics'][key]=copy.deepcopy(result['metrics'][key])
    out['case_passed']=[bool(c['passed']) for c in result.get('cases',[])]
    out['normalized_violations']=copy.deepcopy(result.get('normalized_violations',{}))
    # Diagnostic data only. Never replace the official selection objective or
    # assign financial rank to an infeasible/inconclusive partial evaluation.
    out['case_feedback']=[dict(stage=c['stage'],state_index=c['state_id'],
        converged=c['converged'],passed=c['passed'],
        violations={k:v for k,v in c.get('normalized_violations',{}).items() if v>0},
        hourly_cut=float(c.get('hourly_costs',{}).get('conventional_cut',0.)),
        hourly_curtailment=float(sum(v for k,v in c.get('hourly_costs',{}).items() if k.startswith('curtailment_'))))
        for c in result.get('cases',[])]
    for source,target in zip(result.get('cases',[]),out['case_feedback']):
        if source.get('converged') and 'voltage_real' in source:
            target['voltage_real']=np.asarray(source['voltage_real'],dtype=float)
            target['voltage_imag']=np.asarray(source['voltage_imag'],dtype=float)
    return out

def fresh_population(codec,seed,n,initialization='mat_original'):
    return common_population(codec,SearchConfig('PSO',population=n,seed=seed,initialization=initialization))[0]


def initialization_audit(codec,population,initialization):
    """Persist decoded evidence; the initial tree is not a search constraint."""
    from .construction import original_mat_plan
    original=original_mat_plan(codec.study) if initialization=='mat_original' else None
    members=[]
    for x in population:
        c,_=codec.decode(x)
        exact=original is not None and c.plan.canonical()==original.canonical()
        no_additions=(not c.plan.line_buys and not c.plan.storage_buys and not np.any(c.new_capacity)
            and (c.physical is None or all(not np.any(v) for k,v in c.physical.items() if k.endswith('_choice'))))
        if original is not None:assert exact and no_additions,'Original MAT initializer changed assets or topology'
        members.append(dict(candidate_id=c.candidate_id,plan_id=c.plan_id,
            exact_original_plan=exact,no_added_assets=no_additions,
            closed_edges_by_stage=[sorted(t) for t in c.plan.trees]))
    return dict(profile=initialization,members=members,no_historical_candidate=initialization!='inherited_explicit',
        operational_start=('declared resource-aware P/Q and lossless network construction, not engine improvement'
            if initialization=='mat_original' else ('explicit inherited population; not engine improvement' if initialization=='inherited_explicit' else 'catalogue construction and seed; not engine improvement')),
        tree_fixed_during_search=False,all_catalogue_options_remain_eligible=True,
        feasibility_guaranteed=False)

class AllOptionsScheduler(BalancedScheduler):
    COMPLEMENTS=BalancedScheduler.COMPLEMENTS+('full_native',)
    def next(self,reference):
        full=self.count%4==3 and self.COMPLEMENTS[self.complement_visits%len(self.COMPLEMENTS)]=='full_native'
        block=super().next(reference)
        if full:
            return Block(reference,[Field('all_eligible_coordinates',np.arange(self.codec.dimension),'direct',False)],
                dict(number=self.count-1,scope='full_native',scheduler='seed-only complete-domain coverage'),self.mass)
        return block

def select_population(bank,n):
    """Half elites, half farthest within best remaining status tier; no new AC calls.

    Archive identity is physical controls, not latent keys. Diversity uses
    normalized canonical coordinates. Every selected member was evaluated.
    """
    pool=sorted(bank,key=lambda b:rank(b['record']))
    if not pool:raise ValueError('Empty evaluated archive')
    selected=list(pool[:min((n+1)//2,len(pool))]);rest=pool[len(selected):]
    while rest and len(selected)<n:
        tier=rank(rest[0]['record'])[0]
        eligible=[r for r in rest if rank(r['record'])[0]==tier]
        chosen=max(eligible,key=lambda r:min(float(np.mean((r['canonical']-s['canonical'])**2)) for s in selected))
        selected.append(chosen);rest=[r for r in rest if r['physical_key']!=chosen['physical_key']]
    unique=len(selected)
    while len(selected)<n:selected.append(selected[(len(selected)-unique)%unique])
    return np.asarray([s['x'] for s in selected]),[copy.deepcopy(s['record']) for s in selected],dict(
        unique_physical_members=unique,repeated_members=n-unique,
        distinct_plans=len({s['record']['plan_id'] for s in selected}),
        feasible_members=sum(s['record']['status']=='feasible' for s in selected),
        rule='half rank elites; half maximum minimum canonical distance within best remaining feasibility tier')

class SearchState:
    def __init__(self,study,seed,n,dst,initialization='mat_original',initial_population=None):
        self.study=study;self.codec=ArrayCodec(study);self.seed=seed;self.n=n;self.dst=Path(dst)
        self.dst.mkdir(parents=True,exist_ok=False)
        self.start=time.monotonic();self.charged_seconds=0.;self.evaluations=0
        self.ev=CaseCachedEvaluator(study);self.cache={};self.best=None;self.best_x=None
        self.bank=[];self.improvements=[];self.counters=collections.Counter();self.initial_best=None
        self.initialization=initialization
        self.initial=(fresh_population(self.codec,seed,n,initialization) if initial_population is None
                      else np.asarray(initial_population,dtype=float).copy())
        if self.initial.shape!=(n,self.codec.dimension) or not np.isfinite(self.initial).all() or np.any((self.initial<0)|(self.initial>1)):
            raise ValueError('Invalid explicit initial population')
        self.population=self.initial.copy()
        self.prefix=None;self.coverage=set();self.sessions=[];self.charged_AC_cases=0
        self.variation=HierarchicalStateVariation(self.codec,seed,common_only=True)
        np.save(self.dst/('initial_population.npy' if initialization=='inherited_explicit' else 'fresh_population.npy'),self.initial,allow_pickle=False)
        write(self.dst/'initialization.json',initialization_audit(self.codec,self.initial,initialization))

    def status(self,state='running',phase=None):
        write(self.dst/'status.json',dict(state=state,phase=phase,pid=os.getpid(),evaluations=self.evaluations,
            financial_policy=resolve(self.study.c),
            best=self.best,counters=dict(self.counters),actual_seconds=time.monotonic()-self.start,
            charged_seconds=self.charged_seconds+time.monotonic()-self.start,new_AC_cases=self.ev.case_misses,
            charged_AC_cases=self.charged_AC_cases+self.ev.case_misses,session=len(self.sessions)))

    def decorate_record(self,record,candidate):
        return record

    def history_record(self,record):
        return record

    def evaluate(self,x,event):
        candidate,repairs=self.codec.decode(x);cid=candidate.candidate_id
        before=self.ev.case_misses;hit=cid in self.cache
        r=copy.deepcopy(self.cache[cid]) if hit else compact(self.ev.evaluate(candidate))
        r=self.decorate_record(r,candidate)
        assert r['candidate_id']==cid
        self.cache[cid]=copy.deepcopy(r);previous=copy.deepcopy(self.best)
        accepted=previous is None or rank(r)<rank(previous)
        self.evaluations+=1;origin=event['origin'];self.counters[origin]+=1
        self.counters[origin+'_'+r['status']]+=1;self.counters[origin+'_cache_hits']+=int(hit)
        if accepted:
            self.best=copy.deepcopy(r);self.best_x=np.asarray(x).copy()
            if previous is not None:
                rec=dict(evaluation=self.evaluations,origin=origin,from_status=previous['status'],to_status=r['status'],candidate_id=cid)
                if previous['status']=='feasible' and r['status']=='feasible':
                    rec['metric_deltas']={k:previous['metrics'][k]-r['metrics'][k] for k in METRICS}
                self.improvements.append(rec)
        if previous is not None:assert rank(self.best)<=rank(previous)
        key=physical_key(candidate);same=next((i for i,b in enumerate(self.bank) if b['physical_key']==key),None)
        if same is None or rank(r)<rank(self.bank[same]['record']):
            # Canonical values are diversity metadata, never untested replacement keys.
            item=dict(x=np.asarray(x).copy(),canonical=np.asarray(self.codec.encode(candidate),float),
                      record=copy.deepcopy(r),physical_key=key)
            if same is None:self.bank.append(item)
            else:self.bank[same]=item
            self.bank.sort(key=lambda b:rank(b['record']));self.bank=self.bank[:8*self.n]
        with (self.dst/'history.jsonl').open('a',encoding='utf-8') as f:
            f.write(json.dumps(plain(dict(evaluation=self.evaluations,event=event,genome_sha256=digest(x),
                candidate_cache_hit=hit,new_AC_cases=self.ev.case_misses-before,accepted_global=accepted,
                decoder_repairs=repairs,**self.history_record(r))),ensure_ascii=False,allow_nan=False)+'\n')
        if self.evaluations%self.n==0:self.status(phase=origin)
        return r

    def initialize(self):
        self.status(phase='initialization')
        for i,x in enumerate(self.initial):
            self.evaluate(x,dict(origin='initialization',member=i,
                component=('original_MAT_tree_resource_aware_construction' if self.initialization=='mat_original'
                    else ('explicit_inherited_member' if self.initialization=='inherited_explicit' else ('catalogue_physical_construction' if i==0 else 'random_member')))))
        self.initial_best=copy.deepcopy(self.best)

    def heuristic(self,calls,origin):
        if calls<0 or calls%self.n:raise ValueError('Heuristic calls must be a nonnegative population multiple')
        for _ in range(calls//self.n):
            anchor=self.best_x.copy()
            proposals,events=self.variation.propose(np.tile(anchor,(self.n,1)),anchor)
            for x,detail in zip(proposals,events):
                assert detail['native_credit'] is False
                r=self.evaluate(x,dict(origin=origin,auxiliary_event=detail));self.variation.observe(detail,r)
            self.save_best()

    def save_best(self):
        candidate,_=self.codec.decode(self.best_x)
        write(self.dst/'best_candidate.json',candidate.to_dict());write(self.dst/'improvements.json',self.improvements)

    def export_preparation(self,calls):
        self.population,records,diversity=select_population(self.bank,self.n)
        np.save(self.dst/'prepared_population.npy',self.population,allow_pickle=False)
        np.savez_compressed(self.dst/'archive.npz',genomes=np.asarray([b['x'] for b in self.bank]),
            canonical=np.asarray([b['canonical'] for b in self.bank]),best=self.best_x)
        write(self.dst/'archive.json',[{k:v for k,v in b.items() if k not in ('x','canonical')} for b in self.bank])
        write(self.dst/'variation.json',self.variation.state());write(self.dst/'population_records.json',records);self.save_best()
        result=dict(kind='fresh_heuristic_preparation_v1',seed=self.seed,population=self.n,
            initialization=self.initialization,
            preparation_calls=calls,evaluations=self.evaluations,study_id=self.study.input_id,source_hashes=source_hashes(),
            initial_population_sha256=digest(self.initial),prepared_population_sha256=digest(self.population),
            best=self.best,initial_best=self.initial_best,diversity=diversity,counters=dict(self.counters),
            AC_cases=self.ev.case_misses,elapsed_seconds=time.monotonic()-self.start)
        filenames=['fresh_population.npy','prepared_population.npy','archive.npz','archive.json','variation.json',
                   'population_records.json','best_candidate.json','history.jsonl','improvements.json','initialization.json']
        result['files']={f:hashlib.sha256((self.dst/f).read_bytes()).hexdigest() for f in filenames}
        write(self.dst/'preparation.json',result);self.status('completed','preparation');return result

    def import_preparation(self,path,calls):
        path=Path(path);p=read(path/'preparation.json')
        assert p['kind']=='fresh_heuristic_preparation_v1'
        assert p['seed']==self.seed and p['population']==self.n and p['preparation_calls']==calls
        assert p['initialization']==self.initialization,'Preparation has a different initialization profile'
        assert p['study_id']==self.study.input_id and p['source_hashes']==source_hashes()
        assert p['initial_population_sha256']==digest(self.initial)
        for f,h in p['files'].items():assert hashlib.sha256((path/f).read_bytes()).hexdigest()==h,f
        self.population=np.load(path/'prepared_population.npy',allow_pickle=False)
        assert digest(self.population)==p['prepared_population_sha256']
        records=read(path/'population_records.json')
        for x,r in zip(self.population,records):assert self.codec.decode(x)[0].candidate_id==r['candidate_id']
        with np.load(path/'archive.npz',allow_pickle=False) as arrays:
            self.bank=[dict(**b,x=x.copy(),canonical=c.copy()) for b,x,c in zip(read(path/'archive.json'),arrays['genomes'],arrays['canonical'])]
            self.best_x=arrays['best'].copy()
        self.best=copy.deepcopy(p['best']);self.initial_best=copy.deepcopy(p['initial_best'])
        assert self.codec.decode(self.best_x)[0].candidate_id==self.best['candidate_id']
        self.evaluations=p['evaluations'];assert self.evaluations==self.n+calls
        self.counters=collections.Counter(p['counters']);self.improvements=read(path/'improvements.json')
        self.variation=HierarchicalStateVariation.restore(self.codec,read(path/'variation.json'))
        self.charged_seconds=p['elapsed_seconds'];self.charged_AC_cases=p['AC_cases']
        self.prefix=dict(path=str(path.resolve()),preparation_sha256=hashlib.sha256((path/'preparation.json').read_bytes()).hexdigest())
        write(self.dst/'preparation_reference.json',self.prefix)

    def native(self,method,calls,mass,search_mode='persistent',options=None):
        if search_mode=='persistent':
            from .persistent_search import run_persistent
            return run_persistent(self,method,calls,mass,options)
        if search_mode!='legacy_blocks':raise ValueError('Unknown native search mode')
        if calls<0 or calls%(4*self.n):raise ValueError('Native calls must be a 4*population multiple')
        scheduler=AllOptionsScheduler(self.codec,self.seed,mass)
        for session in range(calls//(4*self.n)):
            block=scheduler.next(self.best_x);description=block.describe()
            self.coverage.update(int(i) for field in block.fields for i in field.indices)
            sources=[self.best_x]+[b['x'] for b in sorted(self.bank,key=lambda b:rank(b['record']))]+list(self.population)
            local_initial,diversity=diverse_projection(block,self.codec,sources,self.n)
            seed=int(np.random.SeedSequence([self.seed,session,3917]).generate_state(1)[0])
            params={'mutation_rate':min(1.,1./block.dimension)} if method=='GA' else {}
            engine=Engine(SearchConfig(method,population=self.n,generations=3,seed=seed,parameters=params,
                progress_schedule='budget',comparison_profile='native_operators'),local_initial)
            local_best=None
            while not engine.done:
                x=engine.ask();origin='context_retest' if engine.generation==0 else 'native_offspring'
                r=self.evaluate(block.decode(x),dict(origin=origin,method=method,session=session,generation=engine.generation,block=description))
                if local_best is None or rank(r)<rank(local_best):local_best=r
                engine.set_budget_progress(float(np.clip((engine.evaluations+1-self.n)/(2*self.n),0,1)));engine.tell(x,r)
                assert rank(engine.best['record'])==rank(local_best)
            self.sessions.append(dict(session=session,block=description,diversity=diversity,best=copy.deepcopy(self.best)))
            self.save_best();write(self.dst/'sessions.json',self.sessions)

    def finish(self,cfg,phase_start):
        search_seconds=time.monotonic()-self.start;search_AC=self.ev.case_misses
        candidate,_=self.codec.decode(self.best_x)
        cached=self.ev.evaluate(candidate);fresh=Evaluator(self.study).evaluate(candidate)
        _compare_reopened(plain({k:v for k,v in cached.items() if k!='wall_seconds'}),plain({k:v for k,v in fresh.items() if k!='wall_seconds'}))
        assert rank(self.decorate_record(compact(fresh),candidate))==rank(self.best) and fresh['candidate_id']==candidate.candidate_id
        if fresh['status']=='feasible':assert fresh['approved_cases']==self.study.K*self.study.S
        if fresh['metrics']:
            m=fresh['metrics'];beta=self.study.c['beta']
            assert np.isclose(m['objective_VP'],m['deterministic_VP']+(1-beta)*m['expected_operation_VP']+beta*m['CVaR_operation_VP'],rtol=1e-13,atol=1e-5)
        assert self.evaluations==cfg['budget']
        summary=dict(**cfg,completed=True,evaluations=self.evaluations,best=compact(fresh),initial_best=self.initial_best,
            financial_policy=resolve(self.study.c),
            phase_start=phase_start,counters=dict(self.counters),initial_population_sha256=digest(self.initial),
            engine_population_sha256=digest(self.population),preparation_reference=self.prefix,
            eligible_dimensions=self.codec.dimension,eligible_coordinates_visited=len(self.coverage),sessions=len(self.sessions),
            charged_search_seconds=self.charged_seconds+search_seconds,actual_search_seconds=search_seconds,
            validation_seconds=time.monotonic()-self.start-search_seconds,charged_AC_cases=self.charged_AC_cases+search_AC,
            validation_AC_cases=self.study.K*self.study.S+self.ev.case_misses-search_AC,
            all_cuts_exactly_zero=bool(np.all(candidate.operation.cut==0)),
            attribution='Preparation and context projections separated from native offspring. Native descendants inherit prepared decisions.')
        if hasattr(self,'persistent_diagnostics'):
            summary['persistent_search']=self.persistent_diagnostics
        self.save_best();write(self.dst/'validation.json',fresh);write(self.dst/'summary.json',summary)
        self.status('completed','final_validation');return summary

def prepare(study,seed,population,calls,output,initialization='mat_original'):
    if population<4 or population%2:raise ValueError('Even population >=4 required')
    if calls<0 or calls%population:raise ValueError('Preparation calls must be a population multiple')
    s=SearchState(study,seed,population,output,initialization);s.initialize();s.heuristic(calls,'heuristic_preparation')
    return s.export_preparation(calls)

def run(study,cfg,output,preparation=None):
    cfg=copy.deepcopy(cfg)
    cfg.setdefault('initialization','mat_original')
    SearchConfig('PSO',initialization=cfg['initialization'])
    n=cfg['population'];mode=cfg['mode'];calls=cfg['preparation_calls']
    if n<4 or n%2:raise ValueError('Even population >=4 required')
    if mode not in ('baseline_native','prepared_native','heuristic_only'):raise ValueError('Unknown mode')
    if cfg['method'] not in METHODS and mode!='heuristic_only':raise ValueError('Unknown native engine')
    search_mode=cfg.get('search_mode','persistent')
    if search_mode not in ('persistent','legacy_blocks'):raise ValueError('Unknown native search mode')
    if type(cfg['budget'])!=int or cfg['budget']<n+calls:raise ValueError('Incompatible total budget')
    if (mode=='heuristic_only' or search_mode=='legacy_blocks') and (cfg['budget']-n-calls)%(4*n):raise ValueError('Incompatible total budget')
    if mode!='heuristic_only' and cfg['budget']-(n if mode=='baseline_native' else n+calls)<n:raise ValueError('Native population budget missing')
    if calls<0 or calls%(4*n):raise ValueError('Preparation budget must be a 4*population multiple')
    if mode=='baseline_native' and preparation is not None:raise ValueError('Baseline cannot receive preparation')
    # Reject an underfunded full-coverage request before preparation or output
    # construction performs any expensive fitness evaluations.
    if mode!='heuristic_only' and search_mode=='persistent':
        from .persistent_search import PersistentOptions, exploration_preflight
        from .execution import resolve_options
        cfg['persistent_options']=resolve_options(cfg.get('persistent_options'))
        options=PersistentOptions(**cfg['persistent_options'])
        options.parameters_for(cfg['method'],n)
        planned=exploration_preflight(ArrayCodec(study),n,options,cfg['seed'],cfg.get('endpoint_mass',.25))
        native_budget=cfg['budget']-(n if mode=='baseline_native' else n+calls)
        if options.coverage_contract=='all_cases_and_families' and native_budget<planned['conservative_calls']:
            raise ValueError(f"Insufficient exploration budget: {native_budget} < {planned['conservative_calls']}; use an explicitly declared partial_pilot")
    if Path(output).exists():raise FileExistsError('Choose an unused output directory')
    if mode!='baseline_native' and preparation is None:
        preparation=Path(str(output)+'_preparation');prepare(study,cfg['seed'],n,calls,preparation,cfg['initialization'])
    s=SearchState(study,cfg['seed'],n,output,cfg['initialization'])
    write(s.dst/'protocol.json',dict(config=cfg,study_id=study.input_id,source_hashes=source_hashes(),
        objective='strict feasibility then configured financial risk objective; no cut priority',
        financial_policy=resolve(study.c),
        native_search_mode=search_mode,
        initialization=cfg['initialization'],no_historical_solution=True,
        budget='initialization, preparation, context and offspring calls including duplicates; final validation extra',
        auxiliary_during_native=False))
    if mode=='baseline_native':s.initialize()
    else:
        s.import_preparation(preparation,calls)
        if mode=='prepared_native':
            # Hand off exactly the selected population, not the entire heuristic archive.
            selected_keys={physical_key(s.codec.decode(x)[0]) for x in s.population}
            s.bank=[b for b in s.bank if b['physical_key'] in selected_keys]
            assert len(s.bank)==len(selected_keys)
            assert any(b['record']['candidate_id']==s.best['candidate_id'] for b in s.bank)
            write(s.dst/'handoff.json',dict(archive_members_received=len(s.bank),population_members=n,
                candidate_ids=[s.codec.decode(x)[0].candidate_id for x in s.population],
                no_unselected_archive_members=True,auxiliary_during_native=False))
    phase_start=copy.deepcopy(s.best);np.save(s.dst/'engine_population.npy',s.population,allow_pickle=False)
    if mode=='heuristic_only':s.heuristic(cfg['budget']-s.evaluations,'heuristic_continuation')
    else:s.native(cfg['method'],cfg['budget']-s.evaluations,cfg.get('endpoint_mass',.25),search_mode,cfg.get('persistent_options'))
    return s.finish(cfg,phase_start)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',required=True,help='Serialized Study JSON with physical and risk options')
    p.add_argument('--config',required=True,help='seed, population, mode, method, budget, preparation_calls')
    p.add_argument('--output',required=True,help='New directory; existing results are not overwritten')
    p.add_argument('--preparation',help='Verified fresh preparation from the same study, source and seed')
    args=p.parse_args();result=run(Study(**read(args.study)),read(args.config),args.output,args.preparation)
    print(json.dumps(plain(result),ensure_ascii=False))

if __name__=='__main__':main()
