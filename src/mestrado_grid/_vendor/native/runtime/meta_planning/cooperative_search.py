"""Budgeted semantic cooperation with deterministic journal replay.

Each episode has a frozen complete context and its own native engine. Scores
are never reused across contexts. Initial neighbourhood sampling is explicitly
credited as such. This is an optional experimental solver, not a new default.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import copy,time
import numpy as np
from .schema import Candidate,Study,plain,fingerprint
from .codec import Codec
from .initialization import initialize
from .engines import Engine,SearchConfig,METHODS
from .evaluator import Evaluator,save_evaluation,audit_saved,rank,source_hash
from .runner import seal,unseal,locked,sha,discrete_id
from .persistence import load_evaluation
from .financial_policy import compact_metrics, resolve
from .semantic_neighborhoods import FAMILIES,Space,make_space,physical_changes


@dataclass(frozen=True)
class CooperativeConfig:
    method:str='PSO'
    seed:int=701
    population:int=4
    proposals:int=100
    episode_generations:int=2
    radius:float=.1
    global_probability:float=.125
    coupled:bool=True
    exploratory:bool=False
    exploration_limit:float=.1
    restart_every:int=0
    spatial_hops:int=1
    all_cycles:bool=False

    def __post_init__(self):
        if self.method not in (*METHODS,'COMMON','CONSTRUCT'):raise ValueError('Unknown cooperative method')
        for n,lo in [('seed',0),('population',4),('proposals',4),('episode_generations',1),('restart_every',0)]:
            if type(getattr(self,n))!=int or getattr(self,n)<lo:raise ValueError('Invalid '+n)
        if self.proposals<self.population:raise ValueError('Incomplete initial budget')
        for n in ('radius','global_probability','exploration_limit'):
            v=getattr(self,n)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 or (n!='global_probability' and v==0):raise ValueError('Invalid '+n)
        if type(self.coupled)!=bool or type(self.exploratory)!=bool:raise ValueError('Boolean policy required')
        if type(self.spatial_hops)!=int or not 1<=self.spatial_hops<=4 or type(self.all_cycles)!=bool:raise ValueError('Invalid neighbourhood breadth')


class Pause(Exception):pass
class Budget(Exception):pass


def initial_candidate(codec,seed):
    _,x,_=initialize(codec,seed,'resource')
    c,_=codec.decode(x);candidate,_=codec.decode(codec.encode(c))
    return candidate,x


def compact(result,directory,manifest,candidate):
    return dict(directory=directory,manifest=manifest,candidate_id=candidate.candidate_id,plan_id=candidate.plan_id,
                discrete_plan_id=discrete_id(candidate),status=result['status'],normalized_violation=result['normalized_violation'],
                failed_cases=result['failed_cases'],expected_cases=result['expected_cases'],approved_cases=result['approved_cases'],
                normalized_violations=result['normalized_violations'],metrics=None if result['metrics'] is None else
                compact_metrics(result['metrics'], ('objective_VP','expected_VP','CVaR_operation_VP','cut_equivalent_MWh')))


def verify(root,record,signature):
    path=(root/record['directory']).resolve()
    if not path.is_relative_to(root.resolve()) or sha(path/'manifest.json')!=record['manifest']:raise ValueError('Changed evaluation manifest')
    snapshot,value,result=load_evaluation(path);candidate=Candidate.from_dict(value)
    if result['signature']!=signature or compact(result,record['directory'],record['manifest'],candidate)!=record:raise ValueError('Changed evaluation record')
    return candidate,result


class Store:
    def __init__(self,study,root,config,stop_after=None):
        self.study=study;self.root=root;self.config=config;self.evaluator=Evaluator(study);self.stop_after=stop_after
        self.entries=sorted((root/'calls').glob('*.json')) if (root/'calls').exists() else []
        if [p.name for p in self.entries]!=[f'{i:06d}.json' for i in range(len(self.entries))]:raise ValueError('Noncontiguous journal')
        self.cache={};self.vectors={};self.history=[];self.best=None;self.pool={};self.fresh=0;self.started=time.perf_counter()

    def evaluate(self,candidate,origin,meta,reference=None):
        i=len(self.history)
        if i>=self.config.proposals:raise Budget()
        if source_hash()!=self.evaluator.code:raise ValueError('Code changed during run')
        cid=candidate.candidate_id;hit=cid in self.cache;t=time.perf_counter()
        deterministic=dict(call=i,candidate_id=cid,origin=origin,meta=plain(meta),cache_hit=hit,
                           changes=None if reference is None else physical_changes(reference,candidate))
        if i<len(self.entries):
            saved=unseal(self.entries[i]);record=saved['record'];verify(self.root,record,self.evaluator.signature)
            if saved['event']!=deterministic:raise ValueError('Journal replay diverged')
        else:
            if hit:record=self.cache[cid];verify(self.root,record,self.evaluator.signature)
            else:
                result=self.evaluator.evaluate(candidate)
                if result['candidate_id']!=cid:raise ValueError('Invalid candidate structure')
                attempt=0;folder=self.root/'evaluations'/f'{i:06d}_{attempt:02d}'
                while folder.exists():attempt+=1;folder=self.root/'evaluations'/f'{i:06d}_{attempt:02d}'
                save_evaluation(folder,self.study,candidate,result,compact=True,shared_study=self.root/'study.json')
                record=compact(result,folder.relative_to(self.root).as_posix(),sha(folder/'manifest.json'),candidate)
            (self.root/'calls').mkdir(exist_ok=True)
            seal(self.root/'calls'/f'{i:06d}.json',dict(event=deterministic,record=record,seconds=time.perf_counter()-t))
            self.fresh+=1
        self.cache[cid]=record;self.vectors[cid]=candidate
        improved=self.best is None or rank(record)<rank(self.best)
        if improved:self.best=record
        # An exploratory archive contains only fully converged violated states.
        # It never changes the strict comparator of the publishable archive.
        if record['status']=='violated' and record['normalized_violation']<=self.config.exploration_limit:
            self.pool[cid]=record
            selected={};ordered=sorted(self.pool.values(),key=rank)
            for r in ordered:
                if r['plan_id'] not in selected:selected[r['plan_id']]=r
                if len(selected)==4:break
            self.pool={r['candidate_id']:r for r in selected.values()}
        self.history.append(dict(**deterministic,record=record,incumbent_improved=improved,best_id=self.best['candidate_id']))
        if self.stop_after is not None and self.fresh>=self.stop_after:raise Pause()
        return record


def search(study,config,output,stop_after=None):
    if stop_after is not None and (type(stop_after)!=int or stop_after<1):raise ValueError('Invalid pause count')
    config=CooperativeConfig(**asdict(config));root=Path(output).resolve();codec=Codec(study)
    with locked(root):
        protocol=dict(version=1,config=asdict(config),input_id=study.input_id,code=source_hash(),
                      financial_policy=resolve(study.c),
                      mode='frozen_context_episodes_native_local_engines',budget='all proposals including episode initialization and cache')
        if (root/'protocol.json').exists():
            if unseal(root/'protocol.json')!=protocol:raise ValueError('Different protocol/code/input')
            if Study(**unseal(root/'study.json')).input_id!=study.input_id:raise ValueError('Different saved study')
        else:
            if any(p.name!='run.lock' for p in root.iterdir()):raise ValueError('Use a new empty directory')
            seal(root/'study.json',dict(data=study.data,electrical=study.electrical));seal(root/'protocol.json',protocol)
        evaluator=Evaluator(study)
        if (root/'result.json').exists():
            r=unseal(root/'result.json')
            if r.get('financial_policy')!=protocol['financial_policy']:
                raise ValueError('Saved cooperative financial policy mismatch')
            paths=sorted((root/'calls').glob('*.json'))
            if len(paths)!=r['proposals'] or fingerprint([sha(p) for p in paths])!=r['journal_id']:raise ValueError('Changed final journal')
            for p in paths:verify(root,unseal(p)['record'],evaluator.signature)
            calls=[unseal(p) for p in paths]
            if r['best']!=min((c['record'] for c in calls),key=rank):raise ValueError('Wrong saved winner')
            reconstructed=[];best=None
            for call in calls:
                rec=call['record'];improved=best is None or rank(rec)<rank(best)
                if improved:best=rec
                reconstructed.append(dict(**call['event'],record=rec,incumbent_improved=improved,best_id=best['candidate_id']))
            if reconstructed!=r['history']:raise ValueError('Changed final history')
            verify(root,r['best'],evaluator.signature);audited=audit_saved(root/r['best']['directory'])
            if fingerprint(audited['metrics'])!=fingerprint(r['metrics']):raise ValueError('Changed final metrics')
            return r
        invocation=unseal(root/'invocation.json') if (root/'invocation.json').exists() else dict(count=0,seconds=0.)
        invocation['count']+=1;seal(root/'invocation.json',invocation,replace=(root/'invocation.json').exists())
        store=Store(study,root,config,stop_after);started=time.perf_counter();initial_ids=[];initial_vectors=[]
        initial_seeds=np.random.SeedSequence([config.seed,17031]).generate_state(config.proposals)
        sampler=np.random.default_rng(np.random.SeedSequence([config.seed,44771]));episode=0;spaces=[]
        try:
            for i in range(config.population):
                c,x=initial_candidate(codec,int(initial_seeds[i]));initial_ids.append(c.candidate_id);initial_vectors.append(x)
                store.evaluate(c,'initial',dict(seed=int(initial_seeds[i])))
            if config.method=='CONSTRUCT':
                for i in range(config.population,config.proposals):
                    c,_=initial_candidate(codec,int(initial_seeds[i]))
                    store.evaluate(c,'construction',dict(seed=int(initial_seeds[i])))
                raise Budget()
            while True:
                if len(store.history)>=config.proposals:raise Budget()
                if config.restart_every and episode and episode%config.restart_every==0:
                    seed=int(np.random.SeedSequence([config.seed,77281,episode]).generate_state(1)[0])
                    store.evaluate(initial_candidate(codec,seed)[0],'restart',dict(seed=seed,episode=episode))
                context=store.best;exploring=False
                if config.exploratory and episode%3==2 and store.pool:
                    pool=sorted(store.pool.values(),key=lambda r:r['candidate_id']);context=pool[int(sampler.integers(len(pool)))];exploring=True
                base=store.vectors[context['candidate_id']]
                family=FAMILIES[episode%len(FAMILIES)]
                space=make_space(codec,base,family,sampler,config.radius,config.coupled,config.global_probability,config.spatial_hops,config.all_cycles)
                space_id=fingerprint(space.spec);epath=root/'episodes'/f'{episode:04d}.npz'
                payload=dict(spec=space.spec,base=base.to_dict(),exploring=exploring)
                if epath.exists():
                    if fingerprint(unseal(epath))!=fingerprint(payload):raise ValueError('Episode context changed')
                else:epath.parent.mkdir(exist_ok=True);seal(epath,payload)
                spaces.append(dict(episode=episode,space_id=space_id,family=space.spec['family'],dimension=space.dimension,exploring=exploring))
                seed=int(np.random.SeedSequence([config.seed,11837,episode]).generate_state(1)[0]);rng=np.random.default_rng(seed)
                initial=space.initial(config.population,rng)
                if config.method=='COMMON':
                    for j in range(config.population*(config.episode_generations+1)):
                        y=initial[j] if j<config.population else rng.random(space.dimension)
                        c,details=space.decode(y)
                        store.evaluate(c,'neighborhood_initial' if j<config.population else 'common',
                                       dict(episode=episode,space_id=space_id,y=y,details=details),base)
                else:
                    ec=SearchConfig(config.method,population=config.population,generations=config.episode_generations,seed=seed,
                                    comparison_profile='native_operators',parameters={'sigma_floor':.001} if config.method=='ACO' else {})
                    engine=Engine(ec,initial)
                    while not engine.done:
                        y=engine.ask();c,details=space.decode(y)
                        rec=store.evaluate(c,'neighborhood_initial' if engine.generation==0 else 'native',
                                           dict(episode=episode,space_id=space_id,y=y,details=details),base)
                        engine.tell(y,rec)
                episode+=1
        except Pause:
            invocation['seconds']+=time.perf_counter()-started;seal(root/'invocation.json',invocation,replace=True)
            return dict(status='paused',proposals=len(store.history),financial_policy=protocol['financial_policy'])
        except Budget:pass
        seconds=invocation['seconds']+time.perf_counter()-started
        invocation['seconds']=seconds;seal(root/'invocation.json',invocation,replace=True)
        audited=audit_saved(root/store.best['directory'])
        result=dict(status='completed',config=asdict(config),code=source_hash(),input_id=study.input_id,
                    financial_policy=protocol['financial_policy'],
                    initial_ids=initial_ids,initial_id=fingerprint(initial_vectors),proposals=len(store.history),unique_AC=len(store.cache),
                    best=store.best,metrics=audited['metrics'],history=store.history,episodes=spaces,
                    search_seconds=seconds,timing_eligible=invocation['count']==1,invocations=invocation['count'],
                    journal_id=fingerprint([sha(p) for p in sorted((root/'calls').glob('*.json'))]))
        if len(store.history)!=config.proposals:raise ValueError('Incomplete fixed budget')
        seal(root/'result.json',result);return plain(result)
