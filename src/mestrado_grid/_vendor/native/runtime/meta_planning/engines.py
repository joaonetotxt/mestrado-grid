"""Seven ask/tell engines over common normalized keys; no dispatch optimizer.

The decoder supplies the actually evaluated coordinates to tell(). Feasibility
ordering, not an arbitrary economic penalty, controls every selection operator.
See METHODS.md for equations, original sources and declared adaptations.
"""
from dataclasses import dataclass,field,asdict
import copy
import numpy as np
from scipy.special import gammaincinv
from .evaluator import rank
from .schema import plain

DEFAULTS={
    'PSO':dict(inertia=.7298,cognitive=1.49618,social=1.49618,velocity_limit=.2,initial_velocity_fraction=.5),
    'HyDE':dict(adaptation_probability=.1,F_initial=.5,CR_initial=.5,F_lower=.1,F_span=.9),
    'GA':dict(mutation_rate=None,tournament_size=2,force_mutation=0),
    'ACO':dict(q=.5,xi=.85,sigma_floor=0.),
    'Vortex':dict(gamma_probability=.1,radius_floor=0.),
    'QPSO':dict(alpha_start=1.,alpha_end=.5),
    'SHADE':dict(memory_size=100,p_max=.2),
}
METHODS=tuple(DEFAULTS)


@dataclass(frozen=True)
class SearchConfig:
    method: str
    population: int=4
    generations: int=2  # offspring batches, AFTER evaluating the common start
    seed: int=0
    wall_seconds: float=600.
    initialization: str='mixed'
    parameters: dict=field(default_factory=dict)
    update_scheme: str='joint'
    block_step: float=.01
    persistence: str='json'
    coordination: str='none'
    vortex_schedule: str='generations'
    progress_schedule: str='generations'  # budget: runner supplies normalized actual budget
    comparison_profile: str='declared_variant'
    categorical_policy: str='none'
    subset_fraction: float=.001
    categorical_rate: float=.125
    direction_source: str='native'
    step_levels: list=field(default_factory=list)
    restart_patience: int=0
    restart_fraction: float=.5
    diversity_floor: float=1e-5

    def __post_init__(self):
        if self.method not in DEFAULTS:raise ValueError('Unknown search method')
        if type(self.population)!=int or self.population<4 or type(self.generations)!=int or self.generations<0:
            raise ValueError('Population >=4 and generations >=0 must be integers')
        if type(self.seed)!=int or self.seed<0:raise ValueError('Nonnegative integer seed required')
        if isinstance(self.wall_seconds,bool) or not np.isfinite(self.wall_seconds) or self.wall_seconds<=0:raise ValueError('Positive wall budget required')
        if self.initialization not in ('constructive','random','mixed','physical','resource','mat_original'):raise ValueError('Unknown initialization')
        if self.update_scheme not in ('joint','blocks','bounded_joint','sparse_local','support_local','dependency'):raise ValueError('Unknown update scheme')
        if isinstance(self.block_step,bool) or not isinstance(self.block_step,(int,float)) or not np.isfinite(self.block_step) or not 0<self.block_step<=1:raise ValueError('Invalid block step')
        if self.persistence not in ('json','compact'):raise ValueError('Unknown persistence format')
        if self.coordination not in ('none','load_balance','service','local_service'):raise ValueError('Unknown coordination')
        if self.comparison_profile not in ('declared_variant','native_operators'):raise ValueError('Unknown comparison profile')
        if self.comparison_profile=='native_operators' and (self.update_scheme!='joint' or self.coordination!='none'):
            raise ValueError('Native-operator comparison requires joint updates and no common search coordination')
        if self.categorical_policy not in ('none','uniform_reset','mixed_reset','local_mixed','dependency_mixed'):raise ValueError('Unknown categorical policy')
        if self.comparison_profile=='native_operators' and self.categorical_policy!='none':
            raise ValueError('Categorical variation is a declared mixed variant, not native operators')
        if (self.update_scheme=='sparse_local')!=(self.categorical_policy=='local_mixed'):
            raise ValueError('Sparse local and local mixed policy must be used together')
        new_mode=self.update_scheme in ('support_local','dependency')
        if new_mode!=(self.categorical_policy=='dependency_mixed'):raise ValueError('Dependency variants require dependency_mixed')
        if self.direction_source not in ('native','common'):raise ValueError('Invalid direction source')
        if not isinstance(self.step_levels,(list,tuple)) or any(isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<x<=1 for x in self.step_levels):raise ValueError('Invalid step levels')
        object.__setattr__(self,'step_levels',list(self.step_levels))
        if type(self.restart_patience)!=int or self.restart_patience<0:raise ValueError('Invalid restart patience')
        for name in ('restart_fraction','diversity_floor'):
            x=getattr(self,name)
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not np.isfinite(x) or not 0<x<=1:raise ValueError('Invalid '+name)
        if not new_mode and (self.direction_source!='native' or self.step_levels or self.restart_patience or self.restart_fraction!=.5 or self.diversity_floor!=1e-5):raise ValueError('Diversification settings require a new variant')
        if new_mode and self.coordination!='none':raise ValueError('Dependency variants prohibit additional coordination')
        for name in ('subset_fraction','categorical_rate'):
            value=getattr(self,name)
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not np.isfinite(value) or not 0<value<=1:raise ValueError('Invalid local parameter: '+name)
        if self.update_scheme not in ('sparse_local','support_local','dependency') and (self.subset_fraction!=.001 or self.categorical_rate!=.125):raise ValueError('Local parameters require a local variant')
        if self.progress_schedule not in ('generations','budget'):raise ValueError('Unknown progress schedule')
        if self.vortex_schedule not in ('generations','wall'):raise ValueError('Unknown Vortex schedule')
        if self.method!='Vortex' and self.vortex_schedule!='generations':raise ValueError('Wall radius schedule is Vortex only')
        p=copy.deepcopy(DEFAULTS[self.method])
        if set(self.parameters)-set(p):raise ValueError('Unsupported method parameter')
        p.update(self.parameters)
        for k,v in p.items():
            if v is None and k=='mutation_rate':continue
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or v<0:raise ValueError('Invalid parameter: '+k)
        if self.method=='PSO' and (not 0<p['velocity_limit']<=1 or p['initial_velocity_fraction']>1):raise ValueError('Invalid velocity limit')
        if self.method=='HyDE' and (any(p[k]>1 for k in ('adaptation_probability','F_initial','CR_initial')) or p['F_lower']+p['F_span']>1):raise ValueError('Invalid HyDE parameters')
        if self.method=='GA' and (type(p['tournament_size'])!=int or not 2<=p['tournament_size']<=self.population or p['mutation_rate'] is not None and p['mutation_rate']>1):raise ValueError('Invalid GA parameters')
        if self.method=='ACO' and (p['q']<=0 or p['xi']<=0 or p['sigma_floor']>1):raise ValueError('Invalid ACO_R parameters')
        if self.method=='Vortex' and not 0<p['gamma_probability']<1:raise ValueError('Invalid gamma probability')
        if self.method=='Vortex' and not 0<=p['radius_floor']<=1:raise ValueError('Invalid radius floor')
        if self.method=='GA' and (type(p['force_mutation']) is not int or p['force_mutation'] not in (0,1)):raise ValueError('Invalid forced mutation switch')
        if self.method=='QPSO' and not 0<p['alpha_end']<=p['alpha_start']<=2:raise ValueError('Invalid QPSO contraction schedule')
        if self.method=='SHADE' and (type(p['memory_size'])!=int or not 1<=p['memory_size']<=10000 or not 0<p['p_max']<=1):raise ValueError('Invalid SHADE parameters')
        object.__setattr__(self,'parameters',p)


def key(record):
    return (4,float('inf')) if record is None else rank(record)


class Engine:
    def __init__(self,config,initial,groups=None,categorical=None,exploration=None,dependencies=None):
        self.config=config;self.n=config.population
        self.population=np.asarray(initial,float).copy()
        if self.population.ndim!=2 or self.population.shape[0]!=self.n or self.population.shape[1]<1 or not np.all(np.isfinite(self.population)) or np.any((self.population<0)|(self.population>1)):
            raise ValueError('Invalid common initial population')
        self.d=self.population.shape[1]
        self.groups=copy.deepcopy(groups if groups is not None else [[[0,self.d]]])
        covered=np.zeros(self.d,dtype=int)
        for group in self.groups:
            if not group:raise ValueError('Empty coordinate group')
            for pair in group:
                if len(pair)!=2 or any(type(x)!=int for x in pair) or not 0<=pair[0]<pair[1]<=self.d:raise ValueError('Invalid block range')
                covered[pair[0]:pair[1]]+=1
        if not self.groups or np.any(covered!=1):raise ValueError('Groups must partition every coordinate')
        from .categorical import validate_spec
        self.categorical=validate_spec(categorical or [],self.d)
        if config.categorical_policy=='uniform_reset' and not self.categorical:raise ValueError('Categorical domains required')
        self.categorical_pending=[None]*self.n
        from .exploration import validate_exploration
        if config.categorical_policy in ('mixed_reset','local_mixed','dependency_mixed') and exploration is None:raise ValueError('Explicit mixed domains required')
        self.exploration=validate_exploration(exploration or dict(continuous=[],links=[]),self.categorical,self.d)
        if config.categorical_policy in ('mixed_reset','local_mixed','dependency_mixed'):
            described={r['start']+j for r in self.categorical+self.exploration['continuous'] for j in range(len(r['sizes']))}
            if described!=set(range(self.d)):raise ValueError('Mixed domains must cover every search coordinate')
        if config.update_scheme=='sparse_local' and not self.exploration['continuous']:raise ValueError('Sparse local requires continuous domains')
        self.local_pending=[None]*self.n
        from .dependency_variation import validate_partition
        self.dependencies=validate_partition(dependencies,self.d) if dependencies is not None else None
        if config.update_scheme in ('support_local','dependency') and self.dependencies is None:raise ValueError('Dependency partition required')
        self.variation_pending=[None]*self.n;self.restart_pending=[None]*self.n
        self.restart_count=0;self.last_restart_generation=0
        self.local_rng_state=np.random.default_rng(np.random.SeedSequence([config.seed,93821])).bit_generator.state
        self.continuous_pending=[None]*self.n;self.activation_pending=[None]*self.n
        # Independent stream from the one used to construct the common start.
        self.rng=np.random.default_rng(np.random.SeedSequence([config.seed,41821]))
        self.pending=self.population.copy();self.received=[];self.records=[None]*self.n
        self.native_pending_credit=[True]*self.n
        self.generation=0;self.cursor=0;self.evaluations=0;self.best=None
        self.budget_progress=0.;self.last_vortex_radius=None
        self.velocity=np.zeros_like(self.population)
        if config.method=='PSO':
            v=config.parameters['velocity_limit']*config.parameters['initial_velocity_fraction']
            self.velocity=self.rng.uniform(-v,v,self.population.shape)
        self.personal=self.population.copy();self.personal_records=[None]*self.n
        self.factors=np.full((self.n,3),config.parameters.get('F_initial',.5))
        self.crossovers=np.full(self.n,config.parameters.get('CR_initial',.5))
        self.trial_factors=self.factors.copy();self.trial_crossovers=self.crossovers.copy()
        if config.method=='SHADE':
            from .advanced_engines import initialize_shade
            self.shade=initialize_shade(config.parameters['memory_size'],self.n)

    @property
    def done(self):return self.generation>self.config.generations

    def ask(self):
        if self.done:raise StopIteration('Search budget of generations completed')
        return self.pending[self.cursor].copy()

    def replace_pending(self,positions,native_credit):
        """Explicit hybrid proposals; only native moves train native parameters."""
        a=np.asarray(positions,float)
        if self.cursor or self.generation==0 or self.done:raise ValueError('Replacement requires a fresh offspring batch')
        if a.shape!=self.pending.shape or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid replacement proposals')
        if len(native_credit)!=self.n or any(type(x)!=bool for x in native_credit):raise ValueError('Explicit provenance required')
        if self.config.method=='PSO':
            self.velocity[abs(a-self.pending)>1e-12]=0.
            self.velocity[np.logical_not(native_credit)]=0.
        self.pending=a.copy();self.native_pending_credit=list(native_credit)

    def set_budget_progress(self,progress):
        if isinstance(progress,bool) or not np.isfinite(progress) or not 0<=progress<=1:raise ValueError('Invalid budget progress')
        self.budget_progress=max(self.budget_progress,float(progress))

    def tell(self,evaluated_position,record):
        if self.done:raise ValueError('No pending candidate')
        x=np.asarray(evaluated_position,float)
        if x.shape!=(self.d,) or not np.all(np.isfinite(x)) or np.any((x<0)|(x>1)):raise ValueError('Invalid evaluated position')
        rank(record)  # finite fitness is required before changing state
        if self.config.method=='PSO':
            self.velocity[self.cursor,abs(x-self.pending[self.cursor])>1e-12]=0.
        self.pending[self.cursor]=x;record=copy.deepcopy(record);self.received.append(record)
        if self.best is None or key(record)<key(self.best['record']):self.best=dict(x=x.copy(),record=record)
        self.evaluations+=1;self.cursor+=1
        if self.cursor==self.n:self._finish_generation()

    def _finish_generation(self):
        method=self.config.method
        restarted=[(i,self.pending[i].copy(),copy.deepcopy(self.received[i])) for i,e in enumerate(self.restart_pending) if e is not None]
        if self.generation==0:
            self.population=self.pending.copy();self.records=copy.deepcopy(self.received)
        elif self.config.direction_source=='common':
            self.population=self.pending.copy();self.records=copy.deepcopy(self.received)
        elif method=='HyDE':
            for i,r in enumerate(self.received):
                if key(r)<key(self.records[i]):
                    self.population[i]=self.pending[i];self.records[i]=r
                    if self.native_pending_credit[i]:
                        self.factors[i]=self.trial_factors[i];self.crossovers[i]=self.trial_crossovers[i]
        elif method=='SHADE':
            from .advanced_engines import accept_shade
            accept_shade(self,key)
        elif method=='ACO':
            pop=np.vstack((self.population,self.pending));records=self.records+self.received
            order=sorted(range(len(records)),key=lambda i:key(records[i]))[:self.n]
            self.population=pop[order];self.records=[records[i] for i in order]
        else:
            self.population=self.pending.copy();self.records=copy.deepcopy(self.received)
        # Restarted rows enter only AFTER their own evaluation. This also frees
        # greedy HyDE/ACO memories; the separate best archive never regresses.
        for i,x,r in restarted:
            self.population[i]=x;self.records[i]=r;self.personal[i]=x;self.personal_records[i]=r
            self.velocity[i]=0.;self.factors[i]=self.config.parameters.get('F_initial',.5)
            self.crossovers[i]=self.config.parameters.get('CR_initial',.5)
        if method in ('PSO','QPSO'):
            for i,r in enumerate(self.records):
                if self.personal_records[i] is None or key(r)<key(self.personal_records[i]):
                    self.personal[i]=self.population[i];self.personal_records[i]=r
        self.generation+=1;self.cursor=0;self.received=[]
        self.native_pending_credit=[True]*self.n
        if not self.done:self.pending=self._propose()

    def _reenter(self,x):
        outside=(x<0)|(x>1)
        x[outside]=self.rng.random(np.count_nonzero(outside));return x

    def active_group(self):
        return None if self.generation==0 or self.config.update_scheme!='blocks' else (self.generation-1)%len(self.groups)

    def _propose(self):
        if self.config.direction_source=='common':
            # A true common-direction baseline bypasses the five native formulas.
            rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(self.local_rng_state)
            proposed=rng.random(self.population.shape);self.local_rng_state=copy.deepcopy(rng.bit_generator.state)
            proposed[0]=self.best['x']
        else:proposed=self._propose_full()
        self.variation_pending=[None]*self.n;self.restart_pending=[None]*self.n
        self.local_pending=[None]*self.n
        if self.config.update_scheme in ('support_local','dependency'):return self._propose_dependency(proposed)
        if self.config.update_scheme=='sparse_local':return self._propose_local(proposed)
        group=self.active_group()
        if self.config.update_scheme=='joint':return self._categorical_propose(proposed)
        # Common contextual block variant: all full-vector memories retain their
        # actual fitness; no fitness is borrowed from an unevaluated sub-vector.
        # Structure has categorical jumps; continuous blocks have a declared
        # step radius in normalized keys. Decoder-induced changes are still logged.
        best=self.best['x'];out=np.broadcast_to(best,proposed.shape).copy()
        active=np.zeros(self.d,bool)
        for start,stop in (self.groups[group] if self.config.update_scheme=='blocks' else [r for g in self.groups for r in g]):active[start:stop]=True
        delta=proposed[:,active]-best[active]
        radius=np.full(self.d,self.config.block_step)
        for start,stop in self.groups[0]:radius[start:stop]=1.
        step=radius[active]
        out[:,active]=best[active]+np.clip(delta,-step,step)
        if self.config.method=='PSO':
            self.velocity[:,~active]=0.
            self.velocity[:,active]=out[:,active]-self.population[:,active]
        return self._categorical_propose(out)

    def _propose_local(self,proposed):
        from .local_variation import propose
        self.categorical_pending=[None]*self.n;self.continuous_pending=[None]*self.n;self.activation_pending=[None]*self.n
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(self.local_rng_state)
        reference=self.best['x'];out=np.broadcast_to(reference,proposed.shape).copy()
        for i in range(self.n):
            if self.config.method=='GA' and i==0:continue
            out[i],self.categorical_pending[i],self.continuous_pending[i],self.activation_pending[i],self.local_pending[i]=propose(
                proposed[i],reference,self.categorical,self.exploration,self.config.block_step,
                self.config.subset_fraction,self.config.categorical_rate,rng)
        self.local_rng_state=copy.deepcopy(rng.bit_generator.state)
        if self.config.method=='PSO':self.velocity=out-self.population
        return out

    def _propose_dependency(self,proposed):
        from .dependency_variation import propose
        self.categorical_pending=[None]*self.n;self.continuous_pending=[None]*self.n;self.activation_pending=[None]*self.n
        rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(self.local_rng_state)
        out=self.population.copy()
        for i in range(self.n):
            if i==0 and (self.config.method=='GA' or self.config.direction_source=='common'):
                out[i]=self.best['x'];continue
            out[i],self.categorical_pending[i],self.continuous_pending[i],self.activation_pending[i],self.variation_pending[i]=propose(
                proposed[i],self.population[i],self.categorical,self.exploration,self.dependencies,self.config,rng)
        self.local_rng_state=copy.deepcopy(rng.bit_generator.state)
        if self.config.method=='PSO':self.velocity=out-self.population
        return out

    def install_restart(self,indices,positions,events):
        if self.cursor or self.generation==0 or self.done or not self.config.restart_patience or any(self.restart_pending):raise ValueError('Invalid restart boundary')
        if len(indices)!=len(positions) or len(events)!=len(indices) or len(set(indices))!=len(indices) or not 0<len(indices)<self.n:raise ValueError('Invalid restart population')
        for i,x,event in zip(indices,positions,events):
            x=np.asarray(x,float)
            if type(i)!=int or not 0<=i<self.n or x.shape!=(self.d,) or not np.isfinite(x).all() or np.any((x<0)|(x>1)):raise ValueError('Invalid restart row')
            self.pending[i]=x;self.restart_pending[i]=copy.deepcopy(event)
            self.variation_pending[i]=self.local_pending[i]=self.categorical_pending[i]=self.continuous_pending[i]=self.activation_pending[i]=None
            if self.config.method=='PSO':self.velocity[i]=0.
        self.restart_count+=1;self.last_restart_generation=self.generation

    def _categorical_propose(self,out):
        from .categorical import reset_one
        self.categorical_pending=[None]*self.n
        self.continuous_pending=[None]*self.n;self.activation_pending=[None]*self.n
        if self.config.categorical_policy=='none':return out
        active=np.ones(self.d,bool)
        group=self.active_group()
        if group is not None:
            active[:]=False
            for start,stop in self.groups[group]:active[start:stop]=True
        for i in range(self.n):
            if self.config.method=='GA' and i==0:continue  # preserve the elite
            self.categorical_pending[i]=reset_one(out[i],self.categorical,active,self.rng)
            if self.config.method=='PSO' and self.categorical_pending[i] is not None:
                self.velocity[i,self.categorical_pending[i]['index']]=0.
            if self.config.categorical_policy in ('mixed_reset','local_mixed'):
                from .exploration import activate_one
                self.continuous_pending[i]=reset_one(out[i],self.exploration['continuous'],active,self.rng)
                self.activation_pending[i]=activate_one(out[i],self.categorical_pending[i],self.exploration['links'],active,self.rng)
                if self.config.method=='PSO':
                    for event in (self.continuous_pending[i],self.activation_pending[i]):
                        if event is not None:self.velocity[i,event['index']]=0.
        return out

    def _propose_full(self):
        p=self.config.parameters;rng=self.rng;pop=self.population;best=self.best['x'];n,d=self.n,self.d
        if self.config.method=='QPSO':
            from .advanced_engines import propose_qpso
            return propose_qpso(self)
        if self.config.method=='SHADE':
            from .advanced_engines import propose_shade
            return propose_shade(self,key)
        if self.config.method=='PSO':
            self.velocity=p['inertia']*self.velocity+p['cognitive']*rng.random((n,d))*(self.personal-pop)+p['social']*rng.random((n,d))*(best-pop)
            self.velocity=np.clip(self.velocity,-p['velocity_limit'],p['velocity_limit'])
            out=pop+self.velocity;self.velocity[(out<0)|(out>1)]=0.
            return np.clip(out,0,1)
        if self.config.method=='HyDE':
            f=self.factors.copy();cr=self.crossovers.copy();tau=p['adaptation_probability']
            change=rng.random(f.shape)<tau;f=np.where(change,p['F_lower']+p['F_span']*rng.random(f.shape),f)
            cr=np.where(rng.random(n)<tau,rng.random(n),cr)
            self.trial_factors=f;self.trial_crossovers=cr;out=np.empty_like(pop)
            for i in range(n):
                r1,r2=rng.choice([j for j in range(n) if j!=i],2,replace=False)
                mutant=pop[i]+f[i,0]*(best*rng.normal(f[i,1],1.)-pop[i])+f[i,2]*(pop[r1]-pop[r2])
                mask=rng.random(d)<cr[i];mask[rng.integers(d)]=True
                out[i]=np.where(mask,mutant,pop[i])
            return self._reenter(out)
        if self.config.method=='GA':
            def parent():
                ids=rng.choice(n,p['tournament_size'],replace=False)
                return pop[min(ids,key=lambda j:key(self.records[j]))]
            out=np.empty_like(pop);out[0]=best  # one global elite, still counted if cached
            rate=1/d if p['mutation_rate'] is None else p['mutation_rate']
            for i in range(1,n):
                a,b=parent(),parent();out[i]=np.where(rng.random(d)<.5,a,b)
                mask=rng.random(d)<rate
                if p['force_mutation'] and not mask.any():mask[rng.integers(d)]=True
                out[i,mask]=rng.random(np.count_nonzero(mask))
            return out
        if self.config.method=='ACO':
            order=sorted(range(n),key=lambda j:key(self.records[j]));archive=pop[order]
            logw=-.5*(np.arange(n)/(p['q']*n))**2;weights=np.exp(logw-logw.max());weights/=weights.sum()
            out=np.empty_like(pop)
            for i in range(n):
                j=rng.choice(n,p=weights)
                sigma=p['xi']*np.sum(abs(archive-archive[j]),axis=0)/(n-1)
                out[i]=rng.normal(archive[j],np.maximum(sigma,p['sigma_floor']))
            return self._reenter(out)
        # Single-center Vortex, gamma inverse with SciPy argument order (a,p).
        a=max(1e-6,1-self.budget_progress) if self.config.vortex_schedule=='wall' or self.config.progress_schedule=='budget' else 1-(self.generation-1)/max(1,self.config.generations)
        prob=p['gamma_probability'];radius=max(p['radius_floor'],.5*gammaincinv(a,prob)/prob)
        self.last_vortex_radius=float(radius)
        return self._reenter(rng.normal(best,radius,size=(n,d)))

    def to_dict(self,native=False):
        value=dict(version=4,config=asdict(self.config),rng=self.rng.bit_generator.state,
            **{k:v for k,v in vars(self).items() if k not in ('config','rng')})
        return value if native else plain(value)

    @classmethod
    def from_dict(cls,value):
        if value['version']!=4:raise ValueError('Unsupported engine checkpoint')
        value=copy.deepcopy(value)
        value.setdefault('native_pending_credit',[True]*len(value['population']))
        obj=cls(SearchConfig(**value['config']),value['population'],value['groups'],value['categorical'],value['exploration'],value['dependencies'])
        if set(value)!=set(obj.to_dict()):raise ValueError('Incomplete engine state')
        arrays={'population','pending','velocity','personal','factors','crossovers','trial_factors','trial_crossovers'}
        for k,v in value.items():
            if k in ('version','config','rng'):continue
            if k not in vars(obj):raise ValueError('Unexpected engine state')
            setattr(obj,k,np.asarray(v,float) if k in arrays else copy.deepcopy(v))
        if obj.best is not None:obj.best['x']=np.asarray(obj.best['x'],float)
        obj.rng.bit_generator.state=value['rng']
        from .categorical import validate_spec,validate_events
        validate_spec(obj.categorical,obj.d)
        validate_events(obj.categorical_pending,obj.categorical,obj.n)
        from .exploration import validate_exploration,validate_mixed_events
        validate_exploration(obj.exploration,obj.categorical,obj.d)
        validate_mixed_events(obj.continuous_pending,obj.activation_pending,obj.categorical_pending,obj.exploration,obj.n)
        from .local_variation import validate_event
        probe=np.random.default_rng();probe.bit_generator.state=copy.deepcopy(obj.local_rng_state)
        if not isinstance(obj.local_pending,list) or len(obj.local_pending)!=obj.n:raise ValueError('Invalid local event population')
        for i,event in enumerate(obj.local_pending):
            eligible=obj.config.update_scheme=='sparse_local' and (obj.evaluations>obj.n or obj.generation>0 and not obj.done) and not (obj.config.method=='GA' and i==0)
            if eligible!=(event is not None):raise ValueError('Local pending event eligibility mismatch')
            validate_event(event,obj.exploration,obj.d,obj.config.block_step,obj.config.subset_fraction,obj.config.categorical_rate)
        from .dependency_variation import validate_event as validate_dependency,validate_restart
        if any(not isinstance(getattr(obj,n),list) or len(getattr(obj,n))!=obj.n for n in ('variation_pending','restart_pending')):raise ValueError('Invalid diversification events')
        if any(type(getattr(obj,n))!=int or getattr(obj,n)<0 for n in ('restart_count','last_restart_generation')) or obj.last_restart_generation>obj.generation or (obj.restart_count==0)!=(obj.last_restart_generation==0):raise ValueError('Invalid restart counters')
        if obj.restart_count>obj.last_restart_generation or not obj.config.restart_patience and obj.restart_count:raise ValueError('Impossible restart count')
        for i,event in enumerate(obj.variation_pending):
            restart=obj.restart_pending[i]
            if restart is not None:
                validate_restart(restart,obj.config,obj.generation-1 if obj.done else obj.generation,i)
                if restart['number']!=obj.restart_count or restart['generation']!=obj.last_restart_generation:raise ValueError('Restart pending/counter mismatch')
                if not obj.done and i>=obj.cursor:
                    from .schema import fingerprint
                    if restart['proposed_genome_id']!=fingerprint(obj.pending[i]):raise ValueError('Restart pending position mismatch')
            eligible=obj.config.update_scheme in ('support_local','dependency') and (obj.evaluations>obj.n or obj.generation>0 and not obj.done) and not (i==0 and (obj.config.method=='GA' or obj.config.direction_source=='common')) and restart is None
            if eligible!=(event is not None):raise ValueError('Dependency event eligibility mismatch')
            if event is not None:
                validate_dependency(event,obj.dependencies,obj.config,obj.d)
                if not obj.done and i>=obj.cursor:
                    from .schema import fingerprint
                    if event['proposal_genome_id']!=fingerprint(obj.pending[i]) or event['reference_genome_id']!=fingerprint(obj.population[i]):raise ValueError('Dependency pending/reference mismatch')
        if isinstance(obj.budget_progress,bool) or not isinstance(obj.budget_progress,(int,float)) or not np.isfinite(obj.budget_progress) or not 0<=obj.budget_progress<=1:raise ValueError('Invalid saved budget progress')
        if obj.last_vortex_radius is not None and (isinstance(obj.last_vortex_radius,bool) or not isinstance(obj.last_vortex_radius,(int,float)) or not np.isfinite(obj.last_vortex_radius) or obj.last_vortex_radius<0):raise ValueError('Invalid saved Vortex radius')
        if any(type(getattr(obj,name))!=int for name in ('n','d','generation','cursor','evaluations')):
            raise ValueError('Invalid checkpoint integer counters')
        if obj.cursor!=len(obj.received) or obj.evaluations!=obj.generation*obj.n+obj.cursor:
            raise ValueError('Inconsistent checkpoint cursor')
        if obj.n!=obj.config.population or obj.d!=obj.population.shape[1] or type(obj.generation)!=int or not 0<=obj.generation<=obj.config.generations+1 or type(obj.cursor)!=int or not 0<=obj.cursor<obj.n:
            raise ValueError('Invalid checkpoint dimensions/counters')
        if obj.done and obj.cursor!=0:raise ValueError('A completed engine cannot have a partial generation')
        if (obj.best is None)!=(obj.evaluations==0):raise ValueError('Checkpoint best/counter mismatch')
        if obj.generation==0 and any(r is not None for r in obj.records+obj.personal_records):raise ValueError('Premature population fitness')
        if obj.generation>0 and (any(r is None for r in obj.records) or obj.config.method in ('PSO','QPSO') and any(r is None for r in obj.personal_records)):
            raise ValueError('Missing population fitness')
        for name,shape in [('pending',(obj.n,obj.d)),('personal',(obj.n,obj.d)),('velocity',(obj.n,obj.d)),('factors',(obj.n,3)),('trial_factors',(obj.n,3)),('crossovers',(obj.n,)),('trial_crossovers',(obj.n,))]:
            a=getattr(obj,name)
            if a.shape!=shape or not np.all(np.isfinite(a)):raise ValueError('Invalid checkpoint array: '+name)
            if name!='velocity' and np.any((a<0)|(a>1)):raise ValueError('Out-of-range checkpoint array')
        if len(obj.records)!=obj.n or len(obj.personal_records)!=obj.n:raise ValueError('Incomplete population records')
        if len(obj.native_pending_credit)!=obj.n or any(type(x)!=bool for x in obj.native_pending_credit):raise ValueError('Invalid native attribution')
        if obj.best is not None and (obj.best['x'].shape!=(obj.d,) or not np.isfinite(obj.best['x']).all() or np.any((obj.best['x']<0)|(obj.best['x']>1))):raise ValueError('Invalid best position')
        if not obj.done:
            from .categorical import category
            for i in range(obj.cursor,obj.n):
                events=(obj.categorical_pending[i],obj.continuous_pending[i],obj.activation_pending[i])
                for j,event in enumerate(events):
                    if event is None:continue
                    if any(later is not None and later['index']==event['index'] for later in events[j+1:]):continue
                    if category(obj.pending[i,event['index']],event['size'])!=event['target']:
                        raise ValueError('Pending mutation target differs from the proposed position')
        if obj.config.method=='SHADE':
            from .advanced_engines import validate_shade
            validate_shade(obj)
        return obj
