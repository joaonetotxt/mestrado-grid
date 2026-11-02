"""Persistent cooperative native search; economic selection is unchanged.

Service blocks receive recurring, seed-only coverage. Native engines choose all
offspring values. Initialization/diversification and context evaluations have
separate provenance and consume the same proposal budget. No power-flow repair,
preferred bus, technology, historical solution or cut-first comparator is used.
"""
import copy
import hashlib
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

from .balanced_blocks import BalancedScheduler, physical_key
from .native_blocks import Block, Field
from .engines import Engine, SearchConfig, key
from .evaluator import rank


@dataclass(frozen=True)
class PersistentOptions:
    generations_per_visit: int = 8
    cooling_generations: int = 64
    min_unique_fraction: float = .75
    coverage_contract: str = 'all_cases_and_families'
    strategy: str = 'consolidated'
    radii: tuple = (.02,.1,.5,1.)
    engine_parameters: dict = field(default_factory=dict)
    ga_expected_mutations: float | None = None
    topology_dispatch_generations: int = 3
    continuation_generations: int = 0
    continuation_fraction: float = .25
    offspring_source: str = 'native'
    measured_guidance: bool = True
    economic_allocation: bool = True
    active_constraints: bool = False
    case_pq_focus: bool = False
    pq_adaptive_windows: bool = False
    pq_fast_coverage: bool = False
    risk_native_acceleration: bool = False
    risk_native_reliability: bool = False
    protected_resource_coverage: bool = False

    def __post_init__(self):
        if type(self.protected_resource_coverage) is not bool or (self.protected_resource_coverage and not self.risk_native_acceleration):
            raise ValueError('Protected resources require native risk scheduling')
        if type(self.risk_native_reliability) is not bool or (self.risk_native_reliability and not self.risk_native_acceleration):
            raise ValueError('Experimental risk reliability requires explicit native acceleration')
        if type(self.risk_native_acceleration) is not bool:
            raise ValueError('Boolean risk_native_acceleration required')
        if self.risk_native_acceleration and (self.strategy!='consolidated' or not self.active_constraints or self.offspring_source!='native'):
            raise ValueError('Risk acceleration requires consolidated native active-constraint search')
        if self.strategy not in ('baseline','structured','diagnostic','physical','productive','energy_linked','network_linked','consolidated','native_construction'):
            raise ValueError('Unknown persistent exploration strategy')
        if type(self.continuation_generations) is not int or self.continuation_generations < 0:
            raise ValueError('Nonnegative continuation generations required')
        if isinstance(self.continuation_fraction,bool) or not np.isfinite(self.continuation_fraction) or not 0 <= self.continuation_fraction <= .5:
            raise ValueError('Continuation reserve must be between zero and half the budget')
        if self.offspring_source not in ('native','symmetric_control'):
            raise ValueError('Unknown offspring source')
        if type(self.measured_guidance) is not bool or type(self.economic_allocation) is not bool:
            raise ValueError('Guidance switches must be boolean')
        if type(self.active_constraints) is not bool:raise ValueError('Boolean active_constraints required')
        if type(self.case_pq_focus) is not bool:raise ValueError('Boolean case_pq_focus required')
        for name in ('pq_adaptive_windows','pq_fast_coverage'):
            if type(getattr(self,name)) is not bool:raise ValueError('Boolean '+name+' required')
            if getattr(self,name) and not self.case_pq_focus:
                raise ValueError(name+' requires case_pq_focus')
        if self.case_pq_focus and (self.strategy!='consolidated' or not self.active_constraints):
            raise ValueError('Case P/Q focus requires consolidated active-constraint search')
        if self.strategy=='physical' and self.continuation_generations:
            raise ValueError('Topology branch budgets exclude adaptive continuation')
        if not isinstance(self.engine_parameters,dict):raise ValueError('Engine parameters must be a dictionary')
        if type(self.topology_dispatch_generations)!=int or self.topology_dispatch_generations<1:
            raise ValueError('Positive topology dispatch generations required')
        if not isinstance(self.radii,(tuple,list)) or not self.radii or any(isinstance(r,bool) or not isinstance(r,(int,float)) or not np.isfinite(r) or not 0<r<=1 for r in self.radii) or 1. not in self.radii:
            raise ValueError('Finite local radii and a full-domain radius of 1 required')
        if self.ga_expected_mutations is not None and (isinstance(self.ga_expected_mutations,bool) or not isinstance(self.ga_expected_mutations,(int,float)) or not np.isfinite(self.ga_expected_mutations) or self.ga_expected_mutations<=0):
            raise ValueError('Expected mutation count must be positive')
        if self.coverage_contract not in ('all_cases_and_families','partial_pilot'):
            raise ValueError('Explicit coverage contract required')
        if self.coverage_contract=='all_cases_and_families' and self.generations_per_visit<8:
            raise ValueError('Full exploration requires at least eight native generations per visit')
        for name in ('generations_per_visit', 'cooling_generations'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError('Positive integer required: ' + name)
        if self.cooling_generations < self.generations_per_visit:
            raise ValueError('Cooling horizon must cover at least one visit')
        if isinstance(self.min_unique_fraction, bool) or not 0 < self.min_unique_fraction <= 1:
            raise ValueError('Invalid physical diversity fraction')

    def parameters_for(self,method,dimension):
        parameters=copy.deepcopy(self.engine_parameters)
        if method=='L-SHADE':return parameters
        if self.ga_expected_mutations is not None:
            if method!='GA' or 'mutation_rate' in parameters:
                raise ValueError('Expected mutation count is GA-only and excludes mutation_rate')
            parameters['mutation_rate']=min(1.,self.ga_expected_mutations/max(1,dimension))
        return SearchConfig(method,parameters=parameters).parameters


def make_scheduler(codec,seed,mass,options):
    if options.protected_resource_coverage:
        from resource_coverage import ResourceScheduler
        return ResourceScheduler(codec,seed,mass,options)
    if options.risk_native_acceleration:
        from .risk_native_search import RiskNativeScheduler
        return RiskNativeScheduler(codec,seed,mass,options)
    if options.strategy=='native_construction':
        from .native_construction import ConstructionScheduler
        return ConstructionScheduler(codec,seed,mass,options)
    if options.strategy=='baseline':return ServiceScheduler(codec,seed,mass)
    if options.strategy=='consolidated':
        if options.case_pq_focus:
            if options.pq_adaptive_windows or options.pq_fast_coverage:
                from .adaptive_case_pq import AdaptiveCasePQScheduler
                return AdaptiveCasePQScheduler(codec,seed,mass,options)
            from .case_pq_search import CasePQScheduler
            return CasePQScheduler(codec,seed,mass,options)
        if options.active_constraints:
            from .active_search import ActiveScheduler
            return ActiveScheduler(codec,seed,mass,options)
        from .consolidated_search import ConsolidatedScheduler
        return ConsolidatedScheduler(codec,seed,mass,options)
    if options.strategy=='network_linked':
        from .network_linked_search import NetworkLinkedScheduler
        return NetworkLinkedScheduler(codec,seed,mass,options)
    if options.strategy=='energy_linked':
        from .energy_linked_search import EnergyLinkedScheduler
        return EnergyLinkedScheduler(codec,seed,mass,options)
    if options.strategy=='productive':
        from .productive_search import ProductiveScheduler
        return ProductiveScheduler(codec,seed,mass,options)
    if options.strategy=='physical':
        from .measured_search import PhysicalScheduler
        return PhysicalScheduler(codec,seed,mass,options)
    from .guided_blocks import GuidedScheduler
    return GuidedScheduler(codec,seed,mass,options)


class ServiceScheduler:
    """Three fresh service scopes, two revisits and one complement per cycle.

Case decks and node subsets are independent of cost, cut and AC diagnostics.
Direct node coordinates coexist with grouped coordinates for the remaining
nodes; endpoints in both directions are reachable. Complements keep investment,
topology, storage chronology and the complete decision domain eligible.
    """
    def __init__(self, codec, seed, mass):
        self.base = BalancedScheduler(codec, seed, mass)
        self.codec = codec
        self.rng = np.random.default_rng(np.random.SeedSequence([seed, 77491]))
        self.step = self.service_count = self.complement_count = self.revisit = 0

    def choose(self, reference, contexts):
        slot = self.step % 6
        self.step += 1
        if slot in (1, 4) and contexts:
            # Oldest circular entry: neither low-cut nor low-cost scopes are preferred.
            chosen = contexts[self.revisit % len(contexts)]
            self.revisit += 1
            return chosen, None
        if slot == 3:
            j = self.complement_count
            self.complement_count += 1
            scopes = ('investment_joint', 'topology', 'temporal_cycle', 'full_native',
                      'state_direct', 'state_family', 'coarse_joint')
            scope = scopes[j % len(scopes)]
            if scope == 'full_native':
                block = Block(reference, [Field('all_eligible_coordinates',
                    np.arange(self.codec.dimension), 'direct', False)],
                    dict(scope=scope, scheduler='persistent_seed_only'), self.base.mass)
            else:
                self.base.count = 3
                self.base.complement_visits = self.base.COMPLEMENTS.index(scope)
                block = self.base.next(reference)
                block.metadata['scheduler'] = 'persistent_seed_only'
            return None, block
        st = self.codec.study
        index = self.base.case_order[self.service_count % (st.K * st.S)]
        k, s = divmod(index, st.S)
        sweep = self.service_count // (st.K * st.S)
        self.service_count += 1
        width = (1, 4, 16)[sweep % 3]
        fields = []
        # Interleave isolated and joint faces within the first sweep, then
        # rotate each case through all three. No case must wait a whole first
        # sweep before dispatch can move together with service.
        face=('cut','p_cut','pq_cut')[(sweep+(self.service_count-1) % (st.K*st.S)) % 3]
        for field in self.base.joint_fields([(k, s)]):
            if face=='cut' and field.name!='cut':continue
            if face=='p_cut' and field.name.split(':')[0] not in ('cut','existing_p','new_p'):continue
            ids = field.indices.copy()
            if not len(ids):
                continue
            # Small independent subsets plus a coarse remainder: no forced common
            # movement of every load bus and no fixed technology/site selection.
            count = min(width if field.name == 'cut' else max(1, width // 2), len(ids))
            selected = np.sort(self.rng.choice(ids, count, replace=False))
            remainder = np.setdiff1d(ids, selected)
            fields.append(Field(field.name + ':individual', selected, 'direct'))
            if len(remainder):
                fields.append(Field(field.name + ':group', remainder))
        block = Block(reference, fields, dict(scope='service_joint',
            scheduler='persistent_seed_only', balanced_case=True,
            case_groups=[[(k, s)]], node_width=width, coverage_cycle=sweep, service_face=face), self.base.mass)
        return None, block


def exploration_preflight(codec, population, options, seed=0, mass=.25):
    """Conservative bound for case/family coverage, not all node combinations.

    No candidate is decoded or evaluated. Refresh may require 2N+2 memories
    and N diversification evaluations; reserve these as well as offspring.
    """
    if not hasattr(codec,'study'):return dict(applicable=False)
    if options.protected_resource_coverage:
        from dataclasses import replace
        from meta_planning.initial_resource_search import resource_tasks
        tasks=resource_tasks(codec)
        original=exploration_preflight(codec,population,replace(options,protected_resource_coverage=False),seed,mass)
        initial=len(tasks)*population*(1+min(2,options.generations_per_visit))
        return dict(applicable=True,conservative_calls=initial+8*original['conservative_calls'],
            initial_family_opportunity_calls=initial,resource_tasks=tasks,
            recurring_outer_resource_fraction=.5,original=original,
            scope='Opportunity bound only; native physical coverage is measured, not assumed; no exhaustive combination guarantee',
            auxiliary_evaluations=0)
    if options.risk_native_acceleration:
        from .risk_native_search import coverage_bound
        return coverage_bound(codec,population,options)
    if options.strategy=='native_construction':
        if options.coverage_contract!='partial_pilot':raise ValueError('Construction cannot promise complete coverage before first feasibility')
        return dict(applicable=False,scope='Stop at first feasible point; alternating constraint/energy and broad native scopes; no coverage or feasibility deadline guaranteed')
    scheduler=make_scheduler(codec,seed,mass,options)
    if options.strategy in ('diagnostic','physical','productive','energy_linked','network_linked','consolidated'):scheduler.coverage_preflight=True
    reference=np.full(codec.dimension,.5);contexts=[];cases=set();families=set();scopes=set()
    required={d['name'] for d in scheduler.base.categories}|{'new_capacity'}
    visits=new=revisits=topology_visits=0
    while visits<10000:
        old,block=scheduler.choose(reference,contexts);visits+=1
        if old is not None:revisits+=1
        else:
            new+=1
            from types import SimpleNamespace
            contexts.append(SimpleNamespace(number=len(contexts)));meta=block.metadata;scopes.add(meta['scope'])
            topology_visits+=int(meta['scope']=='topology')
            if meta['scope']=='service_joint' and (options.strategy not in ('diagnostic','physical','productive','energy_linked','network_linked','consolidated') or meta.get('allocation')=='uniform_coverage'):
                cases.add(tuple(meta['case_groups'][0][0]))
            if meta['scope']=='investment_joint':families.add(meta['family'])
            if options.strategy=='consolidated':
                # The simulator has already charged every alternating visit.
                # Count protected single-state sweeps too, not only the oldest
                # nested service scheduler; joint asset groups do not substitute
                # for this independent state coverage contract.
                if meta['scope']=='interbus_dispatch_service' or (meta['scope']=='coordinated_native'
                        and not meta.get('asset') and meta.get('allocation')=='protected_seed_deck'):
                    cases.add(tuple(meta['case_groups'][0][0]))
                family={'native_cable':'line_schedule','native_storage_cycle':'storage_purchase'}.get((meta.get('asset') or {}).get('scope'))
                if family:families.add(family)
        if len(cases)==codec.study.K*codec.study.S and families>=required and len(scopes)>=8:break
    else:raise ValueError('Unable to plan complete case/family coverage')
    calls=new*population*(1+options.generations_per_visit)
    calls+=revisits*(3*population+2+population*options.generations_per_visit)
    if options.strategy=='physical':calls+=topology_visits*2*population*population*(1+options.topology_dispatch_generations)
    if options.strategy=='network_linked':calls*=4
    if options.strategy=='energy_linked':calls*=2 # One charged linked visit per complementary visit.
    calls += visits*population*options.continuation_generations
    if options.active_constraints:calls*=3  # two focused visits between protected visits
    if options.case_pq_focus:calls*=2  # conservative bound: two original visits per three
    return dict(applicable=True,conservative_calls=int(calls),visits=visits,new_contexts=new,
        revisits=revisits,cases=len(cases),investment_families=sorted(families),
        scope='one case sweep and at least one site per investment family; not exhaustive sites or combinations',
        cooling='each visit spans its actual native batches; reheated on next context visit, native memories retained')


def position_key(x):
    return hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()


def diverse_start(block, codec, sources, n, rng):
    """Distinct *decoded* controls; unscored symmetric stratified initialization.

Random points are not called native offspring, and are not selected by fitness.
Sampling/rejection here checks representation identity only, never AC feasibility.
    """
    if block.metadata.get('resource_coverage'):
        from resource_coverage import resource_initialization
        return resource_initialization(block,codec,sources,n,rng)
    rows, origins, identities = [], [], set()
    if block.metadata.get('scheduler')=='physical' and block.metadata.get('scope')=='topology':
        sources=sources[:1] # Preserve anchor; reserve initial members for distinct tree proposals.
    def add(x, origin):
        identity = physical_key(codec.decode(block.decode(x))[0])
        if identity in identities:
            return False
        identities.add(identity); rows.append(np.asarray(x).copy()); origins.append(origin)
        return True
    for source in sources:
        add(block.project(source), 'context_retest')
        if len(rows) == n:
            break
    if not rows:
        raise ValueError('Missing incumbent in local initialization')
    anchor = rows[0].copy()
    categories=list(getattr(block,'categorical_indices',[]))
    for attempt in range(64 * n):
        if len(rows) == n:
            break
        radius = (.02, .08, .25, 1.)[(attempt // 2) % 4]
        if attempt % 2 == 0:
            direction = rng.uniform(-1., 1., block.dimension)
        z = np.clip(anchor + (1 if attempt % 2 == 0 else -1) * radius * direction, 0, 1)
        if categories:
            # Unscored diversity initialization, explicitly not native gain.
            # Small continuous windows must not confine nominal categories to
            # whichever technology/edge happens to sit at the midpoint.
            z[categories]=rng.random(len(categories))
        if block.metadata.get('scheduler')=='physical' and block.metadata.get('scope')=='topology':
            # Distinct operation alone must not exhaust initialization before
            # any discrete tree alternative is sampled. Uniform, unscored keys.
            z[0]=(1+rng.integers(max(1,len(block.edges)))+rng.random())/(len(block.edges)+1) if block.edges else 0.
            z[1]=rng.random()
        add(z, 'diversity_initialization')
    unique = len(rows)
    while len(rows) < n:
        rows.append(rows[(len(rows) - unique) % unique].copy())
        origins.append('context_retest')
    return np.asarray(rows), origins, dict(physical_unique=unique,
        repeated_fallback=n-unique, generated_initial_members=origins.count('diversity_initialization'))


class Context:
    def __init__(self, number, block, engine):
        self.number, self.block, self.engine = number, block, engine
        self.controlled = np.concatenate([f.indices for f in block.fields])
        self.visits = 0

    def refresh_points(self, incumbent):
        e = self.engine
        points = list(e.population)
        if e.config.method in ('PSO', 'QPSO'):
            points += list(e.personal)
        if e.best is not None:
            points.append(e.best['x'])
        points.append(self.block.project(incumbent))
        return {position_key(x): np.asarray(x).copy() for x in points}

    def refresh(self, incumbent, evaluate):
        """Retain coordinates and learned parameters; re-evaluate ALL fitness.

The mapping of controlled coordinates stays fixed. Only the outside context
changes. Old fitness is never reused across contexts, including personal/global
memories. SHADE's archive stores positions, not fitness, and remains well-defined.
        """
        e = self.engine
        if e.cursor or e.generation == 0:
            raise ValueError('Context refresh requires a complete native batch')
        points = self.refresh_points(incumbent)
        saved = self.block.reference[self.controlled].copy()
        self.block.reference = np.asarray(incumbent).copy()
        self.block.reference[self.controlled] = saved
        records = {h: evaluate(x, 'context_retest') for h, x in points.items()}
        e.records = [copy.deepcopy(records[position_key(x)]) for x in e.population]
        if e.config.method in ('PSO', 'QPSO'):
            e.personal_records = [copy.deepcopy(records[position_key(x)]) for x in e.personal]
            for i, record in enumerate(e.records):
                if key(record) < key(e.personal_records[i]):
                    e.personal[i] = e.population[i].copy()
                    e.personal_records[i] = copy.deepcopy(record)
        h = min(records, key=lambda h: rank(records[h]))
        e.best = dict(x=points[h].copy(), record=copy.deepcopy(records[h]))
        return len(records)


def memory_summary(context):
    e = context.engine
    out = dict(context=context.number, visits=context.visits,
               generation=e.generation, budget_progress=e.budget_progress,
               velocity_norm=float(np.linalg.norm(e.velocity)) if e.velocity is not None else None,
               population=len(e.population),local_evaluations=e.evaluations)
    if e.config.method=='L-SHADE':
        out.update(lshade_updates=e.updates,mf=e.mf.tolist(),mcr=e.mcr.tolist(),memory_index=e.mi,
                   archive=len(e.archive),global_lpsr_progress=e.global_progress)
    if e.config.method == 'SHADE':
        out.update(shade_updates=e.shade['updates'], shade_memory_index=e.shade['index'])
    return out


def run_persistent(state, method, calls, mass, options=None):
    from .prepared_search import write
    from .execution import resolve_options
    from .search_deadline import SearchDeadlineReached
    from .search_telemetry import EffectiveCoverage, decoded_change
    check_budget=getattr(state,"check_budget",lambda:None)
    options = PersistentOptions(**resolve_options(options))
    options.parameters_for(method,state.codec.dimension)
    unbounded=calls is None
    if not unbounded and (type(calls) is not int or calls < state.n):
        raise ValueError('At least one local population is required')
    preflight=exploration_preflight(state.codec,state.n,options,state.seed,mass)
    if not unbounded and options.coverage_contract=='all_cases_and_families' and preflight.get('applicable') and calls<preflight['conservative_calls']:
        raise ValueError(f"Insufficient exploration budget: {calls} < {preflight['conservative_calls']}; explicitly label a smaller run partial_pilot")
    end = float('inf') if unbounded else state.evaluations + calls
    saved = getattr(state, 'persistent_resume', None)
    signature = dict(method=method, mass=mass, options=vars(options))
    if saved is not None:
        if not unbounded or saved['signature'] != signature:
            raise ValueError('Checkpoint requires the identical unbounded native protocol')
        scheduler, rng, contexts, telemetry = (saved[k] for k in ('scheduler','rng','contexts','telemetry'))
        extra_used, ineffective, identities_by_origin = (saved[k] for k in ('extra_used','ineffective','identities_by_origin'))
        telemetry_enabled=hasattr(state.codec,'study')
    else:
        scheduler = make_scheduler(state.codec, state.seed, mass, options)
        scheduler.population=state.n
        rng = np.random.default_rng(np.random.SeedSequence([state.seed, 33919]))
        contexts = state.contexts_factory(scheduler) if hasattr(state,'contexts_factory') else []
        state.persistent_diagnostics = dict(version=2, options=vars(options), preflight=preflight, deferred_blocks=[], visits=[],
            selection=getattr(state,'selection_description','strict feasibility then unchanged economic risk objective'),
            cut_focus=('recurring service blocks; seed-only coverage' if options.strategy=='baseline' else
                       'bounded native joint moves, economic state allocation with protected uniform coverage'),
            attribution='all unscored initialization and context projections separate from native offspring')
        from .search_telemetry import EffectiveCoverage, decoded_change
        telemetry=EffectiveCoverage(getattr(state.codec,'study',None)); extra_used=0
        telemetry_enabled=hasattr(state.codec,'study')
        ineffective=0
        identities_by_origin={name:set() for name in ('context_retest','diversity_initialization','native_offspring')}
    def evaluate(context, z, origin):
        check_budget()
        event = dict(origin=origin, method=method, context=context.number,
            visit=context.visits, generation=context.engine.generation,
            block=context.block.describe())
        full=context.block.decode(z)
        candidate=state.codec.decode(full)[0]
        identity=physical_key(candidate)
        change=decoded_change(context.reference_candidate,candidate) if telemetry_enabled else None
        if change is not None:event['decoded_change_vs_visit']=change
        identities_by_origin[origin].add(identity)
        event['decoded_physical_identity']=identity
        previous=getattr(state,'best',None)
        ac_before=state.ev.case_misses if hasattr(state,'ev') else 0
        record=state.evaluate(full, event)
        if hasattr(scheduler,'observe_charge'):
            scheduler.observe_charge(context,origin,(state.ev.case_misses-ac_before) if hasattr(state,'ev') else 0)
        if change is not None:telemetry.observe(event['block'],origin,identity,change,record,previous)
        if hasattr(scheduler,'observe'):scheduler.observe(context,record,previous,origin)
        if hasattr(scheduler,'observe_candidate'):scheduler.observe_candidate(context,full,record,origin)
        return record
    context=None
    start_eval=state.evaluations
    try:
        while state.evaluations < end:
            check_budget()
            remaining = end - state.evaluations
            if hasattr(scheduler,'update'):scheduler.update(state.best)
            if hasattr(scheduler,'set_reference'):scheduler.set_reference(state.best_x,remaining)
            context, block = scheduler.choose(state.best_x, contexts)
            if context is not None and len(context.refresh_points(state.best_x)) + state.n > remaining:
                context = contexts[-1]
                # Final incomplete visit retains its own already-evaluated context.
                refresh = False
            else:
                refresh = context is not None
            start_eval = state.evaluations
            if context is None:
                if remaining < state.n:
                    context = contexts[-1]; refresh = False
                else:
                    sources = [state.best_x] + [b['x'] for b in state.bank] + list(state.population)
                    if block.metadata['scope']=='topology_dispatch':sources.insert(0,block.reference)
                    initial, origins, diversity = diverse_start(block, state.codec, sources, state.n, rng)
                    if diversity['physical_unique']==1:
                        ineffective+=1
                        state.persistent_diagnostics['deferred_blocks'].append(dict(block=block.describe(),reason='No decoded diversity after finite symmetric identity probes; no AC fitness used',identity_probes_limit=64*state.n))
                        if ineffective>=64:raise ValueError('64 successive blocks without decoded changes; inspect representation')
                        continue
                    ineffective=0
                    seed = int(np.random.SeedSequence([state.seed, len(contexts), 82913]).generate_state(1)[0])
                    if method=='L-SHADE':
                        from lshade_persistent import PersistentLSHADE
                        engine=PersistentLSHADE(initial,seed)
                    else:
                        cfg = SearchConfig(method, population=state.n, generations=(2**63-1 if unbounded else calls+1), seed=seed,
                            progress_schedule='budget', comparison_profile='declared_variant',
                            parameters=options.parameters_for(method,block.dimension))
                        engine=Engine(cfg, initial)
                    context = Context(len(contexts), block, engine)
                    contexts.append(context)
                    context=contexts[-1]
                    if telemetry_enabled:context.reference_candidate=state.codec.decode(block.reference)[0]
                    telemetry.register_block(block.describe())
                    for f in block.fields:
                        state.coverage.update(int(i) for i in f.indices)
                    context.initial_diversity = diversity
                    for z, origin in zip(initial, origins):
                        if method=='L-SHADE':
                            np.testing.assert_array_equal(context.engine.ask(),z)
                        context.engine.tell(z, evaluate(context, z, origin))
                    context.initial_diversity = diversity
            before = memory_summary(context)
            # The outside context defines a new local objective on revisit. Reheat
            # its finite cooling schedule, retaining population, velocity and memory.
            if refresh:context.engine.budget_progress=0.
            if refresh:
                # Freeze outside coordinates for the entire visit. Evaluations may
                # improve the global incumbent, but cannot change this visit's context.
                incumbent = state.best_x.copy()
                if telemetry_enabled:
                    reference=incumbent.copy(); reference[context.controlled]=context.block.reference[context.controlled]
                    context.reference_candidate=state.codec.decode(reference)[0]
                context.refresh(incumbent, lambda z, origin: evaluate(context, z, origin))
                e = context.engine
                identities = [physical_key(state.codec.decode(context.block.decode(x))[0]) for x in e.population]
                local_n=len(e.population)
                if len(set(identities)) < int(np.ceil(local_n * options.min_unique_fraction)):
                    sources = [context.block.decode(e.best['x'])] + [context.block.decode(x) for x in e.population]
                    initial, origins, diversity = diverse_start(context.block, state.codec, sources, local_n, rng)
                    # Every replacement has its own charged evaluation. Partial
                    # replenishment is permitted when approaching the total budget.
                    for i, (x, origin) in enumerate(zip(initial, origins)):
                        if state.evaluations >= end or position_key(x) == position_key(e.population[i]):
                            continue
                        record = evaluate(context, x, origin)
                        e.population[i] = x; e.records[i] = copy.deepcopy(record)
                        if hasattr(e,'personal'):
                            e.personal[i] = x; e.personal_records[i] = copy.deepcopy(record)
                        if e.velocity is not None:e.velocity[i] = 0.
                        if rank(record) < rank(e.best['record']):
                            e.best = dict(x=x.copy(), record=copy.deepcopy(record))
                e.pending = e._propose()
            e = context.engine
            context.visits += 1
            generations=options.generations_per_visit
            if hasattr(scheduler,'visit_generations'):
                generations=scheduler.visit_generations(context,generations)
            if options.strategy=='physical' and context.block.metadata['scope']=='topology':generations=1
            if context.block.metadata['scope']=='topology_dispatch':generations=options.topology_dispatch_generations
            offspring = min(state.n * generations, end-state.evaluations)
            target_generation=e.generation+generations
            batches=int(np.ceil((e.cursor+offspring)/state.n))
            used_progress=[]; schedule_count=0
            before_native=getattr(state,'best',None)
            continuation=0
            i=0
            while (e.generation < target_generation if method=='L-SHADE' else i < offspring) and state.evaluations < end:
                if e.cursor==0:
                    used_progress.append(e.budget_progress);schedule_count+=1
                if options.offspring_source=='native':
                    z=e.ask();origin='native_offspring'
                else:
                    # Unscored symmetric proposals in the same representation, charged
                    # as diversity. No hidden engine output or AC-guided selection.
                    if i%2==0:
                        control_direction=rng.uniform(-1.,1.,context.block.dimension)
                        control_anchor=e.best['x'].copy()
                    radius=options.radii[(i//2)%len(options.radii)]
                    z=np.clip(control_anchor+(1 if i%2==0 else -1)*radius*control_direction,0,1)
                    origin='diversity_initialization'
                result=evaluate(context,z,origin)
                if method=='L-SHADE':e.global_progress=state.evaluations/end
                # tell() generates the NEXT batch: set the schedule before that call,
                # not after ask(), when the pending values have already been created.
                extend_now=False
                # A single bounded extension rewards a native improvement only.
                # Keep room for a complete next visit and cap total extra effort.
                if i+1==offspring and e.cursor==state.n-1 and options.offspring_source=='native' and not continuation and options.continuation_generations:
                    after=getattr(state,'best',None)
                    reserve=state.n*(1+generations)
                    extra=state.n*options.continuation_generations
                    improved=before_native is not None and after is not None and rank(after)<rank(before_native)
                    if improved and end-state.evaluations >= reserve+extra and (unbounded or extra_used+extra <= int(calls*options.continuation_fraction)):
                        continuation=extra;extra_used+=extra;offspring+=extra
                        extend_now=True
                if e.cursor==len(e.pending)-1:
                    if extend_now:
                        # Decide before tell creates the next batch: do not generate
                        # and discard a batch or update PSO velocity twice.
                        e.set_budget_progress(0.);batches=options.continuation_generations;schedule_count=0
                    else:e.set_budget_progress(min(1.,schedule_count/max(1,batches-1)))
                e.tell(z,result);i+=1
            offspring=i
            visit = dict(context=context.number, revisited=refresh, start_evaluation=start_eval,
                end_evaluation=state.evaluations, offspring=offspring, continuation_calls=continuation, cooling_progress_used=used_progress,
                block=context.block.describe(), initial_diversity=context.initial_diversity,
                effective_engine_parameters=copy.deepcopy(e.config.parameters),
                memory_before=before, memory_after=memory_summary(context))
            state.sessions.append(visit)
            if hasattr(scheduler,'finish_visit'):scheduler.finish_visit(context)
            state.persistent_diagnostics['visits'].append(visit)
            state.persistent_diagnostics['effective_coverage']=telemetry.report()
            state.persistent_diagnostics['continuation_calls']=extra_used
            state.save_best()
            write(state.dst/'sessions.json', state.sessions)
            write(state.dst/'persistent_diagnostics.json', state.persistent_diagnostics)
            # Only complete native visits can yield to another independent run.
            # Store every scheduler/RNG/telemetry object; do not rebuild memories.
            if hasattr(state, 'visit_checkpoint'):
                if not unbounded: raise ValueError('Cooperative checkpoint requires unbounded search')
                state.persistent_resume=dict(signature=signature,scheduler=scheduler,rng=rng,
                    contexts=contexts,telemetry=telemetry,extra_used=extra_used,
                    ineffective=ineffective,identities_by_origin=identities_by_origin)
                if state.visit_checkpoint(): return
    except SearchDeadlineReached:
        state.persistent_diagnostics['termination_reason']=getattr(state,'stop_reason',None) or 'wall_time_limit'
        if context is not None and (not state.sessions or state.evaluations>state.sessions[-1]['end_evaluation']):
            state.persistent_diagnostics['interrupted_visit']=dict(context=context.number,
                block=context.block.describe(),start_evaluation=start_eval,end_evaluation=state.evaluations,
                evaluated_calls=state.evaluations-start_eval,
                note='Last visit may have an incomplete population or refresh. It is never resumed; only fully evaluated global incumbent is retained.')
    else:
        state.persistent_diagnostics['termination_reason']='call_budget'
    state.persistent_diagnostics['effective_coverage']=telemetry.report()
    state.persistent_diagnostics['continuation_calls']=extra_used
    state.save_best()
    write(state.dst/'sessions.json',state.sessions)
    state.persistent_diagnostics.update(
        unique_decoded_by_origin={name:len(values) for name,values in identities_by_origin.items()},
        coverage_note='Coordinate eligibility is not effective exploration; service cases and decoded identities reported separately',
        contexts=len(contexts), revisits=sum(v['revisited'] for v in state.sessions),
        origin_counts=dict(Counter({k:state.counters[k] for k in
            ('context_retest','diversity_initialization','native_offspring')})),
        memory=[memory_summary(c) for c in contexts],
        eligible_coordinates_visited=len(state.coverage),
        service_cases_visited=len({tuple(case) for v in state.sessions
            if v['block']['scope'] in ('service_joint','interbus_dispatch_service','coordinated_native')
            for group in v['block']['case_groups'] for case in group}))
    if hasattr(scheduler,'report'):state.persistent_diagnostics['guidance']=scheduler.report()
    if options.strategy=='physical' and scheduler.pending and state.persistent_diagnostics['termination_reason']!='wall_time_limit':raise RuntimeError('Unfinished topology dispatch branches: budget contract broken')
    write(state.dst/'persistent_diagnostics.json', state.persistent_diagnostics)
