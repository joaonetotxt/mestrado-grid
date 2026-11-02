"""Canonical, explicit native entry point. Planning performs no AC evaluations."""
import copy
import time
import hashlib
from pathlib import Path
import numpy as np
from .schema import read


def resolve_options(options=None):
    options=copy.deepcopy(options or {})
    if 'strategy' not in options:
        profile=read(Path(__file__).with_name('perfil_consolidado.json'))
        profile.update(options)
        return profile
    return options


def plan(study, config):
    from .persistent_search import PersistentOptions, exploration_preflight
    from .search_arrays import ArrayCodec
    from .engines import SearchConfig
    if 'physical' not in study.electrical:
        raise ValueError('Canonical native search requires an explicit physical study: supply --study or --physical-config; the physical extension is never enabled silently')
    allowed={'method','seed','population','additional_calls','endpoint_mass','persistent_options','initialization','initial_population','wall_seconds'}
    if set(config)-allowed:raise ValueError('Unknown native configuration fields: '+str(sorted(set(config)-allowed)))
    cfg=copy.deepcopy(config)
    for field in ('seed','population','additional_calls'):
        if type(cfg.get(field)) is not int:raise ValueError('Explicit integer required: '+field)
    seconds=cfg.get('wall_seconds')
    if seconds is not None and (isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not np.isfinite(seconds) or seconds<=0):
        raise ValueError('wall_seconds must be finite and positive')
    n=cfg['population']
    if n<4 or n%2 or cfg['additional_calls']<n:raise ValueError('Even population >=4 and at least one additional population required')
    if bool(cfg.get('initialization')) == bool(cfg.get('initial_population')):
        raise ValueError('Declare exactly one initial population file or fresh initialization mode')
    cfg['persistent_options']=resolve_options(cfg.get('persistent_options'))
    o=PersistentOptions(**cfg['persistent_options'])
    SearchConfig(cfg['method'],seed=cfg['seed'],population=n,initialization=cfg.get('initialization','resource'))
    o.parameters_for(cfg['method'],1)
    mass=cfg.setdefault('endpoint_mass',.25)
    if isinstance(mass,bool) or not isinstance(mass,(float,int)) or not np.isfinite(mass) or not 0<mass<.5:raise ValueError('Invalid endpoint mass')
    if seconds is not None and o.coverage_contract!='partial_pilot':
        raise ValueError('Time-limited search must declare partial_pilot: time cannot guarantee full coverage')
    codec=ArrayCodec(study)
    coverage=exploration_preflight(codec,n,o,cfg['seed'],mass)
    if o.coverage_contract=='all_cases_and_families' and cfg['additional_calls']<coverage['conservative_calls']:
        raise ValueError(f"Insufficient exploration budget: {cfg['additional_calls']} < {coverage['conservative_calls']}; partial_pilot must be explicit")
    initial=None
    if cfg.get('initial_population'):
        path=Path(cfg['initial_population']).resolve()
        a=np.load(path,allow_pickle=False)
        if a.shape!=(n,codec.dimension) or not np.isfinite(a).all() or np.any((a<0)|(a>1)):raise ValueError('Invalid initial population array')
        cfg['initial_population']=str(path)
        initial=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),origin='inherited_explicit')
    return dict(config=cfg,study_id=study.input_id,financial_policy=study.c.get('financial_policy'),
                expected_cases=study.K*study.S,initial_population=initial,preflight=coverage,
                total_calls=n+cfg['additional_calls'],
                budget_kind='wall_time_with_call_safety_cap' if seconds is not None else 'exact_calls',
                time_accounting='Initialization and native search included; read-only planning, construction of state, and independent final audit excluded. Atomic evaluation may overrun; no new evaluation starts after deadline.',entrypoint='canonical_native_v1',
                limitations='Finite case/family coverage is not exhaustive sites/combinations; rank and physics unchanged')


def run(study, config, output):
    from .prepared_search import SearchState,write,source_hashes
    protocol=plan(study,config)  # All gates precede output, initialization and AC calls.
    cfg=protocol['config'];dst=Path(output)
    if dst.exists():raise FileExistsError('Choose an unused output directory')
    population=None
    if protocol['initial_population']:
        path=Path(cfg['initial_population'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=protocol['initial_population']['sha256']:raise ValueError('Initial population changed after planning')
        population=np.load(path,allow_pickle=False)
    before=source_hashes()
    state=SearchState(study,cfg['seed'],cfg['population'],dst,
        cfg.get('initialization','inherited_explicit'),initial_population=population)
    write(dst/'execution_protocol.json',dict(protocol,source_hashes=before))
    write(dst/'study.json',dict(data=study.data,electrical=study.electrical))
    try:
        from .search_deadline import SearchDeadline
        state.start=time.monotonic()
        timer=SearchDeadline(cfg['wall_seconds'],start=state.start) if cfg.get('wall_seconds') is not None else None
        state.initialize();start=copy.deepcopy(state.best)
        # Complete the common population atomically; its time is charged.
        if timer is not None:state.check_budget=timer.check
        state.native(cfg['method'],cfg['additional_calls'],cfg['endpoint_mass'],options=cfg['persistent_options'])
        timing=None
        if timer is not None:
            timing=timer.report()
            timing.update(termination_reason=state.persistent_diagnostics.get('termination_reason','call_safety_cap'),
                actual_proposals=state.evaluations,call_safety_cap=protocol['total_calls'])
            write(dst/'timing.json',timing)
        state.status('running','independent_final_validation')
        result=state.finish(dict(cfg,budget=state.evaluations if timer is not None else protocol['total_calls'],
            budget_kind=protocol['budget_kind'],timing=timing),start)
        if source_hashes()!=before:raise RuntimeError('Execution source hashes changed')
        write(dst/'execution_integrity.json',dict(verified=True,source_hashes=before,
            initial_population=protocol['initial_population'],proposals=state.evaluations))
        return result
    except Exception:
        import traceback
        write(dst/'error.json',dict(error=traceback.format_exc()))
        state.status('failed','execution_or_audit');raise
