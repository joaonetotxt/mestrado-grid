"""Reproducible direct searches with immutable evaluations and checked resumes.

Each method owns a fresh cache. No historical controls enter this path. AC is
evaluated during search and rechecked at publication; there is no redispatch.
"""
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
import copy
import hashlib
import json
import os
import time
import numpy as np
from .schema import Study,Candidate,plain,read,write_new,fingerprint
from .codec import Codec
from .initialization import initialize
from .evaluator import Evaluator,Policy,save_evaluation,audit_saved,rank,source_hash
from .engines import SearchConfig,Engine
from .monitoring import peak_memory_bytes
from .persistence import write_bundle,read_bundle,load_evaluation
from .financial_policy import compact_metrics, resolve


def discrete_id(candidate):
    value=candidate.to_dict()['plan']
    if candidate.physical is not None:
        value=dict(plan=value,physical_installations={k:v for k,v in candidate.physical.items() if k.endswith('_choice')})
    return fingerprint(value)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def seal(path,payload,replace=False):
    """Single-file checksum envelope, committed atomically on the same volume."""
    path=Path(path)
    if path.suffix=='.npz':return write_bundle(path,payload,replace)
    if path.exists() and not replace:raise FileExistsError(path)
    envelope=dict(sha256=fingerprint(payload),payload=payload)
    tmp=path.with_name(path.name+'.pending')
    with tmp.open('w',encoding='utf8') as f:
        json.dump(plain(envelope),f,ensure_ascii=False,allow_nan=False,separators=(',',':'))
        f.flush();os.fsync(f.fileno())
    for attempt in range(6):
        try:
            os.replace(tmp,path);break
        except PermissionError as exc:
            # Windows indexing/synchronization can briefly hold the destination.
            # Retain atomic replace; never delete the committed file to bypass it.
            if os.name!='nt' or exc.winerror not in (5,32) or attempt==5:raise
            time.sleep(.02*2**attempt)


def unseal(path):
    if Path(path).suffix=='.npz':return read_bundle(path)
    value=read(path)
    if set(value)!={'sha256','payload'} or fingerprint(value['payload'])!=value['sha256']:
        raise ValueError('Corrupt checkpoint/result envelope: '+str(path))
    return value['payload']


@contextmanager
def locked(directory):
    """OS lock is released even if the interpreter crashes; no stale PID tests."""
    directory.mkdir(parents=True,exist_ok=True)
    with (directory/'run.lock').open('a+b') as f:
        f.seek(0,os.SEEK_END)
        if not f.tell():f.write(b'0');f.flush()
        f.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(f.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as exc:raise RuntimeError('Another process is using this search directory') from exc
        try:yield
        finally:
            f.seek(0)
            if os.name=='nt':msvcrt.locking(f.fileno(),msvcrt.LK_UNLCK,1)
            else:fcntl.flock(f,fcntl.LOCK_UN)


def verify_record(root,record,signature):
    """Existence, every file hash, physical identity and exact compact fitness."""
    folder=(root/record['directory']).resolve()
    if not folder.is_relative_to(root.resolve()):raise ValueError('Record escaped the search directory')
    if sha(folder/'manifest.json')!=record['manifest_sha256']:raise ValueError('Evaluation manifest changed')
    _,vector,saved=load_evaluation(folder);candidate=Candidate.from_dict(vector)
    if candidate.candidate_id!=record['candidate_id'] or saved['candidate_id']!=candidate.candidate_id or saved['signature']!=signature:
        raise ValueError('Saved candidate/result signature mismatch')
    if saved['status']!=record['status'] or rank(saved)!=rank(record):raise ValueError('Saved fitness mismatch')
    expected=dict(plan_id=saved['plan_id'],discrete_plan_id=discrete_id(candidate),
        **{k:saved[k] for k in ('expected_cases','approved_cases','failed_cases','normalized_violations','normalized_violation')},
        newton_iterations=sum(c.get('newton_iterations',0) for c in saved['cases']),
        sweep_iterations=sum(c.get('sweep_iterations',0) for c in saved['cases']),
        metrics=compact_metrics(saved['metrics']))
    if any(record.get(k)!=v for k,v in expected.items()):raise ValueError('Saved compact diagnostic mismatch')
    if sha(folder/position_file(record))!=record['position_sha256']:raise ValueError('Evaluated position changed')
    return saved


def position_file(record):
    name=record.get('position_file','position.json')
    if name not in ('position.json','position.npz'):raise ValueError('Invalid position filename')
    return name


def read_position(root,record):
    path=root/record['directory']/position_file(record)
    return read_bundle(path) if path.suffix=='.npz' else read(path)


def common_population(codec,config):
    seeds=np.random.SeedSequence([config.seed,17031]).generate_state(config.population)
    # Existing catalogue constructor only; no historical controls or new search.
    # Retain random members to expose all eligible decisions to every engine.
    anchor='physical' if 'physical' in codec.study.electrical else 'constructive'
    modes=[(anchor if i==0 else 'random') if config.initialization=='mixed' else config.initialization for i in range(config.population)]
    rows=[initialize(codec,int(seed),mode)[1] for seed,mode in zip(seeds,modes)]
    return np.asarray(rows),[int(s) for s in seeds]


def _checkpoint(root,engine,cache,positions,history,signature,initial_id,elapsed):
    # Two alternating slots: the current committed state is never overwritten
    # before the pointer commits its successor. An interrupted write is an orphan.
    slot=1-unseal(root/'checkpoint.json')['slot'] if (root/'checkpoint.json').exists() else 0
    suffix='npz' if engine.config.persistence=='compact' else 'json'
    name=f'checkpoint_state_{slot}.{suffix}'
    seal(root/name,dict(version=1,signature=signature,initial_id=initial_id,
        engine=engine.to_dict(native=engine.config.persistence=='compact'),cache=cache,positions=positions,history=history),replace=True)
    digest=sha(root/name)
    seal(root/'checkpoint.json',dict(slot=slot,state_sha256=digest,elapsed_search_seconds=elapsed()),replace=True)


def _audit_best(root,best,signature,codec):
    if best is None:return None
    record=best['record'];verify_record(root,record,signature)
    position=np.asarray(read_position(root,record),float)
    if not np.array_equal(position,best['x']):raise ValueError('Best position differs from saved evaluated position')
    physical,_=codec.decode(position)
    if physical.candidate_id!=record['candidate_id']:raise ValueError('Best position does not decode to the saved candidate')
    return audit_saved(root/record['directory'])


def _verify_engine_positions(root,engine):
    """Every evaluated population/pbest/partial-batch row retains its fitness."""
    pairs=[(engine.population,engine.records,'population'),
           (engine.personal,engine.personal_records,'personal'),
           (engine.pending,engine.received,'pending')]
    for coordinates,records,label in pairs:
        for i,record in enumerate(records):
            if record is not None and not np.array_equal(coordinates[i],read_position(root,record)):
                raise ValueError('Checkpoint '+label+' position differs from evaluated record')


def _verify_history(root,history,cache,config,evaluations,categorical=None,exploration=None,dependencies=None):
    """Check experimental statistics against the immutable evaluated records.

    File checksums alone do not establish agreement between a serialized history
    and its candidate records. Repeated proposals remain calls, not unique ACs.
    """
    if len(history)!=evaluations:raise ValueError('Checkpoint history call count mismatch')
    positions={cid:fingerprint(read_position(root,r)) for cid,r in cache.items()}
    from .dependency_variation import parent_trace
    parents=parent_trace(history,cache,config) if dependencies is not None else None
    seen=set();best=None;previous_elapsed=0.
    for i,h in enumerate(history):
        cid=h.get('candidate_id');r=cache.get(cid)
        if r is None:raise ValueError('Checkpoint history references an unknown candidate')
        generation,index=divmod(i,config.population)
        objective=None if r['metrics'] is None else r['metrics']['objective_VP']
        expected=dict(call=i+1,generation=generation,index=index,evaluated_genome_id=positions[cid],
                      cache_hit=cid in seen,status=r['status'],objective_VP=objective,
                      selection_objective=None if r['metrics'] is None else r['metrics']['selection_objective'],
                      financial_policy=None if r['metrics'] is None else r['metrics']['financial_policy'],
                      violation=r['normalized_violation'],plan_id=r['plan_id'],discrete_plan_id=r['discrete_plan_id'],
                      failed_cases=r['failed_cases'],approved_cases=r['approved_cases'],normalized_violations=r['normalized_violations'])
        previous=best
        if best is None or rank(r)<rank(best):best=r
        expected.update(best_id=best['candidate_id'],best_status=best['status'])
        expected.update(reference_best_id=None if previous is None else previous['candidate_id'],
                        incumbent_improved=previous is None or previous['candidate_id']!=best['candidate_id'])
        if categorical is not None:
            from .categorical import transitions,validate_events
            event=h.get('categorical_event');validate_events([event],categorical,1)
            if (generation==0 or config.categorical_policy=='none' or config.method=='GA' and index==0) and event is not None:
                raise ValueError('Unexpected categorical event')
            if exploration is not None:
                from .exploration import validate_mixed_events
                ce=h.get('continuous_event');ae=h.get('activation_event')
                validate_mixed_events([ce],[ae],[event],exploration,1)
                if (generation==0 or config.categorical_policy not in ('mixed_reset','local_mixed','dependency_mixed') or config.method=='GA' and index==0) and (ce is not None or ae is not None):
                    raise ValueError('Unexpected continuous/activation event')
            from .local_variation import validate_event
            local=h.get('local_event')
            if index==0:batch_reference_id=None if previous is None else positions[previous['candidate_id']]
            eligible=config.update_scheme=='sparse_local' and generation>0 and not (config.method=='GA' and index==0)
            if eligible!=(local is not None):raise ValueError('Local event eligibility mismatch')
            if local is not None:
                validate_event(local,exploration,len(read_position(root,r)),config.block_step,config.subset_fraction,config.categorical_rate)
                if local['reference_genome_id']!=batch_reference_id:raise ValueError('Local context differs from the generation incumbent')
            variation=h.get('variation_event');restart=h.get('restart_event')
            from .dependency_variation import validate_event as validate_dependency,validate_restart
            if restart is not None:
                validate_restart(restart,config,generation,index)
                if any(x is not None for x in (event,ce,ae,local,variation)) or restart['proposed_genome_id']!=h['engine_proposed_genome_id']:raise ValueError('Inconsistent restart history')
            eligible=config.update_scheme in ('support_local','dependency') and generation>0 and not (index==0 and (config.method=='GA' or config.direction_source=='common')) and restart is None
            if eligible!=(variation is not None):raise ValueError('Dependency history eligibility mismatch')
            if variation is not None:
                validate_dependency(variation,dependencies,config,len(read_position(root,r)))
                if variation['proposal_genome_id']!=h['engine_proposed_genome_id'] or variation['reference_genome_id']!=positions[parents[i]]:raise ValueError('Dependency history parent mismatch')
            position=read_position(root,r)
            reference=None if previous is None else read_position(root,previous)
            measured=transitions(categorical,reference,position,position)
            saved=h.get('categorical_transitions')
            if not isinstance(saved,dict) or set(saved)!=set(measured):raise ValueError('Invalid categorical telemetry families')
            limits={row['name']:len(row['sizes']) for row in categorical}
            for name,values in measured.items():
                if not isinstance(saved[name],dict) or set(saved[name])!=set(values) or any(saved[name][k]!=v for k,v in values.items() if k!='proposed'):
                    raise ValueError('Categorical telemetry differs from evaluated positions')
                n=saved[name]['proposed']
                if type(n)!=int or not 0<=n<=limits[name]:raise ValueError('Invalid categorical proposal count')
        if any(h.get(k)!=v for k,v in expected.items()):raise ValueError('Checkpoint history differs from evaluated records')
        elapsed=h.get('elapsed_search_seconds')
        if isinstance(elapsed,bool) or not isinstance(elapsed,(int,float)) or not np.isfinite(elapsed) or elapsed<previous_elapsed:
            raise ValueError('Checkpoint history has inconsistent timing')
        previous_elapsed=elapsed;seen.add(cid)
    if seen!=set(cache):raise ValueError('Checkpoint history/cache coverage mismatch')


def _begin_invocation(root,signature,protocol_existed):
    """Persist before candidate work; interrupted cache hits also lose timing.

    An active marker survives an exception/process death. Unknown elapsed work
    cannot be reconstructed, so such a run remains usable as a candidate search
    but is permanently excluded from comparisons under a wall-time budget.
    """
    path=root/'invocation.json'
    if path.exists():
        state=unseal(path)
        if set(state)!={'version','signature','active','count','unclean','elapsed_search_seconds'} or state['version']!=1 or state['signature']!=signature:
            raise ValueError('Invalid invocation journal')
        if type(state['active'])!=bool or any(type(state[k])!=int or state[k]<0 for k in ('count','unclean')) or not np.isfinite(state['elapsed_search_seconds']) or state['elapsed_search_seconds']<0:
            raise ValueError('Invalid invocation timing')
        state['unclean']+=int(state['active']);state['count']+=1
    else:
        state=dict(version=1,signature=signature,active=False,count=1,unclean=int(protocol_existed),elapsed_search_seconds=0.)
    state['active']=True;seal(path,state,replace=True)
    return state


def _end_invocation(root,state,elapsed):
    state=dict(state,active=False,elapsed_search_seconds=elapsed)
    seal(root/'invocation.json',state,replace=True)


def run_search(study,config,output,policy=None,stop_after=None):
    """stop_after pauses after a number of calls in THIS invocation (test/manual).

    Wall budget includes construction, decode, AC, bookkeeping and checkpoint
    writes; it is checked between candidates, never interrupts an AC case.
    Publication audit is reported separately. No failed candidate is called feasible.
    """
    if 'physical' in study.electrical and config.coordination not in ('none','local_service'):raise ValueError('Legacy coordinated dispatch operators are not defined for the physical extension')
    if stop_after is not None and (type(stop_after)!=int or stop_after<1):raise ValueError('Invalid pause count')
    root=Path(output).resolve();policy=policy or Policy()
    with locked(root):
        study.check_unchanged();config=SearchConfig(**asdict(config))
        started=time.perf_counter();evaluator=Evaluator(study,policy);codec=Codec(study)
        from .dependency_variation import partition,prepare_restart
        dependencies=partition(codec) if config.update_scheme in ('support_local','dependency') else None
        initial_path=root/('initial.npz' if config.persistence=='compact' else 'initial.json')
        protocol=dict(version=2,config=asdict(config),policy=asdict(policy),input_id=study.input_id,
            evaluator_signature=evaluator.signature,code=source_hash(),dimension=codec.dimension,
            initialization_origin='catalogue_rules_and_seed_only',cache_scope='one_run',
            comparison='own_controls_direct_AC_no_redispatch_no_historical_start',
            financial_policy=resolve(study.c),
            comparator='feasible_selected_cost_then_violation_then_failed_fraction_and_measured_violation_then_structure')
        protocol_existed=(root/'protocol.json').exists()
        if protocol_existed:
            if unseal(root/'protocol.json')!=protocol:raise ValueError('Protocol/code/input differs; use a new output directory')
            snapshot=unseal(root/'study.json')
            if Study(**snapshot).input_id!=study.input_id:raise ValueError('Stored input differs')
        else:
            if any(p.name not in ('run.lock','protocol.json.pending','study.json.pending','study.json') for p in root.iterdir()):
                raise ValueError('Directory contains an incomplete or unrelated run; use a fresh directory')
            if (root/'study.json').exists():
                if Study(**unseal(root/'study.json')).input_id!=study.input_id:raise ValueError('Interrupted setup used different input')
            else:seal(root/'study.json',dict(data=study.data,electrical=study.electrical))
            seal(root/'protocol.json',protocol)
        if (root/'result.json').exists():
            saved=unseal(root/'result.json')
            if saved.get('financial_policy')!=protocol['financial_policy']:
                raise ValueError('Saved search financial policy mismatch')
            if saved['signature']!=evaluator.signature:raise ValueError('Saved search signature mismatch')
            if fingerprint(unseal(initial_path)['population'])!=saved['initial_id']:raise ValueError('Saved initial population changed')
            audited=_audit_best(root,saved['best'],evaluator.signature,codec)
            if (saved['best_feasible'] is not None)!=(saved['best'] is not None and saved['best']['record']['status']=='feasible') or saved['best_feasible'] is not None and saved['best_feasible']!=saved['best']:
                raise ValueError('Saved feasible winner mismatch')
            if fingerprint(saved['audited_metrics'])!=fingerprint(None if audited is None else audited['metrics']):raise ValueError('Saved search summary differs from its audited vector')
            return saved
        invocation=_begin_invocation(root,evaluator.signature,protocol_existed)
        if (root/'checkpoint.json').exists():
            pointer=unseal(root/'checkpoint.json')
            if type(pointer['slot'])!=int or pointer['slot'] not in (0,1):raise ValueError('Invalid checkpoint slot')
            suffix='npz' if config.persistence=='compact' else 'json'
            state=root/f"checkpoint_state_{pointer['slot']}.{suffix}"
            if sha(state)!=pointer['state_sha256']:raise ValueError('Checkpoint state changed')
            cp=unseal(state)
            if cp['version']!=1 or cp['signature']!=evaluator.signature:raise ValueError('Checkpoint signature mismatch')
            engine=Engine.from_dict(cp['engine'])
            if asdict(engine.config)!=asdict(config):raise ValueError('Checkpoint configuration mismatch')
            if engine.groups!=codec.search_groups():raise ValueError('Checkpoint coordinate groups mismatch')
            if engine.categorical!=codec.categorical_spec():raise ValueError('Checkpoint categorical domains mismatch')
            if engine.exploration!=codec.exploration_spec():raise ValueError('Checkpoint mixed domains mismatch')
            if engine.dependencies!=dependencies:raise ValueError('Checkpoint dependency partition mismatch')
            cache=cp['cache'];positions=cp['positions'];history=cp['history'];initial_id=cp['initial_id']
            elapsed_before=pointer['elapsed_search_seconds']
            if not np.isfinite(elapsed_before) or elapsed_before<0:raise ValueError('Invalid elapsed time')
            initial=unseal(initial_path)
            if fingerprint(initial['population'])!=initial_id:raise ValueError('Initial population changed')
            for candidate_id,record in cache.items():
                if candidate_id!=record['candidate_id']:raise ValueError('Cache identity mismatch')
                verify_record(root,record,evaluator.signature)
            for position_id,candidate_id in positions.items():
                if candidate_id not in cache or fingerprint(read_position(root,cache[candidate_id]))!=position_id:
                    raise ValueError('Position cache mismatch')
            # Validate every retained reference, including best, pbest and partial
            # generation, rather than trusting just the completed result envelope.
            for r in [*engine.records,*engine.personal_records,*engine.received,*([engine.best['record']] if engine.best else [])]:
                if r is not None and cache.get(r['candidate_id'])!=r:raise ValueError('Engine/cache record mismatch')
            if engine.best is not None:
                pos=read_position(root,engine.best['record'])
                if not np.array_equal(pos,engine.best['x']):raise ValueError('Checkpoint best position mismatch')
            _verify_engine_positions(root,engine)
            _verify_history(root,history,cache,config,engine.evaluations,codec.categorical_spec(),codec.exploration_spec(),dependencies)
            restart_numbers={h['restart_event']['number'] for h in history if h.get('restart_event') is not None}
            pending_numbers={e['number'] for e in engine.restart_pending if e is not None}
            if restart_numbers|pending_numbers!=set(range(1,engine.restart_count+1)):raise ValueError('Checkpoint restart history mismatch')
        else:
            if initial_path.exists():
                initial=unseal(initial_path);population=np.asarray(initial['population'],float)
                expected,seeds=common_population(codec,config)
                if not np.array_equal(population,expected) or initial['seeds']!=seeds:raise ValueError('Initial population mismatch')
            else:
                population,seeds=common_population(codec,config)
                seal(initial_path,dict(population=population,seeds=seeds))
            initial_id=fingerprint(population);engine=Engine(config,population,codec.search_groups(),codec.categorical_spec(),codec.exploration_spec(),dependencies)
            cache={};positions={};history=[];elapsed_before=0.
        elapsed_before=max(elapsed_before,invocation['elapsed_search_seconds'])
        # Resumed elapsed includes prior work; downtime is excluded. Verification
        # and loading in this invocation consume the same configured wall budget.
        def elapsed():return elapsed_before+time.perf_counter()-started
        calls=0
        while not engine.done and elapsed()<config.wall_seconds:
            iteration_started=time.perf_counter();phases=dict(decode_seconds=0.,evaluate_seconds=0.,save_seconds=0.,engine_seconds=0.)
            study.check_unchanged()
            if source_hash()!=evaluator.code:raise ValueError('Code changed during search')
            t=time.perf_counter();prepare_restart(engine,history,codec)
            phases['decode_seconds']+=time.perf_counter()-t
            # Restart construction is inside the wall budget. Do not start an
            # evaluation if that construction already exhausted it.
            if elapsed()>=config.wall_seconds:break
            raw=engine.ask();engine_raw_id=fingerprint(raw);coordination=None
            reference=None if engine.best is None else engine.best['x'].copy()
            previous_best_id=None if engine.best is None else engine.best['record']['candidate_id']
            categorical_event=copy.deepcopy(engine.categorical_pending[engine.cursor])
            continuous_event=copy.deepcopy(engine.continuous_pending[engine.cursor])
            activation_event=copy.deepcopy(engine.activation_pending[engine.cursor])
            local_event=copy.deepcopy(engine.local_pending[engine.cursor])
            variation_event=copy.deepcopy(engine.variation_pending[engine.cursor])
            restart_event=copy.deepcopy(engine.restart_pending[engine.cursor])
            # Preserve ordinary full-domain proposals on even indices. In block
            # mode only the operational group receives this declared variant.
            if config.coordination=='load_balance' and engine.generation>0 and engine.cursor%2==1 and engine.active_group() in (None,1):
                from .coordination import coordinate_load
                t=time.perf_counter();raw,coordination=coordinate_load(codec,raw,engine.best['x'])
                phases['decode_seconds']+=time.perf_counter()-t
            elif config.coordination=='local_service' and engine.generation>0 and engine.cursor%2==1 and engine.active_group() in (None,1):
                from .local_service import local_proposal
                ordinal=(engine.generation-1)*(engine.n//2)+engine.cursor//2
                t=time.perf_counter();raw,coordination=local_proposal(codec,raw,engine.best['x'],ordinal,config.block_step)
                phases['decode_seconds']+=time.perf_counter()-t
            elif config.coordination=='service' and engine.generation>0 and engine.cursor%2==1:
                from .service_moves import service_proposal
                ordinal=(engine.generation-1)*(engine.n//2)+engine.cursor//2
                t=time.perf_counter();raw,coordination=service_proposal(codec,raw,engine.best['x'],ordinal,config.block_step,engine.active_group())
                phases['decode_seconds']+=time.perf_counter()-t
            raw_id=fingerprint(raw)
            if raw_id in positions:
                # Already evaluated coordinates are not re-encoded/rounded again.
                candidate_id=positions[raw_id];record=cache[candidate_id]
                verify_record(root,record,evaluator.signature)
                position=raw;candidate=None;repairs={'known_position':True};hit=True
            else:
                t=time.perf_counter()
                decoded,repairs=codec.decode(raw);position=codec.encode(decoded)
                # Crucial: evaluate D(position), not a different pre-roundtrip
                # physical vector. All engines store this exact evaluated pair.
                candidate,final_repairs=codec.decode(position);candidate_id=candidate.candidate_id
                repairs=dict(proposal=repairs,final_decode=final_repairs)
                phases['decode_seconds']+=time.perf_counter()-t
                hit=candidate_id in cache
                if hit:
                    record=cache[candidate_id];verify_record(root,record,evaluator.signature)
                    # Many keys may describe the same physics. Retain the first
                    # evaluated canonical position for exact archive consistency.
                    position=np.asarray(read_position(root,record),float)
                else:
                    t=time.perf_counter()
                    result=evaluator.evaluate(candidate)
                    phases['evaluate_seconds']=time.perf_counter()-t
                    if result['candidate_id']!=candidate_id:raise ValueError('Decoder produced invalid structure')
                    # Uncommitted crash artifacts are preserved and not counted as
                    # evaluations; a retry writes a new attempt with a new name.
                    base=root/'evaluations'/f'{engine.evaluations:08d}'
                    attempt=0;folder=base.with_name(base.name+f'_{attempt:02d}')
                    while folder.exists():
                        attempt+=1;folder=base.with_name(base.name+f'_{attempt:02d}')
                    t=time.perf_counter();save_evaluation(folder,study,candidate,result,compact=config.persistence=='compact',shared_study=root/'study.json')
                    posname='position.npz' if config.persistence=='compact' else 'position.json'
                    if config.persistence=='compact':write_bundle(folder/posname,position)
                    else:write_new(folder/posname,position)
                    record=dict(directory=folder.relative_to(root).as_posix(),manifest_sha256=sha(folder/'manifest.json'),
                        position_sha256=sha(folder/posname),position_file=posname,candidate_id=candidate_id,status=result['status'],
                        plan_id=result['plan_id'],discrete_plan_id=discrete_id(candidate),
                        expected_cases=result['expected_cases'],approved_cases=result['approved_cases'],failed_cases=result['failed_cases'],
                        normalized_violations=result['normalized_violations'],
                        newton_iterations=sum(c.get('newton_iterations',0) for c in result['cases']),
                        sweep_iterations=sum(c.get('sweep_iterations',0) for c in result['cases']),
                        normalized_violation=result['normalized_violation'],
                        metrics=compact_metrics(result['metrics']))
                    cache[candidate_id]=record
                    phases['save_seconds']=time.perf_counter()-t
                positions[fingerprint(position)]=candidate_id
            generation=engine.generation;index=engine.cursor;active_group=engine.active_group()
            from .categorical import transitions
            categorical_transitions=transitions(engine.categorical,reference,raw,position)
            radius=engine.last_vortex_radius
            engine.set_budget_progress(min(1.,elapsed()/config.wall_seconds))
            t=time.perf_counter();engine.tell(position,record);calls+=1
            phases['engine_seconds']=time.perf_counter()-t
            history.append(dict(call=engine.evaluations,generation=generation,index=index,active_group=active_group,candidate_id=candidate_id,
                engine_proposed_genome_id=engine_raw_id,coordination=coordination,vortex_radius=radius,
                categorical_event=categorical_event,categorical_transitions=categorical_transitions,
                continuous_event=continuous_event,activation_event=activation_event,local_event=local_event,
                variation_event=variation_event,restart_event=restart_event,
                reference_best_id=previous_best_id,incumbent_improved=engine.best['record']['candidate_id']!=previous_best_id,
                proposed_genome_id=raw_id,evaluated_genome_id=fingerprint(position),cache_hit=hit,
                status=record['status'],objective_VP=None if record['metrics'] is None else record['metrics']['objective_VP'],
                selection_objective=None if record['metrics'] is None else record['metrics']['selection_objective'],
                financial_policy=None if record['metrics'] is None else record['metrics']['financial_policy'],
                violation=record['normalized_violation'],best_id=engine.best['record']['candidate_id'],
                best_status=engine.best['record']['status'],elapsed_search_seconds=elapsed(),repairs=repairs,
                phases=phases,plan_id=record['plan_id'],discrete_plan_id=record['discrete_plan_id'],
                failed_cases=record['failed_cases'],approved_cases=record['approved_cases'],
                normalized_violations=record['normalized_violations'],
                population_mean_std=float(np.mean(np.std(engine.population,axis=0))),
                iteration_before_checkpoint_seconds=time.perf_counter()-iteration_started))
            _checkpoint(root,engine,cache,positions,history,evaluator.signature,initial_id,elapsed)
            if stop_after is not None and calls>=stop_after and not engine.done:
                _end_invocation(root,invocation,elapsed())
                return dict(status='paused',evaluations=engine.evaluations,signature=evaluator.signature,
                            financial_policy=protocol['financial_policy'])
        search_elapsed=elapsed()
        audited=_audit_best(root,engine.best,evaluator.signature,codec)
        best=plain(engine.best)
        artifacts=list((root/'evaluations').iterdir()) if (root/'evaluations').exists() else []
        orphan_count=len(artifacts)-len(cache)
        phase_totals={name:sum(r['phases'][name] for r in history) for name in ('decode_seconds','evaluate_seconds','save_seconds','engine_seconds')}
        phase_totals['setup_checks_checkpoints_and_other_seconds']=max(0.,search_elapsed-sum(phase_totals.values()))
        final=dict(status='completed' if engine.done else 'wall_budget_exhausted',signature=evaluator.signature,
            financial_policy=protocol['financial_policy'],
            method=config.method,comparison_profile=config.comparison_profile,
            initial_id=initial_id,evaluations=engine.evaluations,unique_evaluations=len(cache),
            restarts=engine.restart_count,direction_source=config.direction_source,
            cache_hits=sum(r['cache_hit'] for r in history),elapsed_search_seconds=search_elapsed,
            wall_budget_seconds=config.wall_seconds,wall_overrun_seconds=max(0.,search_elapsed-config.wall_seconds),
            initial_zero_variance_coordinates=int(np.count_nonzero(np.ptp(np.asarray(unseal(initial_path)['population']),axis=0)==0.)),
            publication_audit_seconds=elapsed()-search_elapsed,best=best,
            best_feasible=best if best is not None and best['record']['status']=='feasible' else None,
            phase_seconds=phase_totals,process_peak_memory_bytes=peak_memory_bytes(),
            uncommitted_attempt_directories=orphan_count,unclean_invocations=invocation['unclean'],
            invocation_count=invocation['count'],timing_eligible=(orphan_count==0 and invocation['unclean']==0),
            elapsed_interpretation='recorded_search_time_lower_bound_after_unclean_interruption' if invocation['unclean'] else 'accumulated_search_time_excluding_downtime',
            first_feasible_seconds=next((r['elapsed_search_seconds'] for r in history if r['status']=='feasible'),None),
            unique_plans=len({r['plan_id'] for r in cache.values()}),unique_discrete_plans=len({r['discrete_plan_id'] for r in cache.values()}),
            newton_iterations_unique=sum(r['newton_iterations'] for r in cache.values()),
            sweep_iterations_unique=sum(r['sweep_iterations'] for r in cache.values()),
            audited_metrics=None if audited is None else audited['metrics'],history=history,
            conclusion='Budget completion is not an optimality or algorithm-superiority certificate')
        final=plain(final)
        seal(root/'result.json',final)
        _end_invocation(root,invocation,search_elapsed)
        return final
